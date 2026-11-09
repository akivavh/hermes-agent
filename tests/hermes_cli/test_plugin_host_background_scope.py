"""Deferred native host callbacks keep their owner after multiplex activation."""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent import secret_scope
from hermes_cli.plugin_host import PluginHost
from hermes_constants import get_hermes_home, set_hermes_home_override
from tui_gateway import launch_profile_policy

KEY = "HERMES_CODEX_BASE_URL"
POLICY = contextvars.ContextVar("host_test_policy", default=None)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, owner, missing = [tmp_path / name for name in ("launch", "owner", "missing")]
    for home in (launch, owner, missing):
        home.mkdir()
    (launch / ".env").write_text(f"{KEY}=https://launch.fixture.invalid\n", encoding="utf-8")
    (owner / ".env").write_text(f"{KEY}=https://owner.fixture.invalid\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv(KEY, "https://ambient.fixture.invalid")
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    # Only synthetic local files: never consult a developer's managed/external secrets.
    import hermes_cli.env_loader
    import hermes_cli.managed_scope
    monkeypatch.setattr(hermes_cli.env_loader, "get_secret_source_values", lambda home: {})
    monkeypatch.setattr(hermes_cli.managed_scope, "load_managed_env", dict)
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    token = secret_scope.set_secret_scope({})
    try:
        yield launch, owner, missing
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous)


def host_for(home):
    return PluginHost(SimpleNamespace(home_path=home))


def observe(host, origin=None):
    return host._context_for_origin(origin).run(
        lambda: (get_hermes_home(), secret_scope.get_secret(KEY), POLICY.get()))


@pytest.mark.parametrize("created_after_activation", [False, True])
@pytest.mark.parametrize("expired_origin", [None, 17])
def test_delayed_callback_binds_owner_without_ended_caller(homes, created_after_activation, expired_origin):
    _, owner, _ = homes
    if created_after_activation:
        launch_profile_policy.activate_multi_profile_hosting()
    POLICY.set("ended caller approval")
    host = host_for(owner)
    if not created_after_activation:
        launch_profile_policy.activate_multi_profile_hosting()
    # The request no longer exists: the Channel has no context for its old ID.
    host._channel = SimpleNamespace(context_of=lambda origin: None)
    home, value, policy = observe(host, expired_origin)
    assert home == owner
    assert value == "https://owner.fixture.invalid"
    assert policy is None


@pytest.mark.parametrize("activated", [False, True])
def test_active_caller_keeps_its_scope_and_policy(homes, activated):
    launch, owner, _ = homes
    host = host_for(owner)
    if activated:
        launch_profile_policy.activate_multi_profile_hosting()
    caller = contextvars.Context()
    def bind():
        set_hermes_home_override(launch)
        secret_scope.set_secret_scope({KEY: "https://caller.fixture.invalid"}, profile_home=str(launch))
        POLICY.set("caller approval")
    caller.run(bind)
    host._channel = SimpleNamespace(context_of=lambda origin: caller if origin == 23 else None)
    assert observe(host, 23) == (launch, "https://caller.fixture.invalid", "caller approval")
    assert observe(host) == (owner, "https://owner.fixture.invalid", None)
    assert caller.run(POLICY.get) == "caller approval"


def test_concurrent_owners_and_missing_owner_never_borrow_ambient(homes):
    launch, owner, missing = homes
    hosts = [host_for(home) for home in homes]
    launch_profile_policy.activate_multi_profile_hosting()
    with ThreadPoolExecutor(max_workers=3) as executor:
        observed = list(executor.map(observe, hosts))
    assert observed == [(launch, "https://launch.fixture.invalid", None),
                        (owner, "https://owner.fixture.invalid", None), (missing, None, None)]
    assert secret_scope.current_secret_scope() == {}


def test_launch_owner_keeps_frozen_env_only_secret(homes, monkeypatch):
    launch, _, _ = homes
    (launch / ".env").unlink()
    host = host_for(launch)
    launch_profile_policy.activate_multi_profile_hosting()
    monkeypatch.setenv(KEY, "https://poisoned.fixture.invalid")
    assert observe(host) == (launch, "https://ambient.fixture.invalid", None)


@pytest.mark.parametrize("activated", [False, True])
@pytest.mark.parametrize("owner_index", [0, 1])
def test_callback_uses_canonical_profile_file_over_colliding_process_env(
    homes, monkeypatch, activated, owner_index,
):
    home = homes[owner_index]
    host = host_for(home)
    if activated:
        launch_profile_policy.activate_multi_profile_hosting()
    monkeypatch.setenv(KEY, "https://changed-process.fixture.invalid")
    # The canonical launch policy overlays profile files even before activation;
    # a secondary owner never inherits the launch process's credential at all.
    assert observe(host) == (
        home, f"https://{'launch' if owner_index == 0 else 'owner'}.fixture.invalid", None,
    )
    (home / ".env").write_text(f"{KEY}=https://updated-file.fixture.invalid\n", encoding="utf-8")
    assert observe(host) == (home, "https://updated-file.fixture.invalid", None)


def test_launch_env_only_callback_is_live_before_activation_and_frozen_after(homes, monkeypatch):
    launch, _, _ = homes
    (launch / ".env").unlink()
    host = host_for(launch)
    assert observe(host) == (launch, "https://ambient.fixture.invalid", None)
    monkeypatch.setenv(KEY, "https://live-launch.fixture.invalid")
    assert observe(host) == (launch, "https://live-launch.fixture.invalid", None)
    launch_profile_policy.activate_multi_profile_hosting()
    monkeypatch.setenv(KEY, "https://later-ambient.fixture.invalid")
    assert observe(host) == (launch, "https://live-launch.fixture.invalid", None)


def test_failed_owner_scope_does_not_fall_back_to_ambient(homes, monkeypatch):
    _, owner, _ = homes
    host = host_for(owner)
    launch_profile_policy.activate_multi_profile_hosting()
    def unavailable(home):
        raise OSError("synthetic scope unavailable")
    monkeypatch.setattr(launch_profile_policy, "owner_secret_scope", unavailable)
    with pytest.raises(OSError, match="synthetic scope unavailable"):
        observe(host)
    assert secret_scope.current_secret_scope() == {}


def test_host_restart_child_environment_uses_owner_after_activation(homes):
    _, owner, _ = homes
    host = host_for(owner)
    launch_profile_policy.activate_multi_profile_hosting()
    env = host._env()
    assert env["HERMES_HOME"] == str(owner)
    assert env[KEY] == "https://owner.fixture.invalid"


@pytest.mark.parametrize("owner_index, expected", [(1, "https://owner.fixture.invalid"), (2, None)])
def test_spontaneous_owner_before_activation_never_reads_launch_ambient(homes, owner_index, expected):
    home = homes[owner_index]
    host = host_for(home)
    assert not secret_scope.is_multiplex_active()
    assert observe(host) == (home, expected, None)
