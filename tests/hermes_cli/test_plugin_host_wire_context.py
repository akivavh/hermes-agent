"""Context preparation must answer real RPCs without executing an unscoped handler."""

import contextvars
import json
import socket
import threading
from concurrent.futures import Future
from uuid import uuid4

import pytest

from hermes_cli.plugin_host_wire import (
    Channel,
    PluginHostError,
    PluginHostUnavailable,
    serving_request,
)

POLICY = contextvars.ContextVar("wire_test_policy", default=None)


class RecordingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.frames = []

    def write(self, data):
        result = self.stream.write(data)
        self.frames.append(json.loads(data))
        return result

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


class LocalPair:
    """Test-only sockets; the production request has no timeout or provider."""

    def __init__(self, handler, resolver, caller_handler, caller_resolver):
        self.prefix = f"wire-context-{uuid4().hex}"
        self.sockets = socket.socketpair()
        self.writers = [RecordingWriter(sock.makefile("wb")) for sock in self.sockets]
        handlers = (caller_handler, handler)
        resolvers = (caller_resolver, resolver)
        self.channels = [
            Channel(sock.makefile("rb"), writer, callback,
                    name=f"{self.prefix}-{index}", context_for_origin=context_for_origin)
            for index, (sock, writer, callback, context_for_origin)
            in enumerate(zip(self.sockets, self.writers, handlers, resolvers))
        ]
        self.workers = []
        for channel in self.channels:
            channel.start()

    def pending_request(self, method="work", params=None):
        future = Future()
        caller = contextvars.copy_context()

        def invoke():
            try:
                # Deliberately use the production default wait (timeout=None).
                future.set_result(self.channels[0].request(method, params or {}))
            except BaseException as exc:
                future.set_exception(exc)

        worker = threading.Thread(target=caller.run, args=(invoke,),
                                  name=f"{self.prefix}-watchdog-request", daemon=True)
        self.workers.append(worker)
        worker.start()
        return future

    def request(self, method="work", params=None):
        # Only the test supervisor has a deadline. Teardown releases a stuck RPC.
        return self.pending_request(method, params).result(timeout=5)

    def close(self):
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for channel in self.channels:
            channel.close("local test teardown")
        for sock in self.sockets:
            sock.close()
        threads = self.workers + [
            thread for thread in threading.enumerate()
            if thread.name.startswith(self.prefix) and thread not in self.workers
        ]
        for thread in threads:
            thread.join(timeout=5)
        assert not [thread.name for thread in threads if thread.is_alive()]


@pytest.fixture
def local_pair():
    pairs = []

    def create(handler, resolver=None, *, caller_handler=None, caller_resolver=None):
        pair = LocalPair(handler, resolver, caller_handler, caller_resolver)
        pairs.append(pair)
        return pair

    yield create
    for pair in pairs:
        pair.close()


@pytest.mark.parametrize("failure", [PermissionError("scope unavailable"), SystemExit(23)])
def test_resolver_failure_replies_with_same_id_and_next_request_works(local_pair, failure):
    attempts, handled = [], []

    def resolve(origin):
        attempts.append(origin)
        if len(attempts) == 1:
            raise failure
        return contextvars.Context()

    def handler(method, params, origin):
        handled.append(method)
        return "healthy"

    pair = local_pair(handler, resolve)
    reply = pair.request()
    assert reply["id"] == pair.writers[0].frames[0]["id"]
    assert reply["error"]["type"] == type(failure).__name__
    assert handled == []
    expected = PermissionError if isinstance(failure, PermissionError) else PluginHostError
    with pytest.raises(expected) as error:
        pair.channels[0].result_of(reply)
    if isinstance(error.value, PluginHostError):
        assert error.value.type_name == "SystemExit"
    assert pair.channels[0].result_of(pair.request("next")) == "healthy"
    assert handled == ["next"]


@pytest.mark.parametrize("empty_context, expected", [(True, None), (False, "ambient approval")])
def test_empty_context_is_valid_and_only_none_uses_ambient(local_pair, empty_context, expected):
    clean = contextvars.Context()

    def resolve(origin):
        # Contaminate the actual serving thread, not just the test's caller.
        POLICY.set("ambient approval")
        return clean if empty_context else None

    pair = local_pair(lambda *args: [POLICY.get(), serving_request()], resolve)
    reply = pair.request()
    assert pair.channels[0].result_of(reply) == [expected, reply["id"]]
    assert clean.run(serving_request) is None


def test_already_entered_context_replies_without_handler_and_recovers(local_pair):
    context = contextvars.Context()
    context.run(POLICY.set, "owner approval")
    entered, release = threading.Event(), threading.Event()
    handled = []

    def hold():
        entered.set()
        assert release.wait(10), "test did not release the held context"

    holder = threading.Thread(target=context.run, args=(hold,), daemon=True)
    holder.start()
    pair = local_pair(lambda *args: handled.append(POLICY.get()) or "healthy",
                      lambda origin: context)
    try:
        assert entered.wait(5)
        reply = pair.request()
        assert reply["id"] == pair.writers[0].frames[0]["id"]
        with pytest.raises(RuntimeError, match="already entered"):
            pair.channels[0].result_of(reply)
        assert handled == []
    finally:
        release.set()
        holder.join(timeout=5)
        assert not holder.is_alive()
    assert pair.channels[0].result_of(pair.request("next")) == "healthy"
    assert handled == ["owner approval"]
    assert context.run(serving_request) is None


def test_nested_origin_rpc_preserves_caller_policy_and_serving_ids(local_pair):
    owner = contextvars.Context()
    owner.run(POLICY.set, "owner approval")
    observed = []

    def server_context(origin):
        caller = pair.channels[1].context_of(origin)
        return caller.copy() if caller is not None else owner.copy()

    def caller_context(origin):
        return pair.channels[0].context_of(origin).copy()

    def server(method, params, origin):
        observed.append([method, POLICY.get(), origin, serving_request()])
        if method == "outer":
            return pair.channels[1].call("callback", {})
        return "nested result"

    def caller(method, params, origin):
        observed.append([method, POLICY.get(), origin, serving_request()])
        return pair.channels[0].call("leaf", {})

    pair = local_pair(server, server_context, caller_handler=caller, caller_resolver=caller_context)
    token = POLICY.set("caller approval")
    try:
        assert pair.channels[0].result_of(pair.request("outer")) == "nested result"
        assert POLICY.get() == "caller approval"
    finally:
        POLICY.reset(token)
    outer, leaf = [frame for frame in pair.writers[0].frames if "method" in frame]
    callback = next(frame for frame in pair.writers[1].frames if "method" in frame)
    assert observed == [
        ["outer", "owner approval", None, outer["id"]],
        ["callback", "caller approval", outer["id"], callback["id"]],
        ["leaf", "owner approval", callback["id"], leaf["id"]],
    ]
    assert owner.run(serving_request) is None


def test_failed_notification_does_not_reply_or_poison_next_request(local_pair):
    ready = threading.Event()
    serving_threads, handled = [], []

    def resolve(origin):
        if not serving_threads:
            serving_threads.append(threading.current_thread())
            ready.set()
            raise PermissionError("notification scope unavailable")
        return contextvars.Context()

    pair = local_pair(lambda method, *args: handled.append(method) or "healthy", resolve)
    pair.channels[0]._send({"id": None, "method": "notice", "params": {}, "origin": None})
    assert ready.wait(5)
    serving_threads[0].join(timeout=5)
    assert not serving_threads[0].is_alive()
    assert pair.writers[1].frames == []
    assert handled == []
    assert pair.channels[0].result_of(pair.request("next")) == "healthy"
    assert handled == ["next"]


@pytest.mark.parametrize("failure", [ValueError("handler error"), SystemExit(17)])
def test_handler_errors_keep_existing_wire_contract(local_pair, failure):
    def handler(*args):
        raise failure

    pair = local_pair(handler, lambda origin: contextvars.Context())
    reply = pair.request()
    assert reply["id"] == pair.writers[0].frames[0]["id"]
    assert reply["error"]["type"] == type(failure).__name__
    with pytest.raises(ValueError if isinstance(failure, ValueError) else PluginHostError):
        pair.channels[0].result_of(reply)


def test_unserializable_result_uses_existing_type_error_reply(local_pair):
    pair = local_pair(lambda *args: object(), lambda origin: contextvars.Context())
    reply = pair.request()
    assert reply["id"] == pair.writers[0].frames[0]["id"]
    with pytest.raises(TypeError, match="not JSON serializable"):
        pair.channels[0].result_of(reply)


def test_transport_close_releases_default_waiting_request(local_pair):
    entered, release = threading.Event(), threading.Event()

    def handler(*args):
        entered.set()
        assert release.wait(5)
        return "late"

    pair = local_pair(handler)
    pending = pair.pending_request()
    try:
        assert entered.wait(5)
        pair.sockets[1].shutdown(socket.SHUT_RDWR)
        with pytest.raises(PluginHostUnavailable):
            pending.result(timeout=5)
    finally:
        release.set()
