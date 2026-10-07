import pytest
from opentelemetry.sdk.trace import TracerProvider

from gateway.config import Settings
from gateway.providers import Provider, ProviderError, ProviderResponse
from gateway.providers.fake import FakeProvider
from gateway.ratelimit import RateLimiter
from gateway.router import AllProvidersFailedError, Router, UnknownModelError
from gateway.schemas import ChatCompletionRequest
from tests.conftest import make_config


class FlakyThenOK(Provider):
    """Fails `failures` times with the given status, then succeeds."""

    system = "test"

    def __init__(self, name, failures, status=503):
        super().__init__(name)
        self.failures, self.status, self.calls = failures, status, 0

    async def complete(self, req, model):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError.from_status(self.status, "boom")
        return ProviderResponse("ok", "stop", model, 1, 1)

    async def stream(self, req, model):  # pragma: no cover - not used here
        raise NotImplementedError
        yield


REQ = ChatCompletionRequest(model="m-fallback", messages=[{"role": "user", "content": "hi"}])


def make_router(providers, max_retries=1):
    settings = Settings(_env_file=None, max_retries=max_retries, retry_backoff_seconds=0)
    return Router(make_config(), providers, settings, TracerProvider().get_tracer("t"))


async def test_retries_same_provider_before_falling_back():
    down, backup = FlakyThenOK("down", failures=1), FlakyThenOK("backup", failures=0)
    result = await make_router({"down": down, "backup": backup}).complete(REQ)
    assert result.provider == "down" and result.attempts == 2 and backup.calls == 0


async def test_falls_back_after_retries_exhausted():
    down, backup = FlakyThenOK("down", failures=99), FlakyThenOK("backup", failures=0)
    result = await make_router({"down": down, "backup": backup}).complete(REQ)
    assert result.provider == "backup" and down.calls == 2 and result.attempts == 3


async def test_auth_error_skips_retries_but_falls_back():
    down, backup = FlakyThenOK("down", failures=99, status=401), FlakyThenOK("backup", failures=0)
    result = await make_router({"down": down, "backup": backup}).complete(REQ)
    assert result.provider == "backup" and down.calls == 1


async def test_client_error_stops_immediately():
    down, backup = FlakyThenOK("down", failures=99, status=400), FlakyThenOK("backup", failures=0)
    with pytest.raises(AllProvidersFailedError) as e:
        await make_router({"down": down, "backup": backup}).complete(REQ)
    assert e.value.client_error and backup.calls == 0


async def test_all_providers_failing_raises():
    providers = {"down": FakeProvider("down", 0, 1.0), "backup": FakeProvider("backup", 0, 1.0)}
    with pytest.raises(AllProvidersFailedError) as e:
        await make_router(providers).complete(REQ)
    assert len(e.value.errors) == 4


async def test_streaming_falls_back_before_first_chunk():
    providers = {"down": FakeProvider("down", 0, 1.0), "backup": FakeProvider("backup", 0)}
    handle = await make_router(providers).open_stream(REQ)
    text = "".join([c.delta async for c in handle.chunks()])
    assert handle.provider == "backup" and text == "[backup] You said: hi"


def test_unknown_model():
    with pytest.raises(UnknownModelError):
        make_router({}).targets("nope")


def test_rate_limiter_token_bucket():
    now = [0.0]
    rl = RateLimiter(clock=lambda: now[0])
    assert [rl.check("t", 2)[0] for _ in range(3)] == [True, True, False]
    allowed, _, retry_after = rl.check("t", 2)
    assert not allowed and retry_after == pytest.approx(30, abs=0.1)  # 2/min -> 1 token per 30s
    now[0] = 30.0
    assert rl.check("t", 2)[0]
