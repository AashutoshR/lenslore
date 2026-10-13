"""The fallback between models in stream_answer, with the OpenAI client faked."""

from types import SimpleNamespace

import openai
import pytest

from lenslore import vision

# Stand-ins for the SDK's HTTP objects: its error classes only read these attributes.
REQUEST = SimpleNamespace(method="POST", url="https://example.test/chat/completions")
GEMINI_OVERLOAD = [{"error": {"code": 503, "message": "High demand.", "status": "UNAVAILABLE"}}]


def http_error(cls, status, body=None):
    response = SimpleNamespace(request=REQUEST, status_code=status, headers={})
    return cls("boom", response=response, body=body)


def chunks(*texts):
    for text in texts:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def failing_midway(text, exc):
    yield from chunks(text)
    raise exc


@pytest.fixture
def fake_client(monkeypatch):
    """Make each model behave as scripted: an exception to raise, or chunks to stream."""
    behaviour, tried = {}, []

    def create(model, messages, stream):
        tried.append(model)
        outcome = behaviour[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(vision.openai, "OpenAI", lambda **kwargs: client)
    return SimpleNamespace(behaviour=behaviour, tried=tried)


def ask(models, on_skip=None):
    return "".join(vision.stream_answer("key", None, models, [], on_skip=on_skip))


@pytest.mark.parametrize(
    "outcome, reason",
    [
        (http_error(openai.InternalServerError, 503, GEMINI_OVERLOAD), "is overloaded or out of quota"),
        (http_error(openai.RateLimitError, 429), "is overloaded or out of quota"),
        # Overload reported as an error event inside an HTTP 200 stream.
        (openai.APIError("boom", REQUEST, body=GEMINI_OVERLOAD[0]["error"]), "is overloaded or out of quota"),
        (openai.APITimeoutError(request=REQUEST), "took too long to respond"),
        (http_error(openai.NotFoundError, 404), "is not available"),
        (chunks(), "returned an empty answer"),
        (chunks(None, ""), "returned an empty answer"),
    ],
)
def test_unavailable_model_falls_back_to_the_next(fake_client, outcome, reason):
    fake_client.behaviour.update({"first": outcome, "second": chunks("A ", "cat.")})
    skipped = []

    answer = ask(["first", "second", "third"], on_skip=lambda *args: skipped.append(args))
    assert answer == "A cat."
    assert fake_client.tried == ["first", "second"]
    assert skipped == [("first", reason)]


def test_requests_are_given_a_time_limit(fake_client, monkeypatch):
    seen = {}
    client = vision.openai.OpenAI()
    monkeypatch.setattr(vision.openai, "OpenAI", lambda **kwargs: seen.update(kwargs) or client)
    fake_client.behaviour["first"] = chunks("ok")

    ask(["first"])
    assert seen["timeout"].read == 45.0
    assert seen["max_retries"] == 2

    # With fallbacks available, a slow model is not retried.
    fake_client.behaviour.update({"first": chunks("ok"), "second": chunks("ok")})
    ask(["first", "second"])
    assert seen["max_retries"] == 0


def test_error_once_every_model_is_busy(fake_client):
    busy = http_error(openai.InternalServerError, 503, GEMINI_OVERLOAD)
    fake_client.behaviour.update({"first": busy, "second": busy})

    with pytest.raises(vision.LensLoreError, match="`second` is overloaded or out of quota"):
        ask(["first", "second"])
    assert fake_client.tried == ["first", "second"]


def test_error_when_the_only_model_answers_with_nothing(fake_client):
    fake_client.behaviour["first"] = chunks()

    with pytest.raises(vision.LensLoreError, match="returned an empty answer"):
        ask(["first"])


def test_no_fallback_after_the_answer_has_started(fake_client):
    busy = http_error(openai.InternalServerError, 503, GEMINI_OVERLOAD)
    fake_client.behaviour.update({"first": failing_midway("A c", busy), "second": chunks("x")})

    with pytest.raises(vision.LensLoreError, match="mid-answer"):
        ask(["first", "second"])
    assert fake_client.tried == ["first"]


def test_other_errors_do_not_fall_back(fake_client):
    fake_client.behaviour["first"] = http_error(openai.AuthenticationError, 401)

    with pytest.raises(vision.LensLoreError, match="API key was rejected"):
        ask(["first", "second"])
    assert fake_client.tried == ["first"]
