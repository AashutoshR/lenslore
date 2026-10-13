"""Drive the Streamlit UI with a stubbed model, so no network or API key is needed."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from lenslore import ollama, vision
from lenslore.prompts import QUICK_PROMPTS
from lenslore.providers import PROVIDERS

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def calls(monkeypatch):
    """Replace the model call with a stub and record what the app sends it."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    # Behave as if Ollama is not installed, whatever this machine has.
    monkeypatch.setattr(ollama, "vision_models", lambda: None)
    st.cache_data.clear()
    recorded = []

    def fake_stream(api_key, base_url, models, messages, on_skip=None, timeout=None):
        recorded.append(
            {
                "api_key": api_key,
                "base_url": base_url,
                "models": models,
                "messages": messages,
                "timeout": timeout,
            }
        )
        yield "It is "
        yield "a receipt."

    monkeypatch.setattr(vision, "stream_answer", fake_stream)
    return recorded


def run_app():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at


def app_with_sample():
    at = run_app()
    at.button_group[0].set_value("Sample").run()
    assert not at.exception
    return at


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def conversation(at):
    return [(turn["role"], turn["content"]) for turn in at.session_state.history]


def test_questions_need_an_image_first(calls):
    at = run_app()
    assert all(widget(at.button, label).disabled for label in QUICK_PROMPTS)

    at.chat_input[0].set_value("What is this?").run()
    assert not at.exception
    assert calls == []
    assert [t.value for t in at.toast] == ["Add an image first."]


def test_quick_prompt_then_follow_up_question(calls):
    at = app_with_sample()
    widget(at.button, "Read text").click().run()
    at.chat_input[0].set_value("What was the total?").run()

    assert not at.exception
    assert conversation(at) == [
        ("user", QUICK_PROMPTS["Read text"]),
        ("assistant", "It is a receipt."),
        ("user", "What was the total?"),
        ("assistant", "It is a receipt."),
    ]
    # The follow-up carries the whole conversation, with the image on turn one only.
    follow_up = calls[1]["messages"]
    assert [m["role"] for m in follow_up] == ["system", "user", "assistant", "user"]
    assert follow_up[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert follow_up[3]["content"] == "What was the total?"
    # Bookkeeping about the answer is not sent back to the model.
    assert follow_up[2] == {"role": "assistant", "content": "It is a receipt."}


def test_answers_are_labelled_with_the_model_that_gave_them(calls, monkeypatch):
    def skipping_stream(api_key, base_url, models, messages, on_skip=None, timeout=None):
        on_skip(models[0], "is overloaded or out of quota")
        yield "A receipt."

    monkeypatch.setattr(vision, "stream_answer", skipping_stream)
    at = app_with_sample()
    at.chat_input[0].set_value("What is this?").run()

    gemini = PROVIDERS["Gemini"]
    assert at.session_state.history[-1]["model"] == gemini.models[1]
    captions = [c.value for c in at.caption]
    assert f"`{gemini.models[0]}` is overloaded or out of quota, trying the next model…" in captions
    assert any(c.startswith(f"{gemini.models[1]} · ") for c in captions)


def test_free_gemini_provider_is_the_default(calls):
    at = app_with_sample()
    at.chat_input[0].set_value("What is this?").run()

    gemini = PROVIDERS["Gemini"]
    assert calls[0]["api_key"] == "gemini-test-key"
    assert calls[0]["base_url"] == gemini.base_url
    # The chosen model first, then the other free models as fallbacks.
    assert calls[0]["models"] == list(gemini.models)
    # `detail` is an OpenAI-only field, so it must not be sent to Gemini.
    assert calls[0]["messages"][1]["content"][1]["image_url"].keys() == {"url"}


def test_switching_to_openai_uses_its_key_endpoint_and_detail_setting(calls):
    at = app_with_sample()
    widget(at.radio, "Provider").set_value("OpenAI").run()
    widget(at.radio, "Image detail").set_value("high").run()
    at.chat_input[0].set_value("What is this?").run()

    assert not at.exception
    assert calls[0]["api_key"] == "openai-test-key"
    assert calls[0]["base_url"] is None
    # No automatic fallback on a paid provider.
    assert calls[0]["models"] == ["gpt-4o"]
    assert calls[0]["messages"][1]["content"][1]["image_url"]["detail"] == "high"


def test_ollama_explains_setup_when_it_is_not_running(calls):
    at = run_app()
    widget(at.radio, "Provider").set_value("Ollama").run()

    assert not at.exception
    assert "Ollama is not running on this computer." in [w.value for w in at.warning]
    assert [c.value for c in at.code] == [f"ollama pull {ollama.SUGGESTED_MODEL}"]
    assert at.chat_input[0].disabled


def test_ollama_explains_setup_when_it_has_no_vision_model(calls, monkeypatch):
    monkeypatch.setattr(ollama, "vision_models", lambda: [])
    at = run_app()
    widget(at.radio, "Provider").set_value("Ollama").run()

    assert "Ollama is running, but none of its models can read images." in [
        w.value for w in at.warning
    ]
    assert at.chat_input[0].disabled


def test_installed_ollama_model_is_preferred_and_used_locally(calls, monkeypatch):
    monkeypatch.setattr(ollama, "vision_models", lambda: ["gemma4:e4b", "llama3.2-vision:latest"])
    at = app_with_sample()

    assert widget(at.radio, "Provider").value == "Ollama"
    # Ollama cannot fetch images from the web, so that source is not offered.
    assert at.button_group[0].options == ["Upload", "Sample"]

    widget(at.selectbox, "Model").set_value("llama3.2-vision:latest").run()
    at.chat_input[0].set_value("What is this?").run()

    assert not at.exception
    assert calls[0]["base_url"] == "http://127.0.0.1:11434/v1"
    assert calls[0]["models"] == ["llama3.2-vision:latest"]
    assert calls[0]["timeout"] == 300.0
    assert calls[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg")


def test_hosted_copy_does_not_offer_ollama(calls, monkeypatch):
    monkeypatch.setenv("LENSLORE_HOSTED", "true")
    # Even if an Ollama happens to be reachable from the server, it is not the visitor's.
    monkeypatch.setattr(ollama, "vision_models", lambda: ["gemma4:e4b"])
    at = run_app()

    provider = widget(at.radio, "Provider")
    assert provider.options == ["Gemini", "OpenAI"]
    assert provider.value == "Gemini"


def test_switching_image_starts_a_new_conversation(calls):
    at = app_with_sample()
    at.chat_input[0].set_value("What is this?").run()
    assert len(at.session_state.history) == 2

    samples = widget(at.selectbox, "Sample image")
    samples.select_index(1).run()
    assert at.session_state.history == []


def test_new_conversation_button_clears_the_chat(calls):
    at = app_with_sample()
    at.chat_input[0].set_value("What is this?").run()
    widget(at.button, "New conversation").click().run()
    assert at.session_state.history == []


def test_api_failure_is_shown_and_question_is_dropped(calls, monkeypatch):
    def failing_stream(*args, **kwargs):
        raise vision.LensLoreError("The API key was rejected. Check it and try again.")
        yield

    monkeypatch.setattr(vision, "stream_answer", failing_stream)
    at = app_with_sample()
    at.chat_input[0].set_value("What is this?").run()

    assert not at.exception
    assert [e.value for e in at.error] == ["The API key was rejected. Check it and try again."]
    assert at.session_state.history == []
