"""LensLore: show it an image and ask questions about it.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import NamedTuple

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from lenslore import ollama
from lenslore.markdown import escape_dollars
from lenslore.prompts import QUICK_PROMPTS, STYLES
from lenslore.providers import PROVIDERS, Provider
from lenslore.vision import (
    LensLoreError,
    PreparedImage,
    build_messages,
    image_from_url,
    prepare_image,
    stream_answer,
)

load_dotenv()

SAMPLES_DIR = Path(__file__).parent / "samples"
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "bmp"]
AVATARS = {"user": ":material/person:", "assistant": "🔎"}
QUICK_ICONS = {
    "Describe": ":material/image_search:",
    "Read text": ":material/notes:",
    "Explain data": ":material/bar_chart:",
    "Alt text": ":material/accessibility_new:",
}
GETTING_STARTED = """
#### Start with an image

1. **Add an image**: upload one, try a sample, or paste one into the message box below.
2. **Ask a question**, or use one of the quick prompts.
3. **Follow up**. LensLore keeps the whole conversation in mind.
"""

st.set_page_config(page_title="LensLore", page_icon="🔎", layout="wide")
# Keep tall images from pushing the conversation off the screen.
st.html(
    """<style>
    [data-testid="stImage"] img {
        max-height: 60vh; width: auto; max-width: 100%; margin: 0 auto; object-fit: contain;
    }
    </style>"""
)


class Settings(NamedTuple):
    provider_name: str
    provider: Provider
    api_key: str | None
    model: str | None
    style: str
    detail: str | None


def configured_api_key(name: str) -> str | None:
    """API key from the environment/.env, or from Streamlit secrets when deployed."""
    if key := os.getenv(name):
        return key
    try:
        return st.secrets.get(name)
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return None


@st.cache_data(show_spinner=False)
def cached_prepare(raw: bytes) -> PreparedImage:
    return prepare_image(raw)


@st.cache_data(ttl=15, show_spinner=False)
def installed_vision_models() -> list[str] | None:
    return ollama.vision_models()


def pick_local_model(models: list[str] | None) -> str | None:
    """Choose among Ollama's installed vision models, or explain how to get one."""
    if models:
        return st.selectbox(
            "Model",
            models,
            key="model_Ollama",
            help="Vision models installed in Ollama. Add more with `ollama pull`.",
        )
    if models is None:
        st.warning("Ollama is not running on this computer.")
        st.markdown(
            "1. Install it from [ollama.com/download](https://ollama.com/download).\n"
            "2. Download a vision model by running this in a terminal:"
        )
    else:
        st.warning("Ollama is running, but none of its models can read images.")
        st.markdown("Download a vision model by running this in a terminal:")
    st.code(f"ollama pull {ollama.SUGGESTED_MODEL}", language="bash")
    if st.button("Check again", icon=":material/refresh:"):
        installed_vision_models.clear()
        st.rerun()
    return None


def sidebar() -> Settings:
    with st.sidebar:
        st.subheader("Model")
        # A hosted copy (see render.yaml) cannot reach an Ollama on the visitor's computer.
        hosted = bool(os.getenv("LENSLORE_HOSTED"))
        local_models = None if hosted else installed_vision_models()
        names = [n for n in PROVIDERS if not (hosted and PROVIDERS[n].local)]
        name = st.radio(
            "Provider",
            names,
            # Prefer the local model once it is set up: no limits, nothing leaves the computer.
            index=names.index("Ollama") if local_models else 0,
            captions=[PROVIDERS[n].blurb for n in names],
        )
        provider = PROVIDERS[name]
        if provider.local:
            api_key = "ollama"  # the SDK insists on a key; Ollama ignores it
            model = pick_local_model(local_models)
        else:
            api_key = configured_api_key(provider.key_env)
            if api_key:
                st.caption(f":material/check_circle: Key loaded from `{provider.key_env}`")
            else:
                api_key = st.text_input(
                    "API key",
                    type="password",
                    key=f"key_{name}",
                    help=f"Kept in this browser session only. Set {provider.key_env} in .env to skip this.",
                )
                st.caption(f"[Get a {name} key]({provider.key_url})")
            model = st.selectbox(
                "Model",
                provider.models,
                accept_new_options=True,
                key=f"model_{name}",
                help="Any vision-capable model from this provider. Type a model id to use one not listed.",
            )

        st.subheader("Answer")
        style = st.selectbox("Style", list(STYLES), help="How long and how technical answers are.")
        detail = None
        if provider.supports_detail:
            detail = st.radio(
                "Image detail",
                ["auto", "low", "high"],
                horizontal=True,
                help="`low` is faster and cheaper; `high` reads small text and fine detail better.",
            )

        st.divider()
        if st.button("New conversation", icon=":material/restart_alt:", width="stretch"):
            st.session_state.history = []
        st.caption(f":material/lock: {provider.privacy}")
    return Settings(name, provider, api_key, model, style, detail)


def pick_image(accepts_urls: bool) -> tuple[PreparedImage, bytes | str] | None:
    """Render the image picker. Returns the image for the model plus a preview source."""
    sources = ["Upload", "Sample"] + (["URL"] if accepts_urls else [])
    source = st.segmented_control(
        "Image source", sources, default="Upload", required=True, label_visibility="collapsed"
    )
    if source == "Upload":
        file = st.file_uploader("Upload an image", type=IMAGE_TYPES, label_visibility="collapsed")
        if file is None:
            return None
        raw = file.getvalue()
        return cached_prepare(raw), raw
    if source == "Sample":
        samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lstrip(".") in IMAGE_TYPES)
        if not samples:
            st.info("No sample images found. Run `python scripts/make_samples.py`.")
            return None
        path = st.selectbox(
            "Sample image",
            samples,
            format_func=lambda p: p.stem.replace("_", " ").capitalize(),
            label_visibility="collapsed",
        )
        raw = path.read_bytes()
        return cached_prepare(raw), raw
    url = st.text_input(
        "Image URL", placeholder="https://example.com/photo.jpg", label_visibility="collapsed"
    )
    if not url:
        return None
    return image_from_url(url), url.strip()


def take_pasted_image() -> None:
    """Make an image pasted or dropped into the message box the current image.

    Runs as the message box's callback, before the page is redrawn, so the new
    image is already in place when the picker and the conversation render.
    """
    submission = st.session_state.message
    if not submission.files:
        return
    raw = submission.files[0].getvalue()
    try:
        st.session_state.current = (cached_prepare(raw), raw)
    except LensLoreError as exc:
        st.session_state.paste_error = str(exc)
    else:
        st.session_state.pending_question = submission.text or QUICK_PROMPTS["Describe"]


def render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"], avatar=AVATARS[turn["role"]]):
        st.markdown(escape_dollars(turn["content"]))
        if "model" in turn:
            st.caption(f"{turn['model']} · {turn['seconds']:.1f} s")


def answer(question: str, image: PreparedImage, history: list[dict], settings: Settings) -> None:
    """Ask the model and stream its reply into the current container."""
    provider = settings.provider
    models = [settings.model]
    if provider.auto_fallback:
        models += [m for m in provider.models if m != settings.model]
    skipped = []

    def on_skip(model: str, why: str) -> None:
        skipped.append(model)
        st.caption(f"`{model}` {why}, trying the next model…")

    history.append({"role": "user", "content": question})
    render_turn(history[-1])
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        waiting = "Looking at the image…"
        if provider.local:
            waiting += " The first answer is slower while the model loads."
        started = time.perf_counter()
        text = ""
        try:
            with st.spinner(waiting):
                reply = st.empty()
                for piece in stream_answer(
                    settings.api_key,
                    provider.base_url,
                    models,
                    build_messages(image, history, settings.style, settings.detail),
                    on_skip=on_skip,
                    timeout=provider.timeout,
                ):
                    text += piece
                    reply.markdown(escape_dollars(text) + " ▌")
            reply.markdown(escape_dollars(text))
        except LensLoreError as exc:
            # Drop the unanswered question so the next attempt starts clean.
            history.pop()
            st.error(str(exc))
            return
        turn = {
            "role": "assistant",
            "content": text,
            # Models are tried in order, so the one that answered follows the skipped ones.
            "model": models[len(skipped)],
            "seconds": time.perf_counter() - started,
        }
        history.append(turn)
        st.caption(f"{turn['model']} · {turn['seconds']:.1f} s")


settings = sidebar()
ready = bool(settings.api_key and settings.model)

st.title("🔎 LensLore")
st.caption("Show it an image, then ask anything about it.")

image_col, chat_col = st.columns([5, 7], gap="large")

with image_col, st.container(border=True):
    st.markdown("**Image**")
    picked = None
    try:
        picked = pick_image(settings.provider.accepts_image_urls)
    except LensLoreError as exc:
        st.error(str(exc))
    if paste_error := st.session_state.pop("paste_error", None):
        st.error(paste_error)
    # The picker replaces the current image only when its own choice changes, so an
    # image pasted into the message box stays until the user picks something else.
    picker_key = picked[0].key if picked else None
    if picker_key != st.session_state.get("picker_key"):
        st.session_state.picker_key = picker_key
        st.session_state.current = picked
    current = st.session_state.get("current")
    if current:
        image, preview = current
        st.image(preview, width="stretch")
        if image.size:
            st.caption(f"Sent to the model at {image.size[0]} × {image.size[1]} px")

with chat_col, st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**Conversation**")
        if ready:
            st.badge(f"{settings.provider_name} · {settings.model}", icon=":material/bolt:", color="green")
        else:
            st.badge("Not set up", icon=":material/settings:", color="orange")

    # A new image starts a new conversation.
    image_key = current[0].key if current else None
    if st.session_state.get("image_key") != image_key:
        st.session_state.image_key = image_key
        st.session_state.history = []
    history: list[dict] = st.session_state.setdefault("history", [])

    transcript = st.container(height=440)
    with transcript:
        hint = st.empty()
        if not current:
            hint.markdown(GETTING_STARTED)
        elif not history:
            hint.caption("Ask anything about this image, or start with a quick prompt below.")
        for turn in history:
            render_turn(turn)

    if not ready:
        if settings.provider.local:
            st.warning("Set up Ollama in the sidebar to ask questions.")
        else:
            st.warning("Add your API key in the sidebar to ask questions.")

    question = st.session_state.pop("pending_question", None)
    with st.container(horizontal=True):
        for label, prompt in QUICK_PROMPTS.items():
            if st.button(
                label, icon=QUICK_ICONS[label], help=prompt, disabled=not (ready and current)
            ):
                question = prompt
    submission = st.chat_input(
        "Ask about the image, or paste a new one here…",
        key="message",
        accept_file=True,
        file_type=IMAGE_TYPES,
        disabled=not ready,
        on_submit=take_pasted_image,
    )
    if submission and not submission.files:
        question = submission.text

    if question and not current:
        st.toast("Add an image first.", icon=":material/image:")
    elif question:
        hint.empty()
        with transcript:
            answer(question, current[0], history, settings)
