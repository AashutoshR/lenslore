"""The model providers LensLore can talk to. All speak the OpenAI Chat Completions format."""

from __future__ import annotations

from dataclasses import dataclass

from . import ollama


@dataclass(frozen=True)
class Provider:
    blurb: str  # one-line summary shown under the provider's name
    privacy: str  # where the user's images go
    base_url: str | None  # None means the OpenAI SDK's default endpoint
    models: tuple[str, ...] = ()  # suggestions only; the UI accepts any model id
    key_env: str | None = None  # environment variable / Streamlit secret holding the API key
    key_url: str = ""  # where the user creates a key
    # Runs on this computer through Ollama: no key, and models are whatever is installed.
    local: bool = False
    supports_detail: bool = False  # whether image_url takes OpenAI's `detail` field
    accepts_image_urls: bool = True  # whether the provider will fetch an image from a URL
    # Whether to move on to the next model when the chosen one is busy. Only safe
    # where every model is free, since the next model may be a pricier one.
    auto_fallback: bool = False
    timeout: float = 45.0  # seconds to wait for the model before giving up on it


PROVIDERS = {
    "Gemini": Provider(
        blurb="Free · Google's cloud · daily limits",
        privacy="Images are sent to Google, which may use free-tier content to improve its products.",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # Ordered by how the free tier behaved when tested: the first answered in
        # seconds; the last two were prone to stalling, so they are tried last.
        models=(
            "gemini-3-flash-preview",
            "gemini-3.8-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
        ),
        key_env="GEMINI_API_KEY",
        key_url="https://aistudio.google.com/apikey",
        auto_fallback=True,
    ),
    "Ollama": Provider(
        blurb="Free · runs on this computer · private",
        privacy="Images never leave this computer.",
        base_url=ollama.BASE_URL,
        local=True,
        accepts_image_urls=False,
        # Loading a model into memory for the first question can take minutes.
        timeout=300.0,
    ),
    "OpenAI": Provider(
        blurb="Paid · OpenAI's cloud",
        privacy="Images are sent to OpenAI.",
        base_url=None,
        # "GPT-4V" (gpt-4-vision-preview) was retired; these are its successors.
        models=("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"),
        key_env="OPENAI_API_KEY",
        key_url="https://platform.openai.com/api-keys",
        supports_detail=True,
    ),
}
