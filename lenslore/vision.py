"""Image preparation and vision-model calls for LensLore."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from urllib.parse import urlparse

import openai
from PIL import Image, ImageOps, UnidentifiedImageError

from .prompts import STYLES, SYSTEM_PROMPT

# Vision models scale large images down anyway, so sending more only costs upload time.
MAX_SIDE = 2048
JPEG_QUALITY = 90
# Overloaded servers sometimes accept a request and then stall; the SDK default is 10 minutes.
TIMEOUT = 45.0  # seconds

log = logging.getLogger(__name__)


class LensLoreError(Exception):
    """An error whose message is safe to show to the user."""


@dataclass(frozen=True)
class PreparedImage:
    url: str  # data: URL or public http(s) URL, exactly as sent to the model
    key: str  # stable id, used to reset the chat when the image changes
    size: tuple[int, int] | None = None  # (width, height) as sent; unknown for remote URLs


def prepare_image(raw: bytes) -> PreparedImage:
    """Normalise an uploaded image to a right-way-up JPEG no larger than MAX_SIDE."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        if img.mode in ("RGBA", "LA", "P"):
            # JPEG has no alpha channel; flatten onto white rather than black.
            img = img.convert("RGBA")
            flat = Image.new("RGB", img.size, "white")
            flat.paste(img, mask=img.getchannel("A"))
            img = flat
        else:
            img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise LensLoreError("That file could not be read as an image.") from exc

    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return PreparedImage(
        url=f"data:image/jpeg;base64,{encoded}",
        key=hashlib.sha1(raw).hexdigest(),
        size=img.size,
    )


def image_from_url(url: str) -> PreparedImage:
    """Reference a public image by URL. The provider fetches it; this app never does."""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LensLoreError("Enter a full image URL starting with http:// or https://.")
    return PreparedImage(url=url, key=hashlib.sha1(url.encode()).hexdigest())


def build_messages(
    image: PreparedImage, history: list[dict], style: str, detail: str | None = None
) -> list[dict]:
    """Turn the chat history into a Chat Completions payload.

    `history` alternates user/assistant text turns, starting with the user. The
    image is attached once, to the first user turn, so follow-up questions do
    not re-send it. `detail` is OpenAI-specific; leave it None for other providers.
    """
    messages: list[dict] = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{STYLES[style]}"}]
    for i, turn in enumerate(history):
        content = turn["content"]
        if i == 0:
            image_part = {"url": image.url}
            if detail:
                image_part["detail"] = detail
            content = [
                {"type": "text", "text": turn["content"]},
                {"type": "image_url", "image_url": image_part},
            ]
        messages.append({"role": turn["role"], "content": content})
    return messages


def _error_body(exc: openai.APIError) -> dict:
    """The provider's error object, without the SDK's or Gemini's wrapping around it."""
    body = exc.body
    if isinstance(body, list) and body:  # Gemini wraps its error object in a list
        body = body[0]
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return error
    return {}


def _reason(exc: openai.APIError) -> str:
    return _error_body(exc).get("message") or exc.message


def _is_busy(exc: openai.APIError) -> bool:
    """Whether the model is overloaded or out of quota, so another model may still work."""
    if isinstance(exc, (openai.RateLimitError, openai.InternalServerError)):
        return True
    # Gemini can also report overload as an error event inside an HTTP 200 stream.
    body = _error_body(exc)
    return body.get("code") in (429, 500, 503) or body.get("status") in (
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
    )


def stream_answer(
    api_key: str,
    base_url: str | None,
    models: list[str],
    messages: list[dict],
    on_skip: Callable[[str, str], None] | None = None,
    timeout: float = TIMEOUT,
) -> Iterator[str]:
    """Yield the answer piece by piece. Raises LensLoreError on API failure.

    `models` is tried in order. A model that is overloaded, out of quota, missing,
    too slow, or that answers with nothing is reported through
    `on_skip(model, reason)` and the next one is used instead.
    """
    # With fallbacks, moving to the next model beats retrying a struggling one.
    retries = 0 if len(models) > 1 else 2
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=retries,
        timeout=openai.Timeout(timeout, connect=10.0),
    )
    for position, model in enumerate(models):
        started = False
        cause = None
        try:
            stream = client.chat.completions.create(model=model, messages=messages, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    started = True
                    yield chunk.choices[0].delta.content
            if started:
                return
            reason = "returned an empty answer"
        except openai.AuthenticationError as exc:
            raise LensLoreError("The API key was rejected. Check it and try again.") from exc
        except openai.BadRequestError as exc:
            raise LensLoreError(f"The request was rejected: {_reason(exc)}") from exc
        except openai.NotFoundError as exc:
            cause, reason = exc, "is not available"
        except openai.APITimeoutError as exc:
            cause, reason = exc, "took too long to respond"
        except openai.APIConnectionError as exc:
            raise LensLoreError(
                "Could not reach the model. Check your internet connection, "
                "or that Ollama is running if you are using it."
            ) from exc
        except openai.APIError as exc:
            if not _is_busy(exc):
                raise LensLoreError(f"The provider returned an error: {_reason(exc)}") from exc
            cause, reason = exc, "is overloaded or out of quota"

        log.warning("%s %s%s", model, reason, f": {_reason(cause)}" if cause else "")
        if started:
            raise LensLoreError(f"`{model}` failed mid-answer. Ask again.") from cause
        if position == len(models) - 1:
            raise LensLoreError(f"`{model}` {reason}. Try again in a minute.") from cause
        if on_skip:
            on_skip(model, reason)
