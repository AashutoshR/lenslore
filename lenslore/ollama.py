"""Find the vision models installed in Ollama, which runs models on this computer."""

from __future__ import annotations

import json
import socket
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

# The IP address rather than "localhost": on Windows that name tries IPv6 first,
# where Ollama does not listen, adding two seconds to every request.
HOST = "http://127.0.0.1:11434"
BASE_URL = f"{HOST}/v1"  # Ollama's OpenAI-compatible endpoint
# Shown in the setup instructions. About 6.6 GB; fits a GPU with 8 GB of memory.
SUGGESTED_MODEL = "gemma4:e4b"


def _is_listening() -> bool:
    """Whether anything is on Ollama's port.

    Windows takes about two seconds to refuse a connection to a closed local
    port, which would stall every page load while Ollama is not running. An open
    local port answers at once, so a short timeout settles it either way.
    """
    address = urlsplit(HOST)
    try:
        with socket.create_connection((address.hostname, address.port), timeout=0.3):
            return True
    except OSError:
        return False


def _call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=3) as response:
        return json.load(response)


def vision_models() -> list[str] | None:
    """Names of installed models that accept images, or None if Ollama is not running."""
    if not _is_listening():
        return None
    try:
        installed = [model["name"] for model in _call("/api/tags").get("models", [])]
        return [
            name
            for name in installed
            if "vision" in _call("/api/show", {"model": name}).get("capabilities", [])
        ]
    except (OSError, ValueError):
        return None
