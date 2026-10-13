"""Model discovery against a stand-in Ollama server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lenslore import ollama

CAPABILITIES = {
    "gemma4:e4b": ["completion", "vision"],
    "nomic-embed-text:latest": ["embedding"],
    "llama3.2-vision:latest": ["completion", "vision"],
    "old-model:latest": None,  # older Ollama builds omit the field
}


class FakeOllama(BaseHTTPRequestHandler):
    def reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/api/tags"
        self.reply({"models": [{"name": name} for name in self.server.installed]})

    def do_POST(self):
        assert self.path == "/api/show"
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        capabilities = CAPABILITIES[request["model"]]
        self.reply({} if capabilities is None else {"capabilities": capabilities})

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), FakeOllama)
    httpd.installed = list(CAPABILITIES)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(ollama, "HOST", f"http://127.0.0.1:{httpd.server_port}")
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_only_models_that_accept_images_are_listed(server):
    assert ollama.vision_models() == ["gemma4:e4b", "llama3.2-vision:latest"]


def test_empty_list_when_nothing_is_installed(server):
    server.installed = []
    assert ollama.vision_models() == []


def test_none_when_ollama_is_not_running(server, monkeypatch):
    port = server.server_port
    server.shutdown()
    server.server_close()
    monkeypatch.setattr(ollama, "HOST", f"http://127.0.0.1:{port}")
    assert ollama.vision_models() is None
