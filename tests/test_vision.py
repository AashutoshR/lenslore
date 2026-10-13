import base64
import io

import pytest
from PIL import Image

from lenslore.prompts import STYLES, SYSTEM_PROMPT
from lenslore.vision import MAX_SIDE, LensLoreError, build_messages, image_from_url, prepare_image


def png_bytes(size, color, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def decode(prepared):
    header, encoded = prepared.url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_large_image_is_downscaled_keeping_aspect_ratio():
    prepared = prepare_image(png_bytes((4096, 1024), "red"))
    assert prepared.size == (MAX_SIDE, 512)
    assert decode(prepared).size == (MAX_SIDE, 512)


def test_small_image_is_not_upscaled():
    assert prepare_image(png_bytes((64, 48), "red")).size == (64, 48)


def test_transparency_is_flattened_onto_white():
    prepared = prepare_image(png_bytes((10, 10), (0, 0, 0, 0), mode="RGBA"))
    assert decode(prepared).getpixel((5, 5)) == (255, 255, 255)


def test_key_depends_on_image_content():
    red, blue = png_bytes((8, 8), "red"), png_bytes((8, 8), "blue")
    assert prepare_image(red).key == prepare_image(red).key
    assert prepare_image(red).key != prepare_image(blue).key


def test_non_image_bytes_are_rejected():
    with pytest.raises(LensLoreError):
        prepare_image(b"definitely not an image")


@pytest.mark.parametrize("bad", ["", "example.com/cat.jpg", "ftp://example.com/cat.jpg", "file:///etc/passwd"])
def test_only_http_urls_are_accepted(bad):
    with pytest.raises(LensLoreError):
        image_from_url(bad)


def test_url_image_is_passed_through_untouched():
    image = image_from_url("  https://example.com/cat.jpg ")
    assert image.url == "https://example.com/cat.jpg"
    assert image.size is None


def test_detail_is_omitted_unless_requested():
    image = image_from_url("https://example.com/cat.jpg")
    messages = build_messages(image, [{"role": "user", "content": "What is this?"}], "Concise")
    assert messages[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.jpg"},
    }


def test_image_is_attached_to_the_first_user_turn_only():
    image = image_from_url("https://example.com/cat.jpg")
    history = [
        {"role": "user", "content": "What is this?"},
        {"role": "assistant", "content": "A cat."},
        {"role": "user", "content": "What colour?"},
    ]
    system, first, reply, follow_up = build_messages(image, history, "Concise", detail="low")

    assert system == {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{STYLES['Concise']}"}
    assert first["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg", "detail": "low"}},
    ]
    assert reply == {"role": "assistant", "content": "A cat."}
    assert follow_up == {"role": "user", "content": "What colour?"}
