"""Make model output safe to hand to Streamlit's Markdown renderer."""

from __future__ import annotations

import re

# A fenced block (closed, or still open while the answer streams in) or an inline code span.
_CODE = re.compile(r"(```.*?(?:```|\Z)|`[^`\n]*`)", re.DOTALL)


def escape_dollars(text: str) -> str:
    """Escape `$` outside code, so prices like "$5 and $7" are not rendered as LaTeX."""
    parts = _CODE.split(text)
    # re.split with one capture group alternates: text, code, text, code, ...
    return "".join(
        part if i % 2 else part.replace("$", "\\$") for i, part in enumerate(parts)
    )
