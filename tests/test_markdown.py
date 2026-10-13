from lenslore.markdown import escape_dollars


def test_prices_are_escaped_so_they_are_not_read_as_latex():
    assert escape_dollars("Subtotal $26.95, tax $2.16.") == "Subtotal \\$26.95, tax \\$2.16."


def test_code_is_left_alone():
    text = "Total `$29.11`:\n```\nTOTAL   $29.11\n```\nThat is $29.11."
    assert escape_dollars(text) == "Total `$29.11`:\n```\nTOTAL   $29.11\n```\nThat is \\$29.11."


def test_code_block_still_being_streamed_is_left_alone():
    assert escape_dollars("Cost $5:\n```\nTOTAL $29") == "Cost \\$5:\n```\nTOTAL $29"


def test_text_without_dollars_is_unchanged():
    assert escape_dollars("A cat on a mat.") == "A cat on a mat."
