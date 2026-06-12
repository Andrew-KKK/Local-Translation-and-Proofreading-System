import pytest

from novel_translator.gutenberg import (
    strip_gutenberg_boilerplate,
    suggested_filename,
)


def test_strips_gutenberg_header_and_footer():
    source = """license
*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
Chapter I
Story text.
*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
footer
"""
    assert strip_gutenberg_boilerplate(source) == "Chapter I\nStory text.\n"


def test_suggests_filename_from_url():
    assert suggested_filename("https://example.com/pg55.txt") == "pg55.txt"


def test_rejects_reversed_markers():
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate(
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK X ***"
        )
