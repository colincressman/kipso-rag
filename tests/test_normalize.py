import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.normalize.clean_markdown import clean_markdown


def test_clean_markdown_normalises_spacing_and_bullets():
    raw = "Title\n\n\n•   item one\nword-\n wrap\nA    B\n\n\n"
    cleaned = clean_markdown(raw)
    assert "- item one" in cleaned
    assert "wordwrap" in cleaned
    assert "A B" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_markdown_removes_bs_and_replacement_noise():
    raw = "Part I The Basics\x08����������������1\n"
    cleaned = clean_markdown(raw)
    assert "\x08" not in cleaned
    assert "�" not in cleaned
    assert "Part I The Basics" in cleaned


def test_clean_markdown_normalises_noisy_leaders():
    raw = "Chapter 1 Introduction����������������3\n"
    cleaned = clean_markdown(raw)
    assert "Chapter 1 Introduction ... 3" in cleaned

