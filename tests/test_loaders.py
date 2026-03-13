import os
import tempfile

import pytest

from policy_app.ingest.loaders import _safe_int_page, load_seed_policy_txt


# ---- _safe_int_page ----

def test_safe_int_page_with_int():
    assert _safe_int_page(5) == 5


def test_safe_int_page_with_digit_string():
    assert _safe_int_page("12") == 12


def test_safe_int_page_with_padded_string():
    assert _safe_int_page("  7  ") == 7


def test_safe_int_page_with_roman_numeral():
    assert _safe_int_page("iv") is None


def test_safe_int_page_with_compound():
    assert _safe_int_page("12 of 45") is None


def test_safe_int_page_with_none():
    assert _safe_int_page(None) is None


def test_safe_int_page_with_float():
    assert _safe_int_page(3.5) is None


# ---- load_seed_policy_txt ----

def test_load_seed_missing_file():
    chunks = load_seed_policy_txt("/nonexistent/path.txt")
    assert chunks == []


def test_load_seed_short_text_filtered():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("tiny")
        f.flush()
        path = f.name
    try:
        chunks = load_seed_policy_txt(path)
        assert chunks == []
    finally:
        os.unlink(path)


def test_load_seed_valid_text():
    content = (
        "This is a sufficiently long policy document that contains enough text "
        "to pass the minimum chunk character threshold. It describes the rules "
        "and regulations that employees must follow in the workplace. Additional "
        "sentences are included here to ensure the text exceeds the minimum "
        "chunk size of one hundred characters and produces at least one chunk."
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name
    try:
        chunks = load_seed_policy_txt(path)
        assert len(chunks) >= 1
        assert all(c.doc_name.endswith(".txt") for c in chunks)
        assert all(c.page is None for c in chunks)
        assert all(len(c.text) > 0 for c in chunks)
    finally:
        os.unlink(path)


def test_load_seed_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        f.flush()
        path = f.name
    try:
        chunks = load_seed_policy_txt(path)
        assert chunks == []
    finally:
        os.unlink(path)
