import pytest

from policy_app.utils.text import (
    normalize_whitespace,
    lowercase,
    batch_items,
    tokenize,
    detokenize,
    unicode_normalize,
    remove_boilerplate,
    _normalize_token,
)


# ---- normalize_whitespace ----

def test_normalize_whitespace_collapses_spaces():
    assert normalize_whitespace("  hello   world  ") == "hello world"


def test_normalize_whitespace_tabs_newlines():
    assert normalize_whitespace("a\t\nb\r\nc") == "a b c"


def test_normalize_whitespace_empty():
    assert normalize_whitespace("") == ""


def test_normalize_whitespace_single_word():
    assert normalize_whitespace("  word  ") == "word"


# ---- lowercase ----

def test_lowercase():
    assert lowercase("Hello World") == "hello world"


def test_lowercase_already_lower():
    assert lowercase("abc") == "abc"


# ---- batch_items ----

def test_batch_items_even_split():
    items = ["a", "b", "c", "d"]
    batches = list(batch_items(items, 2))
    assert batches == [["a", "b"], ["c", "d"]]


def test_batch_items_uneven_split():
    items = ["a", "b", "c", "d", "e"]
    batches = list(batch_items(items, 2))
    assert batches == [["a", "b"], ["c", "d"], ["e"]]


def test_batch_items_larger_than_list():
    items = ["a", "b"]
    batches = list(batch_items(items, 10))
    assert batches == [["a", "b"]]


def test_batch_items_empty():
    batches = list(batch_items([], 5))
    assert batches == []


# ---- tokenize / detokenize ----

def test_tokenize_basic():
    assert tokenize("hello world foo") == ["hello", "world", "foo"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_detokenize_roundtrip():
    text = "hello world foo"
    assert detokenize(tokenize(text)) == text


# ---- unicode_normalize ----

def test_unicode_normalize_fullwidth():
    assert unicode_normalize("\uff21\uff22\uff23") == "ABC"


# ---- _normalize_token ----

def test_normalize_token_lowercase():
    assert _normalize_token("Hello") == "hello"


def test_normalize_token_strip_punctuation():
    assert _normalize_token("(word)") == "word"


def test_normalize_token_number_to_placeholder():
    assert _normalize_token("42") == "<NUM>"
    assert _normalize_token("3.14") == "<NUM>"


def test_normalize_token_empty_after_strip():
    assert _normalize_token("...") == ""


def test_normalize_token_smart_quotes():
    # \u2018 and \u2019 (left/right single quotes) get replaced with '
    result = _normalize_token("\u2018word\u2019")
    assert "word" in result


# ---- remove_boilerplate ----

def test_remove_boilerplate_empty():
    assert remove_boilerplate([]) == []


def test_remove_boilerplate_single_page():
    result = remove_boilerplate(["This is a single page of content."])
    assert len(result) == 1
    assert "single page" in result[0]


def test_remove_boilerplate_no_boilerplate():
    pages = [
        "First page has unique content about animals.",
        "Second page discusses different topics entirely.",
        "Third page covers something completely separate.",
    ]
    result = remove_boilerplate(pages)
    assert len(result) == 3
