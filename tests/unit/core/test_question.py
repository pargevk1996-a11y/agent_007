"""The bounds a research question is held to."""

import pytest
from pydantic import ValidationError

from researchmind.core.question import MAX_QUESTION_LENGTH, ResearchQuestion


def test_a_question_keeps_its_text() -> None:
    question = ResearchQuestion(text="How has Rust adoption in fintech backends evolved?")
    assert question.text == "How has Rust adoption in fintech backends evolved?"


def test_surrounding_whitespace_is_stripped_before_the_length_is_judged() -> None:
    assert ResearchQuestion(text="  a question  ").text == "a question"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_a_question_with_no_content_is_rejected(text: str) -> None:
    # Whitespace is stripped first, so a blank question fails as empty rather than being
    # stored as a string of spaces.
    with pytest.raises(ValidationError):
        ResearchQuestion(text=text)


def test_a_question_at_the_limit_is_accepted() -> None:
    assert len(ResearchQuestion(text="q" * MAX_QUESTION_LENGTH).text) == MAX_QUESTION_LENGTH


def test_a_question_past_the_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResearchQuestion(text="q" * (MAX_QUESTION_LENGTH + 1))
