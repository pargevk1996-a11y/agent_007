"""The question a run exists to answer.

A single field with bounds is still worth a type. The question is what the planner
decomposes, what the synthesiser answers and what the evaluation harness scores against,
so it travels through every layer; a bare ``str`` would arrive at each of them
unvalidated and indistinguishable from a sub-question, a prompt or a title.

The upper bound is a decision, not a measurement. The text enters a prompt and a database
column, and an unbounded input is a cost and a storage failure mode rather than an
unusually thorough user. Raising the limit later costs nothing; discovering in production
that there was never one costs a truncated column.

What is deliberately absent: no tenant, no user, no timestamp. Those belong to the run
that asks the question, not to the question itself.
"""

from typing import Final

from pydantic import Field

from researchmind.core.base import DomainModel

MAX_QUESTION_LENGTH: Final = 2000
"""Longest question accepted, in characters."""


class ResearchQuestion(DomainModel):
    """An open-ended question submitted for research.

    ``DomainModel`` strips surrounding whitespace before the length constraints apply, so
    a question of nothing but spaces is rejected as empty rather than stored as blank.
    """

    text: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
