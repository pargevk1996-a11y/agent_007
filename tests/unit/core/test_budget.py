"""What counts as a budget, and what only looks like one."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from researchmind.core.budget import Budget, BudgetScopeKind
from researchmind.core.money import Money


def test_a_budget_may_set_one_limit_of_any_kind() -> None:
    assert Budget(scope=BudgetScopeKind.RUN, max_tokens=100_000).max_spend is None
    assert (
        Budget(scope=BudgetScopeKind.CALL, max_spend=Money.from_dollars("0.50")).max_tokens is None
    )
    assert (
        Budget(scope=BudgetScopeKind.SUB_QUESTION, max_duration=timedelta(minutes=5)).max_tokens
        is None
    )


def test_a_budget_may_set_all_three() -> None:
    budget = Budget(
        scope=BudgetScopeKind.RUN,
        max_tokens=1_000_000,
        max_spend=Money.from_dollars("5"),
        max_duration=timedelta(minutes=30),
    )
    assert budget.max_tokens == 1_000_000
    assert budget.max_duration == timedelta(minutes=30)


def test_a_budget_that_bounds_nothing_is_rejected() -> None:
    # The paperwork of control without the fact of it.
    with pytest.raises(ValidationError, match="at least one of"):
        Budget(scope=BudgetScopeKind.RUN)


@pytest.mark.parametrize("tokens", [0, -1])
def test_a_token_limit_must_leave_something_to_spend(tokens: int) -> None:
    with pytest.raises(ValidationError):
        Budget(scope=BudgetScopeKind.RUN, max_tokens=tokens)


@pytest.mark.parametrize("dollars", ["0", "-1"])
def test_a_spending_limit_must_be_positive(dollars: str) -> None:
    # Money itself allows zero and negatives, because a released reservation is a negative
    # movement. A ceiling is not a movement.
    with pytest.raises(ValidationError, match="greater than zero"):
        Budget(scope=BudgetScopeKind.RUN, max_spend=Money.from_dollars(dollars))


@pytest.mark.parametrize("duration", [timedelta(0), timedelta(seconds=-1)])
def test_a_duration_limit_must_be_positive(duration: timedelta) -> None:
    with pytest.raises(ValidationError):
        Budget(scope=BudgetScopeKind.RUN, max_duration=duration)


def test_the_three_scopes_are_the_ones_the_decision_names() -> None:
    assert {str(scope) for scope in BudgetScopeKind} == {"call", "sub_question", "run"}


def test_a_budget_decides_nothing() -> None:
    # Reserving, clamping and committing are the enforcer's job in phase 4, where there is
    # state. This test exists so that adding a decision here is a deliberate act.
    budget = Budget(scope=BudgetScopeKind.RUN, max_tokens=100)
    assert not hasattr(budget, "permits")
    assert not hasattr(budget, "remaining")
