"""The root of every error this system raises.

One root, then a subtree per module: ``ProviderError``, ``ToolError``, ``BudgetError``
and so on each descend from :class:`ResearchmindError` and are defined by the module that
raises them. Two things follow. A caller can write ``except ResearchmindError`` and catch
everything we consider ours while letting genuine bugs — a ``TypeError``, a
``KeyError`` — travel upward where they belong. And ``except Exception`` never needs to
appear in this codebase.

A test in the suite asserts the first half of that: every exception defined anywhere in
the project descends from this root.
"""


class ResearchmindError(Exception):
    """Base class for every error raised by researchmind.

    Subtrees add their own structure — a provider error carries the provider and the
    status, a budget error carries the scope that was exhausted. This root deliberately
    carries nothing, so that catching it never implies a payload that may not be there.
    """


class CoreError(ResearchmindError):
    """Base class for errors raised by the domain layer.

    Most failures in ``core`` surface as pydantic validation errors rather than as this,
    because they are violations of a model's contract. This exists for the failures that
    are not: invariants that span more than one object and cannot be expressed as a field
    constraint.
    """
