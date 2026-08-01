"""One root over every exception the project defines, as `core.errors` promises.

That module states the guarantee — a caller may write `except ResearchmindError` and catch
everything we consider ours, while a genuine bug travels upward where it belongs — and
says a test asserts it. This is that test.

The guarantee is only worth anything if it holds at the edges. A single exception that
descends from bare `Exception` makes every `except ResearchmindError` in the codebase
quietly incomplete, and the failure shows up as an unhandled traceback in the one path
nobody exercised.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType

import researchmind
from researchmind.core.errors import ResearchmindError


def _modules() -> list[ModuleType]:
    """Import every module in the tree, so nothing escapes by not being imported already."""
    walked = [researchmind]
    walked.extend(
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(researchmind.__path__, prefix="researchmind.")
    )
    return walked


def _our_exceptions() -> dict[str, type[BaseException]]:
    """Collect the exception classes this project defines, keyed by qualified name.

    The filter on ``__module__`` is what keeps borrowed classes out. A module importing
    ``ValidationError`` exposes it as a member, and it is neither ours to constrain nor
    ours to reclassify.
    """
    collected: dict[str, type[BaseException]] = {}
    for module in _modules():
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseException) and obj.__module__.startswith("researchmind"):
                collected[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return collected


def test_every_exception_we_define_descends_from_the_root() -> None:
    stray = sorted(
        name for name, obj in _our_exceptions().items() if not issubclass(obj, ResearchmindError)
    )
    assert stray == []


def test_the_search_reaches_every_corner_of_the_tree() -> None:
    # "Every member of the empty set satisfies it" is true and worthless. These three are
    # a root, a class nested deep inside core, and one from another package entirely, so
    # a walk that quietly stopped early cannot leave the test above passing.
    found = set(_our_exceptions())
    assert "researchmind.core.errors.ResearchmindError" in found
    assert "researchmind.core.report.ReferentialIntegrityError" in found
    assert "researchmind.providers.errors.ProviderTimeoutError" in found
