"""Every error we raise descends from one root, so `except Exception` is never needed."""

import inspect

from researchmind.core import errors


def test_the_root_is_an_exception() -> None:
    assert issubclass(errors.ResearchmindError, Exception)


def test_the_domain_subtree_hangs_off_the_root() -> None:
    assert issubclass(errors.CoreError, errors.ResearchmindError)


def test_every_error_defined_here_descends_from_the_root() -> None:
    defined = [
        obj
        for _, obj in inspect.getmembers(errors, inspect.isclass)
        if issubclass(obj, BaseException) and obj.__module__ == errors.__name__
    ]
    assert defined, "the module should define at least the root"
    assert all(issubclass(cls, errors.ResearchmindError) for cls in defined)
