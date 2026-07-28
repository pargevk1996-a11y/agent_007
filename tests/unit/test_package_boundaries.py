"""The package tree is the architecture, so its shape is asserted rather than assumed.

Adding or removing a package is an architectural act. These tests make it a deliberate
one: the expected set below has to be edited by hand, in the same commit.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType

import researchmind

EXPECTED_PACKAGES = frozenset(
    {
        "api",
        "budget",
        "config",
        "core",
        "critic",
        "eval",
        "events",
        "executor",
        "memory",
        "obs",
        "planner",
        "providers",
        "sdk",
        "storage",
        "synthesizer",
        "tools",
        "worker",
    }
)


def _subpackages() -> dict[str, ModuleType]:
    """Import every direct subpackage of researchmind, keyed by its short name."""
    return {
        info.name: importlib.import_module(f"researchmind.{info.name}")
        for info in pkgutil.iter_modules(researchmind.__path__)
    }


def test_package_set_matches_the_declared_architecture() -> None:
    assert set(_subpackages()) == set(EXPECTED_PACKAGES)


def test_every_package_documents_its_boundary() -> None:
    undocumented = sorted(
        name for name, module in _subpackages().items() if not (module.__doc__ or "").strip()
    )
    assert undocumented == []
    assert (researchmind.__doc__ or "").strip()


def test_package_ships_type_information() -> None:
    assert (Path(researchmind.__path__[0]) / "py.typed").is_file()
