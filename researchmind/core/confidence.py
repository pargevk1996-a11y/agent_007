"""How sure the system is about something.

Three levels, not a probability. Language models are poorly calibrated on a continuous
scale: a returned ``0.83`` is not eighty-three percent, it is a number that looks like
one. Three ordered levels state the resolution we actually have.

Four levels were considered and rejected for the usual reason an even scale is preferred
over an odd one — with a middle available, the middle becomes where uncertain judgements
are parked, and the distinction stops carrying information. Three levels keep ``MEDIUM``
meaningful because ``LOW`` is not far away.

The enum is not a ``StrEnum``. Inheriting from ``str`` would bring string ordering with
it, and ``"high" < "low"`` is alphabetically true and semantically nonsense.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class Confidence(Enum):
    """An ordered confidence level: ``LOW < MEDIUM < HIGH``."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Return the position in the ordering, lowest first."""
        return _RANKS[self]

    def __lt__(self, other: object) -> bool:
        """Order by confidence, not by the spelling of the label."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        """Order by confidence, not by the spelling of the label."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        """Order by confidence, not by the spelling of the label."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        """Order by confidence, not by the spelling of the label."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank >= other.rank

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``high`` and not ``Confidence.HIGH``."""
        return self.value


_RANKS: Final[dict[Confidence, int]] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}
