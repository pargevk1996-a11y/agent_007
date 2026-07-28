"""Exact monetary amounts.

Costs in this system are small and numerous: a single token can be priced around three
millionths of a dollar, and a run makes thousands of calls. Two failure modes follow, and
both are designed out here rather than watched for.

Binary floating point cannot represent those amounts exactly, so it is not used. Amounts
are integers of nanodollars — a thousand millionths of a dollar — which leaves three
orders of magnitude below any per-token price we expect to meet.

Unit confusion is the other failure mode. Dollars, cents, nanodollars and token counts all
look alike as plain integers, and adding the wrong pair produces a plausible number rather
than an error. Wrapping the amount in a type makes that mistake fail to typecheck.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from pydantic import Field

from researchmind.core.base import DomainModel

NANODOLLARS_PER_DOLLAR = 1_000_000_000
"""One dollar, in nanodollars."""

_NANO = Decimal("1E-9")

# The persisted representation is a PostgreSQL bigint, so an amount outside its range
# could be constructed but never stored. Bounding the value object surfaces that at
# construction rather than at INSERT. Roughly plus or minus 9.2 billion dollars.
_MIN_NANODOLLARS = -(2**63)
_MAX_NANODOLLARS = 2**63 - 1


class Money(DomainModel):
    """A signed monetary amount, exact to the nanodollar.

    Negative amounts are legitimate: budget reservations are released by subtraction
    (ADR-0004), and a released reservation is a negative movement.
    """

    nanodollars: int = Field(ge=_MIN_NANODOLLARS, le=_MAX_NANODOLLARS)

    @classmethod
    def from_dollars(cls, amount: Decimal | int | str) -> Money:
        """Build an amount from dollars.

        ``float`` is excluded from the signature on purpose, so passing one fails to
        typecheck: a float would silently reintroduce the representation error this type
        exists to avoid. Callers holding a float convert explicitly, through
        ``Decimal(str(x))``, and thereby state which decimal value they meant. Fractions
        below a nanodollar
        are rounded half to even, the unbiased default for accounting. Code that needs a
        conservative upper bound — budget reservation, for one — rounds up explicitly at
        the call site rather than relying on this.
        """
        nanos = (Decimal(amount) * NANODOLLARS_PER_DOLLAR).quantize(
            Decimal(1), rounding=ROUND_HALF_EVEN
        )
        return cls(nanodollars=int(nanos))

    def as_dollars(self) -> Decimal:
        """Return the amount in dollars, exactly, with nine decimal places."""
        # scaleb shifts the exponent rather than dividing, so no precision is lost.
        return Decimal(self.nanodollars).scaleb(-9).quantize(_NANO)

    def __add__(self, other: Money) -> Money:
        """Add two amounts."""
        return Money(nanodollars=self.nanodollars + other.nanodollars)

    def __sub__(self, other: Money) -> Money:
        """Subtract one amount from another."""
        return Money(nanodollars=self.nanodollars - other.nanodollars)

    def __mul__(self, factor: int) -> Money:
        """Multiply an amount by a whole number, such as a token count."""
        return Money(nanodollars=self.nanodollars * factor)

    def __lt__(self, other: object) -> bool:
        """Order by amount."""
        if not isinstance(other, Money):
            return NotImplemented
        return self.nanodollars < other.nanodollars

    def __le__(self, other: object) -> bool:
        """Order by amount."""
        if not isinstance(other, Money):
            return NotImplemented
        return self.nanodollars <= other.nanodollars

    def __gt__(self, other: object) -> bool:
        """Order by amount."""
        if not isinstance(other, Money):
            return NotImplemented
        return self.nanodollars > other.nanodollars

    def __ge__(self, other: object) -> bool:
        """Order by amount."""
        if not isinstance(other, Money):
            return NotImplemented
        return self.nanodollars >= other.nanodollars

    def __str__(self) -> str:
        """Render as dollars without trailing zeros, for logs and reports."""
        return f"${self.as_dollars().normalize():f}"
