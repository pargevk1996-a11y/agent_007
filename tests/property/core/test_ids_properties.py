"""Properties of minted identifiers, over batches rather than single values."""

from hypothesis import given
from hypothesis import strategies as st

from researchmind.core.ids import new_fact_id, new_run_id

BATCH_SIZES = st.integers(min_value=2, max_value=500)


@given(BATCH_SIZES)
def test_minted_identifiers_do_not_collide(count: int) -> None:
    minted = [new_run_id() for _ in range(count)]
    assert len(set(minted)) == count


@given(BATCH_SIZES)
def test_the_timestamp_prefix_never_goes_backwards(count: int) -> None:
    # The claim is deliberately weaker than "identifiers are sorted". UUIDv7 carries a
    # millisecond timestamp in its leading 48 bits and random data after it, so two
    # identifiers minted within the same millisecond have no defined order between them.
    # Anything needing a total order uses the per-run sequence number (ADR-0008).
    prefixes = [int.from_bytes(new_run_id().bytes[:6]) for _ in range(count)]
    assert prefixes == sorted(prefixes)


@given(BATCH_SIZES)
def test_identifiers_of_different_kinds_do_not_collide_either(count: int) -> None:
    runs = {new_run_id() for _ in range(count)}
    facts = {new_fact_id() for _ in range(count)}
    assert runs.isdisjoint(facts)
