"""Identifiers are UUIDv7, minted here, and only for the entities we own."""

from uuid import UUID

from researchmind.core import ids

FACTORIES = (
    ids.new_run_id,
    ids.new_plan_id,
    ids.new_sub_question_id,
    ids.new_source_id,
    ids.new_fact_id,
    ids.new_claim_id,
    ids.new_call_id,
)


def test_every_factory_produces_a_standard_library_uuid() -> None:
    # uuid_utils returns its own Rust-backed class from its top-level module, which
    # pydantic rejects. Everything here must go through the compat module.
    for factory in FACTORIES:
        assert type(factory()) is UUID


def test_every_factory_produces_version_7() -> None:
    for factory in FACTORIES:
        assert factory().version == 7


def test_identities_we_receive_have_no_factory() -> None:
    # Tenants and users are authenticated elsewhere and arrive already identified.
    # A factory here would mean the domain had invented an identity.
    assert not hasattr(ids, "new_tenant_id")
    assert not hasattr(ids, "new_user_id")
