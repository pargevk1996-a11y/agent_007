"""Append-only run events: types, PostgreSQL persistence, Redis Streams fan-out.

Events are the source of truth for the trace, so a run is reconstructible from them
alone. Publication to Redis happens only after the owning transaction commits.
"""
