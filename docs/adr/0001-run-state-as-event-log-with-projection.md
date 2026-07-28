# 1. Keep run state in an event log with a state projection

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 2 (schema), phase 8 (first real writer)
- **Related:** ADR-0007, ADR-0008, ADR-0010

## Context

A run must be reconstructible from its trace alone: every LLM call and tool call with its
tokens, latency, cost, provider, model, temperature and prompt version, correlated by
`run_id → step_id → call_id`. That is a hard requirement, not a debugging convenience —
it is what makes evaluation and cost attribution possible at all.

At the same time the control plane answers "what is this run doing right now" on every
SSE poll and every UI render. Those two demands pull in opposite directions: one wants an
immutable history, the other wants a cheap current answer.

## Decision

Both, with an explicit division of labour.

- `events` is append-only and immutable. It is the source of truth for the trace.
- `runs`, `sub_questions`, `plan_steps` are a projection of current state.
- **One transaction writes one event and the projection update it implies.** There is no
  asynchronous projector and no eventual consistency inside the database.
- Every write to the projection goes through a single repository module in `storage`. A
  direct `UPDATE` from anywhere else is a defect, caught in review and by the layering
  contracts.

## Alternatives considered

### Full event sourcing, with state derived by replay

Rejected: rebuilding run state on every status read puts replay on the hot path of the
event stream, and we would pay the cost of event schema versioning for a capability we do
not want. We inspect history; we never re-derive the present from it.

### A mutable state table with logs to a file or to OpenTelemetry

Rejected: it breaks the reconstruction requirement outright. Logs have no transactional
relationship with the state they describe, and they are sampled, rotated and lost.

## Consequences

- Roughly twice the write volume on the control path. At our rate — a handful of events
  per agent step — this is irrelevant.
- The projection can drift from the log if anyone bypasses the repository. That risk is
  concentrated in one module by design, so it is reviewable.
- Event payloads are large (see ADR-0010) and are stored apart from event metadata so
  that retention can be applied to the bulk without losing the trace skeleton.

## How to invalidate

If the projection tables turn out never to be read independently of the event stream —
the UI builds everything from events — then the projection is dead weight. Conversely, if
a third, fourth and fifth projection appear, we should have built a real projector and
moved to full event sourcing.
