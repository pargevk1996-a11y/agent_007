# 7. Make the run a durable state machine so it can wait for a human

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 8 (machine), phase 14 (approval and cancel endpoints)
- **Related:** ADR-0001, ADR-0002, ADR-0004

## Context

No tool call happens before the user approves the plan. Hours can pass between the plan
being produced and someone looking at it. Whatever the worker does during those hours
defines the reliability of the whole system.

Cancellation has the same shape: the request arrives from the API process, while the work
is happening in a different process.

## Decision

A run is a state machine whose state lives in PostgreSQL:

```
created → planning → awaiting_plan_approval → executing → synthesizing → completed
                                    ↘ failed | cancelled | budget_exhausted
```

- On reaching `awaiting_plan_approval` the worker **finishes its task and frees the
  slot**. It holds nothing in memory.
- Approval writes a plan revision through the API and puts a message on the Redis stream.
  **Any** worker may pick the run up — not necessarily the one that planned it.
- Cancellation sets a flag in the database and signals the stream. The worker observes it
  at the next checkpoint — the same checkpoints as ADR-0004 — and unwinds its cancel scope.
- Cancellation semantics: completed steps stay completed and stay paid for; the current
  step is marked `cancelled`; **its partial result is not persisted as fact**, because an
  unverified fragment must never reach a report.

## Alternatives considered

### The worker awaits approval on a pub/sub subscription

Rejected: a deploy or a restart destroys every waiting run, an idle wait occupies a slot
for hours, and nothing survives a crash.

### Cancel by killing the worker process

Rejected: it takes neighbouring runs down with it and leaves transactions open.

## Consequences

- Every state transition is a database write plus a round trip through Redis. For short
  steps that overhead is visible; it is the price of surviving a restart.
- Cancellation is not instantaneous. It is bounded by the timeout of one tool call, and
  that bound is stated rather than implied.
- Workers are interchangeable, so scaling them is adding processes.

## How to invalidate

If approval in practice always happens within seconds — the user is watching the screen —
then durable suspension is over-engineering and a short in-memory wait with a timeout
would have done.
