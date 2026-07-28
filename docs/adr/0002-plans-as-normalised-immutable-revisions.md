# 2. Store plans as normalised rows in immutable revisions

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 7
- **Related:** ADR-0001, ADR-0007

## Context

The plan is the artefact the user inspects, edits or rejects before execution spends
anything on tools. It is also the spine of the trace: a sub-question identifier appears
in every event, every cost row and every extracted fact.

Two properties follow. The plan must be addressable at step granularity, and a step must
never change meaning after events have pointed at it.

## Decision

- `plans(id, run_id, revision, status, created_by)` where `created_by` is `planner` or
  `user`; `plan_steps(id, plan_id, ordinal, sub_question_text, rationale, expected_tools,
  depends_on)`.
- **Editing a plan never mutates it.** A user edit creates `revision = N + 1`. Earlier
  revisions are kept forever. Only the revision with `status = approved` executes.
- In memory a plan is a nested Pydantic model; `storage` assembles and disassembles it.
- `depends_on` is a list of step identifiers. Acyclicity is checked during plan
  validation and covered by a property-based test.

## Alternatives considered

### A single `plans.plan_json` column

Rejected: phase 16 and 17 ask questions like "across all runs, what did the steps that
failed have in common". Against JSONB that is either unindexable or an exercise in GIN
acrobatics. More fundamentally, a step is the thing costs and facts attach to, so it
needs a real primary key.

### A mutable plan with an audit table

Rejected: a `step_id` recorded in the trace could then point at a step whose text has
since changed, which quietly makes the trace untrue. Immutability of revisions is what
makes ADR-0001 honest.

## Consequences

- More joins, and code that assembles and disassembles the nested model.
- Revisions grow the table; plans are small and revisions are few, so this is noise.
- Resuming a run after approval is trivial: the approved revision is a row, not a
  coroutine's local variable. This is what makes ADR-0007 cheap.

## How to invalidate

If plans are always read whole, there is never more than one revision, and no analytical
query ever touches `plan_steps` by shape, then normalisation bought nothing and a JSONB
document would have been the cheaper design.
