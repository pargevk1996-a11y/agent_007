# Architecture decision records

One file per decision. A record is written when the decision is made, not afterwards, and
it is never edited to look correct in hindsight — a decision that turns out wrong is
superseded by a new record that says so.

Every record carries a **How to invalidate** section. If nobody can name the observation
that would prove the decision wrong, it was a preference rather than a decision.

| # | Decision | Implemented in |
| --- | --- | --- |
| [0001](0001-run-state-as-event-log-with-projection.md) | Run state lives in an append-only event log with a state projection written in the same transaction | phase 2, 8 |
| [0002](0002-plans-as-normalised-immutable-revisions.md) | Plans are normalised rows, and editing one creates a new immutable revision | phase 7 |
| [0003](0003-narrow-provider-interface.md) | The provider interface is narrow, and structured output is a declared property of each adapter | phase 3 |
| [0004](0004-reserve-then-commit-budgets.md) | Budgets reserve an upper bound before a call and commit the actual cost after it | phase 4 |
| [0005](0005-record-and-replay-llm-calls-too.md) | Determinism requires recording LLM calls, not only tool calls | phase 5, 16 |
| [0006](0006-critic-runs-on-evidence-only-context.md) | The critic sees the evidence and not the executor's reasoning, and its recall is measured | phase 10, 16 |
| [0007](0007-runs-wait-in-the-database.md) | A run awaiting human approval waits in the database, not in a coroutine | phase 8, 14 |
| [0008](0008-sse-with-postgres-backfill.md) | Progress streams over SSE, backfilled from PostgreSQL, with Redis carrying only the hot tail | phase 14 |
| [0009](0009-uuidv7-identifiers.md) | Identifiers are UUIDv7 generated in the application | phase 2 |
| [0010](0010-prompts-as-versioned-files-with-snapshots.md) | Prompts are versioned files, and the rendered text is snapshotted with every call | phase 7 onward |

New records start from [0000-template.md](0000-template.md).
