# 4. Enforce budgets by reserving before a call and committing after it

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 4
- **Related:** ADR-0003, ADR-0007

## Context

Budgets must be hard at three scopes: per call, per sub-question, per run. But the cost of
an LLM call is unknown until it finishes, because output tokens are only counted after
generation. A check performed afterwards is not enforcement, it is a notification of
overspend: a single call with a large `max_tokens` can blow a remaining budget several
times over.

Python adds a second constraint. Cancellation in asyncio is cooperative — it is delivered
at `await` points. A synchronous call inside a C extension cannot be interrupted.

## Decision

- **Before a call**, compute an upper bound: counted input tokens times input price, plus
  `max_tokens` times output price. If the reservation does not fit the remaining scope,
  the call **does not start**. `max_tokens` is clamped to what the budget can afford
  rather than taken from configuration unexamined.
- **After the call**, the actual cost from the provider's usage report is committed and
  the reservation released. The difference returns to the scope.
- `BudgetScope` is an async context manager. Remaining budget is checked at fixed
  checkpoints: before each LLM call, before each tool call, between steps.
- CPU-bound synchronous parsers run in an executor with their own hard timeout. They are
  **not** interruptible mid-document, and that limit is documented rather than implied.
- Prices live in a versioned file with `effective_from` dates. A computed cost is stored
  as an absolute number and **never recomputed**, so historical cost reports do not move
  when a vendor changes its price list.

## Alternatives considered

### Check the accumulated cost after each call

Rejected: this permits an arbitrary overshoot on the call that crosses the line. It is a
budget report, not a budget.

### Pre-emptive interruption by timer or signal

Rejected: in Python that means `signal` — main thread only, and hostile to asyncio — or
killing the process, which violates the requirement that a cancelled scope leaves no
partial writes.

## Consequences

- The system is conservative: it will sometimes refuse a call that would in fact have fit.
- Cancellation is bounded, not instant. The worst case is one outstanding tool call.
- Every call has a price attached at the moment it happens, which is what makes cost per
  report a measurable quantity rather than an estimate.

## How to invalidate

If refusals appear regularly while actual spend sits far below the reservation, the upper
bound is too crude and should be replaced by a prediction from the observed distribution
of output tokens for that step type.
