# 11. Build a modular monolith and enforce its layering in CI

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 1
- **Related:** ADR-0001, ADR-0007

## Context

The agent has six stages — plan, search, read, reason, critique, cite — and it is
tempting to give each one a service. That temptation should be answered once, in writing,
because reversing it later is expensive in both directions.

Microservices solve three problems: scaling components independently, deploying them
independently across teams, and isolating blast radius. None of the three applies here.
There is one team, one deployment cadence, and the stages have near-identical load
profiles because they run in sequence within a single run.

A second question follows immediately: if the modules live in one process, what stops the
boundaries between them from dissolving into a single tangle within a few phases?

## Decision

**One repository. Two runtime services. Everything else is a Python module.**

- `api` — FastAPI, short-lived request and response plus SSE streaming
- `worker` — long-lived, executes runs

They are separate containers because their lifecycles genuinely differ, not because they
are different domains. PostgreSQL, Qdrant and Redis are the three data services.

Inside `worker`, planner, executor, critic, synthesizer and memory are modules called
through typed Python calls. Transactions are PostgreSQL transactions. There are no sagas.

**Layering, enforced by `import-linter` contracts in `make arch` and in CI:**

```
api : worker
planner : executor : critic : synthesizer
memory : tools
providers : budget
events
storage
core
```

Dependencies point downward only; names on the same line are siblings and may not import
each other. `config` and `obs` are cross-cutting, sit outside the stack, and may not
reach into the agent. Three further contracts hold the important invariants: vendor SDKs
appear only in `providers`, `asyncpg` appears only in `storage`, and `core` imports
nothing of ours.

Two positions in that stack were corrected while writing the contracts, and the
corrections are the interesting part:

- **`memory` and `tools` sit above `providers`,** because semantic memory needs an
  embedding provider. As siblings they could not have imported it.
- **`events` sits above `storage`,** because events are persisted through storage rather
  than beside it.

**The constraint that makes future extraction real:** a transaction must never cross a
module boundary. Where it would, the boundary is drawn in the wrong place.

## Alternatives considered

### A service per agent stage

Rejected: it converts type boundaries into network boundaries, turns a database
transaction into a distributed one, and makes a failing run harder to read. It buys
independent scaling that nothing in the workload asks for.

### One package with layering as a convention

Rejected: conventions decay silently and are only noticed at the third phase after the
violation. A contract that fails the build is noticed in the pull request that breaks it.

## Consequences

- One deploy, one local environment, one debugger attaches to the whole run.
- The layering is checked on empty packages today. It is a trap set for later code, not
  evidence that today's code is correct.
- **Extraction is not uniformly cheap, and the honest version of that claim is worth
  stating.** A module whose transactions stay inside itself can be lifted out in about a
  week, because its interface already exists. `executor` and `memory` share a transaction
  with `storage` — a fact, its embedding and its event are written together — so
  separating those would mean an outbox and a rewrite, not a week. The layering rule above
  exists to keep that set as small as possible.

## How to invalidate

If one stage develops a different scaling profile — a local vLLM needing GPU nodes, or a
tool that requires a sandboxed runtime — extraction becomes worth its cost for that stage
alone. If a crash in one stage starts taking unrelated runs down with it, the blast radius
argument stops being theoretical.
