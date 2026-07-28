# 8. Stream over SSE with a PostgreSQL backfill and a Redis hot tail

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 14
- **Related:** ADR-0001, ADR-0007

## Context

Progress has to reach the client while a run executes, and a client that reconnects must
not silently miss what happened while it was away. The traffic is one-directional: the
worker emits, the client observes. Control actions — approve, cancel — are ordinary
requests, not stream messages.

## Decision

**Server-sent events.** The stream is one-way, and SSE provides reconnection, the
`Last-Event-ID` header, transparent passage through proxies and debuggability with `curl`
without inventing a protocol.

Delivery guarantee:

- `event.id` in the SSE frame is a **monotonic per-run sequence number** taken from
  PostgreSQL, not a UUID. UUIDv7 (ADR-0009) is only millisecond-ordered and is unsuitable
  as a delivery cursor.
- On reconnect the client sends `Last-Event-ID`. The API **first backfills from the
  `events` table** where `seq > last`, then switches to the live tail from the Redis
  stream.
- Redis therefore carries only the hot tail. It needs no durability guarantee, and its
  `MAXLEN` can stay small. Its container has no volume for exactly this reason.
- Publication to Redis happens **after** the owning transaction commits (ADR-0001).

The polling endpoint `GET /runs/{id}/events?since=N` is exposed publicly as well, because
the SDK is better served by polling than by a stream.

## Alternatives considered

### WebSocket

Rejected: bidirectionality is not needed, and it costs a bespoke message protocol,
heartbeat logic, manual client reconnection and worse proxy traversal.

### Long polling as the primary transport

Rejected on latency and load, though it exists anyway as the backfill endpoint above.

## Consequences

- Each SSE connection holds an open request. At tens of thousands of clients this meets
  worker limits; at our scale it does not matter.
- An event committed but not published — a crash in the gap — is recovered by the client's
  next backfill rather than lost.

## How to invalidate

If interaction appears inside a running run — editing a sub-question mid-execution, or a
conversation with the agent — then SSE plus POST becomes awkward and a WebSocket earns
its cost.
