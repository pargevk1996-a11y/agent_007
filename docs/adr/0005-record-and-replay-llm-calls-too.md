# 5. Record and replay LLM calls, not only tool calls

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 5 (transport), phase 16 (harness)
- **Related:** ADR-0003, ADR-0010

## Context

The design principle is that identical inputs produce an identical plan and an identical
execution path. Recording tool responses does not achieve this. An LLM call is not
deterministic even at temperature zero: provider-side batching, mixture-of-experts routing
and non-deterministic GPU reduction all move the output. Anthropic exposes no seed at all;
OpenAI's `seed` is best-effort.

So either LLM responses are recorded as well, or determinism is a claim rather than a
property — and every quality measurement in phase 16 becomes noise.

## Decision

One cassette layer covering both kinds of outbound call. The cassette key is derived from
our own domain, not from HTTP bytes:

```
key = sha256(kind, provider, model, prompt_version_hash, rendered_prompt,
             canonical_params, schema_hash)
```

Three modes:

- `live` — real calls, cassettes written
- `replay` — the network is forbidden; a miss is a test failure, never a fallback request
- `record-missing` — for refreshing seeds deliberately

HTTP-level mocking with `respx` stays, but for tools only, where request bodies are stable.

## Alternatives considered

### HTTP-level record and replay for everything, including LLM calls

Rejected as the sole mechanism: it is brittle against SDK changes and streaming responses,
and keys computed over request bodies break on irrelevant differences such as key ordering
or timestamps. A domain-level key is stabler.

### Accept non-determinism and measure across N runs

Rejected as the primary mode: N times the cost of a report per measurement, and too much
variance to detect a regression caused by editing a prompt. Retained as a secondary mode
for measuring stability itself.

## Consequences

- Cassettes expire when a prompt changes. That is intended — changing a prompt should
  force a re-recording — but it means a refresh command and cassette diffs in review.
- Cassettes for twenty seed questions over long documents are large. Compression or
  git-lfs will probably be needed.
- Evaluation becomes a measurement: the same question yields the same trace, so a change
  in output is attributable to a change we made.

## How to invalidate

If cassettes need re-recording on nearly every commit, the key is too sensitive and should
be coarsened, starting with whitespace normalisation of the rendered prompt.
