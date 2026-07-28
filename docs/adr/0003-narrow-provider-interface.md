# 3. Keep the provider interface narrow and make structured output an adapter property

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 3
- **Related:** ADR-0004, ADR-0005

## Context

Anthropic, OpenAI and vLLM must be interchangeable by configuration. But their guarantees
around structured output are genuinely different: Anthropic and OpenAI constrain
generation through native tool use or a JSON schema mode, while vLLM produces valid
schemas only with guided decoding and otherwise merely tries. Hiding that difference
behind one uniform call makes the weakest case invisible.

## Decision

- Two methods: `complete(CompletionRequest) -> CompletionResult` and
  `complete_structured(StructuredRequest[T]) -> StructuredResult[T]`.
- Embeddings live behind a **separate** `EmbeddingProvider` interface. With vLLM they may
  be a different server entirely, so conflating them would be wrong from day one.
- Every adapter declares `structured_output_mode`: `native_tools`, `json_schema`,
  `guided_decoding` or `prompted`. Retry policy is derived from it — one or two attempts
  where generation is constrained, more where it is not.
- `instructor` implements retry **inside** an adapter. It is not the entry point.
- No vendor SDK is imported outside `providers`. This is enforced by an `import-linter`
  contract, not by convention.

## Alternatives considered

### Normalise everything to OpenAI-shaped messages

Rejected: it discards vendor features we actually want — Anthropic prompt caching, which
is a large saving on long source excerpts, and extended thinking. An abstraction at the
lowest common denominator costs more than it saves here.

### Use `instructor` as the only entry point

Rejected: `instructor` covers neither cost accounting, nor a provider failure taxonomy,
nor prompt caching, nor cancellation. We would write a layer around it regardless, so the
layer should be ours and the library should sit inside it.

## Consequences

- We own an interface and must maintain it as vendor APIs move.
- Guarantees differ per adapter and that difference is visible in the type. Good for
  honesty, and it means tests branch per mode.
- Swapping a provider is a configuration change, so evaluation can compare providers on
  identical inputs.

## How to invalidate

If the vLLM adapter is never built because no endpoint materialises, we paid for a
three-way abstraction and used two. Equally, if `structured_output_mode` never influences
any behaviour, it should collapse into a simple retry count.
