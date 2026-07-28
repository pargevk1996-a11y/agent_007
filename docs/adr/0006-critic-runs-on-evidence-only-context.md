# 6. Give the critic an evidence-only context, and measure it

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 10, measured in phase 16
- **Related:** ADR-0003, ADR-0005

## Context

A critic that reviews the executor's work is worthless if it inherits the executor's
frame. The usual prescription — "use a different model" — addresses the smaller half of
the problem. Self-preference bias mostly enters through *seeing the reasoning that
produced the claim*, not through sharing weights with its author.

There is a second problem that is rarely stated: without a metric, nobody can tell whether
a critic catches anything at all. An unmeasured critic is a ritual.

## Decision

Three measures, in descending order of importance.

1. **Evidence-only context.** The critic receives the claim, the source excerpts with
   their URLs, and the sub-question. It does not receive the executor's reasoning, its
   confidence wording, or the order in which facts were gathered.
2. **A different task, not just a different prompt.** The critic performs a typed
   classification — `supported`, `partially_supported`, `contradicted`, `no_source` — and
   a verdict of `supported` **requires** quoting the span of the source that supports it.
   The obligation to quote is a stronger anti-hallucination mechanism than model choice.
3. **A different model**, by default a different model from the same vendor. A
   cross-vendor critic is available by configuration.

Measurement, in phase 16: seed errors synthetically into verified facts — alter a number,
swap the subject, reattach a statement to the wrong source — and report the critic's
**recall on seeded errors** and its **false-positive rate on untouched facts**.

## Alternatives considered

### Mandatory cross-vendor critic

Rejected as mandatory: it doubles the failure surface — second key, second structured
output mode, second error taxonomy, second price list — for less bias reduction than the
evidence-only context provides. Kept as an option.

### Self-consistency: N samples of the same model, majority vote

Rejected: N times the cost, and correlated errors survive it untouched. A model that is
confidently wrong is confidently wrong N times.

## Consequences

- More false positives. The critic cannot see the context in which a claim was a fair
  generalisation. We accept that asymmetry deliberately: over-flagging is preferable to
  letting an unsupported claim through.
- The critic's own cost is a visible line in the cost per report.

## How to invalidate

If the false-positive rate rises far enough that a reader starts ignoring the flags, the
critic has become noise and must be given more context, trading away some independence.
