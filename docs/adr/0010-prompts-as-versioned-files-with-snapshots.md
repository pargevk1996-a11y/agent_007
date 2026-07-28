# 10. Version prompts as files and snapshot the rendered text in the log

- **Status:** Accepted
- **Date:** 2026-07-28
- **Implemented in:** phase 7 onward
- **Related:** ADR-0001, ADR-0005

## Context

Prompts are the largest single influence on output quality, so they belong in review,
in diffs and in version control — they are code. But a file alone does not make a run
reproducible a year later. The file may have changed, and even an unchanged template
says nothing about the values that were substituted into it on the day.

## Decision

**The file is the source, the hash is the identity, the rendered text is the record.**

- Files live at `prompts/<module>/<name>.v<N>.md`. A new version is a **new file**; an
  existing version file is never edited.
- A YAML front matter block carries `name`, `version`, `model_hint`,
  `expected_output_schema` (the name of the Pydantic class) and a `changelog` line.
- The body is hashed with SHA-256. Every row in `calls` records `prompt_name`,
  `prompt_version` and `prompt_hash`.
- **The fully rendered prompt** — after substitution — is stored with the call. Without it
  the reconstruction requirement of ADR-0001 is false, because substituted values cannot
  be recovered from a template hash.
- CI checks that every prompt file has a corresponding output schema, that every version
  referenced in code exists on disk, and that **no existing version file has changed**
  relative to its recorded hash.

## Alternatives considered

### Prompts in the database with an editing UI

Rejected: it separates a prompt from the code that parses its output, so the two drift.
Prompts must ship with the code that depends on them.

### Store only the hash, not the rendered text

Rejected: the substitution is unrecoverable, which defeats the purpose.

## Consequences

- The log grows substantially. A rendered prompt carrying source excerpts can be tens of
  kilobytes per call, and megabytes per run. Large payloads are therefore stored in a
  table separate from call metadata, so retention can delete the bulk while the trace
  skeleton survives.
- Editing a prompt invalidates cassettes (ADR-0005). That is the intended coupling.

## How to invalidate

If payload storage grows faster than the usefulness of old traces, payloads move to object
storage and the database keeps a reference. If prompts are never compared across versions,
the front matter is ceremony and can shrink.
