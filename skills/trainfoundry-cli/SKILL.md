---
name: trainfoundry-cli
description: Operate TrainFoundry datasets, local relational metadata, and training-data infrastructure through the trainfoundry CLI. Use when the user wants to discover, plan, fetch, reconcile, inspect, verify, initialize, read, or write TrainFoundry dataset metadata, lineage, quality, annotation, or training records.
---

# TrainFoundry CLI

Use the short `tf` executable as the stable interface to TrainFoundry.
`trainfoundry` is the compatible long alias.

## Start

1. Run `command -v tf` before the first operation.
2. If the executable is unavailable inside the TrainFoundry repository, use
   `uv run tf`.
3. Run `tf commands --output json` when command capabilities or
   arguments are uncertain.
4. Read [references/commands.md](references/commands.md) only when detailed
   command or error semantics are needed.

## Execute read-only operations

Use `--output json` and parse the response envelope:

```json
{"ok": true, "command": "fetch.list", "data": {}, "error": null}
```

Use these read-only commands without confirmation when they answer the request:

- `tf fetch list --output json`
- `tf dataset inspect <source-id> --output json`
- `tf dataset verify [source-id] --output json`
- `tf config show --output json`
- `tf config validate --output json`
- `tf metadata status --output json`
- `tf metadata dataset get <dataset-id> --output json`
- `tf metadata dataset list --output json`
- `tf metadata version get <version-id> --output json`
- `tf metadata version list --output json`
- `tf metadata lineage trace <version-id> --output json`

## Execute metadata writes

1. Run `tf metadata status --output json`.
2. If uninitialized, run `tf metadata init --output json`.
3. Use the narrow entity command under `metadata dataset`, `version`, `run`,
   `lineage`, `quality`, `annotation`, or `training`.
4. Inspect `METADATA_CONFLICT` instead of retrying immutable, unique, or
   foreign-key violations.
5. Commit a DatasetVersion before binding it to a TrainingRun.

## Execute dataset fetches

1. Resolve the narrowest set of source IDs that satisfies the request.
2. Run `tf fetch run ... --dry-run --output json`.
3. Review the tool, destination, revision, estimated bytes, action, and reason.
4. Run the same command without `--dry-run` after the requested scope is clear.
5. Do not add `--all` unless the user requests the complete catalog.
6. Do not add `--force` unless the user requests a fresh execution or the
   existing record must intentionally be replaced.

The service safely skips a matching verified record. It marks corrupt local
state as `repair` and refreshes it while preserving the old complete file until
the replacement download succeeds.

## Handle results

- Treat exit code `0` and `"ok": true` as success.
- Inspect `error.code`, `error.message`, and `error.retryable` on failure.
- Retry only errors marked retryable, unless the user gives different guidance.
- Report source ID, revision, action, file count, total bytes, and final status.
- Keep command logs separate from the JSON result; stdout is the result channel.
