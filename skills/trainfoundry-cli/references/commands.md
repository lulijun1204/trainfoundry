# TrainFoundry CLI command reference

## Command discovery

```bash
tf commands --output json
tf --help
tf <group> <command> --help
```

`trainfoundry` is a compatible long alias for every `tf` command.

## Fetch commands

```bash
trainfoundry fetch list --output json
trainfoundry fetch run <source-id>... --dry-run --output json
trainfoundry fetch run <source-id>... --output json
trainfoundry fetch run --group huggingface --output json
trainfoundry fetch run --group non_text --output json
trainfoundry fetch run --all --output json
trainfoundry fetch run <source-id> --force --output json
trainfoundry fetch reconcile --output json
```

`--dry-run` and `--force` are mutually exclusive. A fetch requires exactly one
target mode: source IDs, `--group`, or `--all`.

Plan actions:

- `download`: no matching completed record exists.
- `skip`: request fingerprint matches and every local file verifies.
- `repair`: request fingerprint matches but a file is missing or corrupt.
- `force`: caller explicitly bypassed deduplication.

## Dataset commands

```bash
trainfoundry dataset inspect <source-id> --output json
trainfoundry dataset verify <source-id> --output json
trainfoundry dataset verify --output json
```

The no-argument verify command requires the complete expected catalog to be
present. Use the source-specific form for partial local installations.

## Configuration commands

```bash
trainfoundry config show --output json
trainfoundry config validate --output json
```

Validation checks Python, curl, and all configured storage paths.

## Metadata commands

Initialize or inspect the project-local SQLite database:

```bash
trainfoundry metadata status --output json
trainfoundry metadata init --output json
```

Discover entity-specific arguments with `--help`:

```bash
trainfoundry metadata dataset create --help
trainfoundry metadata version create --help
trainfoundry metadata run create --help
trainfoundry metadata lineage create --help
trainfoundry metadata quality create --help
trainfoundry metadata annotation create --help
trainfoundry metadata training create --help
trainfoundry metadata training bind-version --help
```

Read commands:

```bash
trainfoundry metadata dataset get <dataset-id> --output json
trainfoundry metadata dataset list --output json
trainfoundry metadata version get <version-id> --output json
trainfoundry metadata version list --dataset-id <dataset-id> --output json
trainfoundry metadata run get <run-id> --output json
trainfoundry metadata run list --output json
trainfoundry metadata lineage trace <version-id> --output json
trainfoundry metadata quality get <result-set-id> --output json
trainfoundry metadata quality list --output json
trainfoundry metadata annotation get <result-set-id> --output json
trainfoundry metadata annotation list --output json
trainfoundry metadata training get <training-run-id> --output json
trainfoundry metadata training list --output json
```

DatasetVersion content, schema, storage, identity, stage, and split become
immutable after leaving `DRAFT`. Training bindings accept only `COMMITTED` or
`PUBLISHED` versions.

## Output envelope

Success:

```json
{
  "ok": true,
  "command": "fetch.run",
  "data": {},
  "error": null
}
```

Failure:

```json
{
  "ok": false,
  "command": "fetch.run",
  "data": null,
  "error": {
    "code": "DOWNLOAD_FAILED",
    "message": "description",
    "retryable": true
  }
}
```

## Exit codes

- `0`: success
- `1`: internal or unclassified failure
- `2`: invalid arguments
- `3`: configuration or required-tool error
- `4`: dataset or registry record not found
- `5`: download failure
- `6`: verification failure
- `7`: permission failure
- `8`: metadata uniqueness, foreign-key, immutable, or lifecycle conflict
