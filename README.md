# TrainFoundry

> A reproducible multimodal training-data and AI infrastructure playground.

TrainFoundry is a hands-on project for exploring the training-data lifecycle
across text, image, video, and robot-trajectory data.

For concrete, abstraction-free walkthroughs of the actual data operations, see:

- [`pipeline/examples/data_processing/wikitext/`](pipeline/examples/data_processing/wikitext/README.md)
  for JSONL validation, normalization, Arrow batching, Lance writing, and read-back;
- [`pipeline/examples/data_processing/common_crawl/`](pipeline/examples/data_processing/common_crawl/README.md)
  for streaming gzip/WARC-WET file-format and conversion-record validation.

The current milestone implements reproducible source ingestion: dataset
fetchers, resumable archive downloads, file-size and SHA-256 metadata, a shared
source registry, and an offline-RL loading demo. Transformation, immutable
snapshots, lineage, DataLoaders, and distributed experiments are planned work.

## Implemented sources

The fetchers prepare seven representative public datasets without trying to
mirror them at production scale:

| Modality | Dataset | Current output | Approximate download |
| --- | --- | --- | ---: |
| Text | WikiText-2 Raw | Split JSONL files | Small |
| Text | Common Crawl WET | One WET archive from the latest crawl | Varies |
| Text | Dolly 15K | Split JSONL files | Small |
| Text | Anthropic HH-RLHF helpful-base | Split JSONL files | Small |
| Image | COCO 2017 Validation | Images and annotation archives | 1.1 GB |
| Video | UCF101 | Video and official split archives | 6.9 GB |
| Robot | D4RL PointMaze UMAZE via Minari | Local Minari dataset | Varies |

Every fetcher reads its output location from `config/paths.toml`. Downloads stay
under the Git-ignored `model_data/` directory in the repository by default and
are recorded in a shared source registry with file sizes and SHA-256 checksums.
Each successful fetch also creates or reuses a governed Dataset and committed
RAW DatasetVersion in the project-local metadata database.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- `curl`, used for resumable HTTP archive downloads
- Enough disk space for the selected sources

Hugging Face may require an `HF_TOKEN` for gated or rate-limited access. Keep
tokens in environment variables; never add them to `config/paths.toml`.

## Setup

Verify the prerequisites:

```bash
python --version  # Python 3.11+
uv --version
curl --version
```

For local development, create the environment and run the CLI through `uv`:

```bash
uv sync --dev
uv run tf --version
```

Install the CLI for direct use outside the repository:

```bash
uv tool install .
tf --version
```

After updating the source, reinstall the user-level CLI:

```bash
uv tool install --force .
```

Verify the executable and configuration:

```bash
command -v trainfoundry
command -v tf
tf --version
tf config validate --output json
```

If the executable is not on `PATH`, run `uv tool update-shell` and open a new
terminal.

The committed `config/paths.toml` uses portable paths under the repository's
Git-ignored `model_data/` directory. Change them if you want datasets stored
elsewhere.

The installed CLI first checks `TRAINFOUNDRY_CONFIG_DIR`, then a
`config/paths.toml` under the current working directory, and finally its
packaged defaults. Set `TRAINFOUNDRY_PROJECT_ROOT` to change the base used for
relative paths. Outside a project checkout, the default base is
`~/.local/share/trainfoundry`.

To use an explicit configuration from any working directory:

```bash
export TRAINFOUNDRY_CONFIG_DIR=/path/to/trainfoundry/config
export TRAINFOUNDRY_PROJECT_ROOT=/path/to/trainfoundry
trainfoundry config show --output json
```

## CLI

TrainFoundry exposes one extensible command tree through the short `tf`
executable. The original `trainfoundry` name remains as a fully compatible
alias:

```bash
tf commands --output json
trainfoundry commands --output json
```

All examples written as `trainfoundry ...` can therefore be shortened to
`tf ...`; both invoke `trainfoundry.cli:main`.

Inspect configuration:

```bash
trainfoundry config show --output json
trainfoundry config validate --output json
```

### Local metadata

Initialize the project-local SQLite metadata database:

```bash
trainfoundry metadata status --output json
trainfoundry metadata init --output json
```

The default database is `.trainfoundry/metadata.db`. It stores Dataset,
DatasetVersion, DatasetRun, DatasetLineage, quality, annotation, and training
metadata. The database is local and Git-ignored; its schema and all access
methods live in [`metadata/`](metadata/README.md).

Basic read/write examples:

```bash
trainfoundry metadata dataset create \
  --namespace examples \
  --name dolly_sft \
  --purpose SFT \
  --modality TEXT \
  --owner training-data \
  --output json

trainfoundry metadata dataset list --output json
trainfoundry metadata version list --dataset-id <dataset-id> --output json
trainfoundry metadata lineage trace <version-id> --output json
```

Use `trainfoundry metadata --help` for the complete command tree.

### Fetch datasets

List the declared source requirements:

```bash
trainfoundry fetch list --output json
```

Plan a download without changing local state:

```bash
trainfoundry fetch run coco_2017_validation \
  --dry-run \
  --output json
```

Fetch one or more sources, a named group, or the complete catalog:

```bash
trainfoundry fetch run wikitext_2_raw --output json
trainfoundry fetch run coco_2017_validation ucf101 --output json
trainfoundry fetch run --group huggingface --output json
trainfoundry fetch run --group non_text --output json
trainfoundry fetch run --all --output json
```

Run the repository batch script to validate configuration, initialize metadata,
download the complete catalog, and print the registered Dataset and
DatasetVersion records:

```bash
scripts/fetch_all_and_register.sh --dry-run
scripts/fetch_all_and_register.sh
```

Use `--force` only when every acquisition should be executed again. Identical
content at the same storage location still reuses its existing DatasetVersion.

Bypass deduplication and deliberately execute a source again only when needed:

```bash
trainfoundry fetch run coco_2017_validation \
  --force \
  --output json
```

Direct HTTP archive downloads use a `.part` file and resume automatically.
Hugging Face and Minari downloads use their respective client libraries.
Completed raw archives are retained rather than extracted in place.

Before downloading, `FetcherService` compares the resolved request fingerprint
with the registry and verifies every existing file:

- `skip`: matching record and checksums are valid
- `repair`: matching record exists but local files are missing or corrupt
- `download`: no matching completed record exists
- `force`: explicitly bypass deduplication with `--force`

If text sources were downloaded before the registry was introduced, reconcile
them with:

```bash
trainfoundry fetch reconcile --output json
```

Inspect or verify registry records:

```bash
trainfoundry dataset inspect coco_2017_validation --output json
trainfoundry dataset verify coco_2017_validation --output json
trainfoundry dataset verify --output json
```

The no-argument verify command expects all seven catalog sources.

The Common Crawl fetcher selects the newest crawl available at runtime, so its
revision and checksum can change between runs.

### Structured output

Every leaf command accepts `--output json` and returns a stable envelope:

```json
{
  "ok": true,
  "command": "fetch.run",
  "data": {},
  "error": null
}
```

Errors contain a code, message, and `retryable` flag. Command logs go to
`stderr`; the JSON result stays on `stdout`.

CLI exit codes are:

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 1 | Internal or unclassified failure |
| 2 | Invalid arguments |
| 3 | Configuration or required-tool error |
| 4 | Dataset or registry record not found |
| 5 | Download failure |
| 6 | Verification failure |
| 7 | Permission failure |
| 8 | Metadata uniqueness, foreign-key, or lifecycle conflict |

### Agent skill

The distributable Codex skill lives in `skills/trainfoundry-cli`. Install it
into the default Codex skill directory:

```bash
cp -R skills/trainfoundry-cli "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Open a new Codex session after installation. Example requests:

```text
Use trainfoundry to list the supported datasets.
Use trainfoundry to preview a COCO download.
Use trainfoundry to verify UCF101.
```

The skill directs an Agent to discover commands, prefer JSON, run a dry-run
before downloads, avoid broad `--all` or `--force` operations unless requested,
and interpret structured failures.

## Run the robot-data demo

```bash
uv run python main.py
```

The demo loads the configured Minari dataset and prints its step, episode, and
sample schema.

Run the learning-oriented full validation in two explicit stages:

```bash
uv run python -m pipeline.examples.data_processing.minari_pointmaze.step_01_validate_file_format
uv run python -m pipeline.examples.data_processing.minari_pointmaze.step_02_validate_data
```

The first command validates and hashes the physical Minari package; the second
checks every trajectory's temporal schema and PointMaze semantics. Both produce
quality results only and do not create a new DatasetVersion.

## Fetch architecture

Download requirements are declared as `DatasetMeta` objects in
`fetch/catalog.py`; they contain dataset identity, governance metadata, output
location, revision, URLs, and expected sizes, but no download implementation.
The complete execution flow, deduplication rules, retry behavior, and extension
points are documented in [`fetch/README.md`](fetch/README.md).

```text
DatasetMeta in catalog
        |
        v
optional resolver -------- Common Crawl discovery
        |
        v
tool adapter ------------- curl / Hugging Face datasets / Minari
        |
        v
optional materializer ---- JSONL export / Minari schema inspection
        |
        v
DatasetRecord
        |
        v
atomic registry upsert
```

Fixed HTTP datasets such as COCO and UCF101 use the same `CurlDownloader`.
Adding another fixed-file dataset normally requires only a new `DatasetMeta`,
not another downloader. Logical dataset services keep dedicated adapters
because Hugging Face and Minari control their own caching and storage formats.

`FetcherService.fetch(meta)` is the operation that combines these layers.
It writes metadata only after acquisition and materialization succeed. Registry
writes use `source_id` as the identity key, while a resolved request fingerprint
and file checksums prevent redundant work.

```python
from fetch import FetcherService
from fetch.catalog import get_source

meta = get_source("coco_2017_validation")
record = FetcherService().fetch(meta)
print(record.to_dict())
```

## Development

Run the same checks used by GitHub Actions:

```bash
uv sync --locked --dev
uv run ruff check .
uv run pytest
uv run trainfoundry commands --output json
```

The test suite uses temporary local files and does not download datasets.

## Project layout

```text
config/                  TOML configuration and path helpers
trainfoundry/            Multi-command CLI and machine-readable contract
metadata/                SQLite schema, repository, and metadata CLI
fetch/catalog.py         Declarative dataset requirements
fetch/models.py          Requirement and result models
fetch/fetcher_service.py Single fetch, materialize, and register service
fetch/downloaders/       curl, Hugging Face, and Minari tool adapters
fetch/resolvers/         Dynamic source discovery
fetch/materializers/     Post-download conversion and inspection
fetch/registry.py        Atomic DatasetRecord persistence
skills/trainfoundry-cli/ Codex Agent skill
tests/                   Offline unit tests
main.py                  Minari robot-data loading demo
```

## Roadmap

- Raw, Processed, and Dataset storage layers
- Unified Asset, Sample, Annotation, QualityResult, and Lineage schemas
- Idempotent validation, quarantine, retry, and recovery
- Immutable snapshots, diffs, lineage, and rollback
- Ray or PySpark distributed processing
- PyTorch DataLoader and CPU DDP benchmarks
- Throughput, latency, failure, backlog, resource, and cost metrics

## Data and licensing

This repository contains code and metadata, not the downloaded datasets.
Public download access does not imply a uniform open-source license. Review
each source's terms and any item-level rights before use or redistribution.

The TrainFoundry source code is available under the [MIT License](LICENSE).
