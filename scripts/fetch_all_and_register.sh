#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/fetch_all_and_register.sh [--dry-run] [--force]

Download every dataset declared in the TrainFoundry catalog. Each successful
fetch is automatically registered as a governed Dataset and committed RAW
DatasetVersion in the local metadata database.

Options:
  --dry-run  Print fetch plans without downloading or writing metadata.
  --force    Re-run every acquisition; content-identical versions are reused.
  -h, --help Show this help text.
EOF
}

dry_run=false
force=false

while (($# > 0)); do
  case "$1" in
    --dry-run)
      dry_run=true
      ;;
    --force)
      force=true
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$dry_run" == true && "$force" == true ]]; then
  echo "--dry-run and --force cannot be used together" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"

if command -v tf >/dev/null 2>&1; then
  tf_command=(tf)
elif command -v uv >/dev/null 2>&1; then
  tf_command=(uv run tf)
else
  echo "Neither tf nor uv is available on PATH" >&2
  exit 3
fi

"${tf_command[@]}" config validate --output json

fetch_args=(fetch run --all --output json)
if [[ "$dry_run" == true ]]; then
  fetch_args+=(--dry-run)
  "${tf_command[@]}" "${fetch_args[@]}"
  exit 0
fi
if [[ "$force" == true ]]; then
  fetch_args+=(--force)
fi

"${tf_command[@]}" metadata status --output json
"${tf_command[@]}" metadata init --output json

"${tf_command[@]}" "${fetch_args[@]}"

"${tf_command[@]}" metadata dataset list \
  --namespace catalog \
  --output json
"${tf_command[@]}" metadata version list --output json
