"""Compatibility entry point for the unified TrainFoundry CLI."""

import sys

from trainfoundry.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fetch", *sys.argv[1:]]))
