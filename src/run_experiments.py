"""CLI entry point reserved for the full pruning/quantization matrix."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuantPrune-X experiment matrix")
    parser.add_argument("config", help="TOML experiment configuration")
    parser.add_argument("--output", required=True, help="run artifact directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"experiment pipeline is owned by experiment team: {args.config}")


if __name__ == "__main__":
    main()
