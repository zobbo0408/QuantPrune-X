"""CLI entry point reserved for bulk Q-DQ checkpoint export."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Q-DQ checkpoints")
    parser.add_argument("config", help="TOML export configuration")
    parser.add_argument("--output", required=True, help="output checkpoint directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"export pipeline is owned by quantization team: {args.config}")


if __name__ == "__main__":
    main()
