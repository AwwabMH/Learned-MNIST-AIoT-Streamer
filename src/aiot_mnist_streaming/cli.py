from __future__ import annotations

import argparse

from .pipeline import run_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIoT streaming MNIST project")
    parser.add_argument("--quick", action="store_true", help="Run the lightweight configuration")
    parser.add_argument("--full", action="store_true", help="Run the full configuration")
    parser.add_argument("--output-root", default="outputs_aiot_project_v2", help="Output root directory")
    parser.add_argument("--no-plots", action="store_true", help="Create figures without opening interactive windows")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    quick_mode = True if args.quick or not args.full else False
    run_project(quick_mode=quick_mode, output_root=args.output_root, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
