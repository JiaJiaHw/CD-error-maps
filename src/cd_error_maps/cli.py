"""Command line entry point; all relative paths resolve from project root."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import CDMapError
from .config import PROJECT_ROOT, load_config
from .import_fixed import import_archive
from .pipeline import choose_combinations, generate, validate_config_inputs, verify


def _project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _print_summary(report):
    short = {key: value for key, value in report.items() if key not in ("pairs", "image_hashes", "manifest_hashes")}
    print(json.dumps(short, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manifest-driven change detection error maps")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "generate"):
        sub = subs.add_parser(command, help="Check inputs" if command == "validate" else "Generate a new verified run")
        sub.add_argument("--config", help="Project-relative or absolute JSON config; default local.fixed100.json")
        sub.add_argument("--dataset", action="append", help="Repeat to select datasets (Cartesian product with models)")
        sub.add_argument("--model", action="append", help="Repeat to select models (canonical directory names)")
        sub.add_argument("--pair", action="append", help="Repeat explicit DATASET:MODEL combinations")
        if command == "validate":
            sub.add_argument("--samples-only", action="store_true", help="Check GT and manifests only; exclude Prediction validation")
            sub.add_argument("--report", help="Optional new JSON report path; refuses overwrite")
        else:
            sub.add_argument("--limit", type=int, help="Preview first N per combination, after full-group validation")
            sub.add_argument("--out", help="New run directory; defaults to unique directory under output_root")
    ver = subs.add_parser("verify", help="Read back and verify a completed run including source inputs")
    ver.add_argument("--run", required=True)
    imp = subs.add_parser("import-fixed", help="Verify and safely import an unchanged frozen sample archive")
    imp.add_argument("--config")
    imp.add_argument("--archive", required=True)
    imp.add_argument("--checksum-file")
    imp.add_argument("--expected-archive-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            _print_summary(verify(_project_path(args.run)))
            return 0
        config = load_config(args.config)
        if args.command in ("validate", "generate"):
            combinations = choose_combinations(config, args.dataset, args.model, args.pair)
            if args.command == "generate":
                _print_summary(generate(config, combinations, args.limit, args.out))
            else:
                report = validate_config_inputs(config, combinations, args.samples_only)
                if args.report:
                    path = _project_path(args.report)
                    for root in (config["samples_root"], config["prediction_root"]):
                        if path == root or root in path.parents:
                            raise CDMapError("Validation report overlaps an input root: " + str(root))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                        stream.write("\n")
                _print_summary(report)
            return 0
        dest = config["samples_root"]
        try:
            relative = dest.relative_to(PROJECT_ROOT).as_posix() + "/selection.json"
        except ValueError:
            relative = None
        if relative is not None and (PROJECT_ROOT / ".git").exists():
            proc = subprocess.run(["git", "check-ignore", "--no-index", relative], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                raise CDMapError("Refusing real data import before Git ignore rule is effective: " + relative)
        report = import_archive(_project_path(args.archive), _project_path(args.checksum_file) if args.checksum_file else None,
                                dest, args.expected_archive_sha256, config.get("expected_selection_sha256"),
                                config.get("expected_counts"), config.get("expected_size"))
        local = PROJECT_ROOT / "local"
        local.mkdir(exist_ok=True)
        (local / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _print_summary(report)
        return 0
    except (CDMapError, OSError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: no new run is complete without a verified COMPLETE.json", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
