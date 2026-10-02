"""CLI for integrity-checked fixed sample import (pipeline added in stage 3)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import CDMapError
from .config import PROJECT_ROOT, load_config
from .import_fixed import import_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manifest-driven change detection error maps")
    subs = parser.add_subparsers(dest="command", required=True)
    imp = subs.add_parser("import-fixed", help="Verify and safely import an unchanged frozen sample archive")
    imp.add_argument("--config")
    imp.add_argument("--archive", required=True)
    imp.add_argument("--checksum-file")
    imp.add_argument("--expected-archive-sha256")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        dest = config["samples_root"]
        try:
            relative = dest.relative_to(PROJECT_ROOT).as_posix() + "/selection.json"
        except ValueError:
            relative = None
        if relative is not None and (PROJECT_ROOT / ".git").exists():
            proc = subprocess.run(["git", "check-ignore", "--no-index", relative], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                raise CDMapError("Refusing real data import before Git ignore rule is effective: " + relative)
        report = import_archive(Path(args.archive), Path(args.checksum_file) if args.checksum_file else None,
                                dest, args.expected_archive_sha256, config.get("expected_selection_sha256"),
                                config.get("expected_counts"), config.get("expected_size"))
        local = PROJECT_ROOT / "local"
        local.mkdir(exist_ok=True)
        (local / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        short = {key: value for key, value in report.items() if key not in ("image_hashes", "manifest_hashes")}
        print(json.dumps(short, ensure_ascii=False, indent=2))
        return 0
    except (CDMapError, OSError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
