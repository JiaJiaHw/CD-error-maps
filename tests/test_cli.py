"""CLI behavior tested exclusively with independent synthetic masks."""
import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cd_error_maps.cli import main


class CLITests(unittest.TestCase):
    def setUp(self):
        (REPO / ".tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(REPO / ".tmp"))
        self.root = Path(self.temp.name)
        (self.root / "samples").mkdir()
        (self.root / "pred/D/M").mkdir(parents=True)
        Image.fromarray(np.array([[0, 0], [255, 255]], dtype=np.uint8)).save(self.root / "samples/0001.png")
        Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(self.root / "pred/D/M/0001.png")
        with (self.root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "sample_id", "gt_relative_path"])
            writer.writerow(["D", "0001.png", "0001.png"])
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"schema_version": 1, "samples_root": str(self.root / "samples"),
            "samples_manifest": str(self.root / "samples.csv"), "prediction_root": str(self.root / "pred"),
            "output_root": str(self.root / "output"), "datasets": ["D"], "models": ["M"]}), encoding="utf-8")
        self.base = ["--config", str(self.config)]

    def tearDown(self):
        self.temp.cleanup()

    def call(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(args)
        return status, out.getvalue(), err.getvalue()

    def test_validate_generate_verify_and_explicit_reports(self):
        report = self.root / "report.json"
        self.assertEqual(self.call(["validate"] + self.base + ["--pair", "D:M", "--report", str(report)])[0], 0)
        self.assertEqual(json.loads(report.read_text())["pair_count"], 1)
        self.assertEqual(self.call(["validate"] + self.base + ["--report", str(report)])[0], 2)
        run = self.root / "output/run"
        self.assertEqual(self.call(["generate"] + self.base + ["--out", str(run)])[0], 0)
        self.assertEqual(self.call(["verify", "--run", str(run)])[0], 0)
        manifest = json.loads((run / "run_manifest.json").read_text())
        self.assertEqual(manifest["pairs"][0]["counts"], {"TN": 1, "TP": 1, "FP": 1, "FN": 1,
                                                         "ignored": 0, "valid": 4, "width": 2, "height": 2})
        self.assertEqual(self.call(["generate"] + self.base + ["--out", str(run)])[0], 2)

    def test_samples_only_empty_prediction_and_errors(self):
        (self.root / "pred/D/M/0001.png").unlink()
        code, text, _ = self.call(["validate"] + self.base + ["--samples-only"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(text)["prediction_checked"])
        code, _, error = self.call(["generate"] + self.base + ["--out", str(self.root / "output/missing")])
        self.assertEqual(code, 2)
        self.assertIn("Prediction not ready", error)
        self.assertFalse((self.root / "output/missing").exists())

    def test_invalid_scope_limits_and_input_report_conflict(self):
        for extra in (["--pair", "D:M", "--dataset", "D"], ["--model", "unknown"], ["--pair", "D:M", "--pair", "D:M"]):
            self.assertEqual(self.call(["validate"] + self.base + extra)[0], 2)
        self.assertEqual(self.call(["generate"] + self.base + ["--limit", "0"])[0], 2)
        self.assertEqual(self.call(["validate"] + self.base + ["--report", str(self.root / "samples/report.json")])[0], 2)

    def test_source_entry_from_different_cwd(self):
        result = subprocess.run([sys.executable, str(REPO / "run.py"), "validate"] + self.base,
                                cwd=str(self.root), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["pair_count"], 1)

    def test_keyboard_interrupt_returns_non_success(self):
        with patch("cd_error_maps.cli.generate", side_effect=KeyboardInterrupt):
            self.assertEqual(self.call(["generate"] + self.base)[0], 130)


if __name__ == "__main__":
    unittest.main()
