"""Configuration and portable path validation, independent of real data."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cd_error_maps.common import CDMapError, check_unique_paths, output_id, safe_join, validate_relative
from cd_error_maps.config import effective_config, encoding_for, load_config

DEFAULT_CONFIG = object()


def temporary_directory():
    scratch = Path(__file__).resolve().parents[1] / ".tmp"
    scratch.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=scratch)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root / "config.json"
        self.base = {"schema_version": 1, "samples_manifest": "data/samples.csv",
                     "datasets": ["D"], "models": ["M"]}

    def load(self, values=DEFAULT_CONFIG):
        self.file.write_text(json.dumps(self.base if values is DEFAULT_CONFIG else values), encoding="utf-8")
        with patch("cd_error_maps.config.PROJECT_ROOT", self.root):
            return load_config(self.file)

    def test_default_paths_and_config_path_are_project_relative_not_cwd(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.file.write_text(json.dumps(self.base), encoding="utf-8")
        old_cwd = Path.cwd()
        try:
            os.chdir(elsewhere)
            with patch("cd_error_maps.config.PROJECT_ROOT", self.root):
                config = load_config("config.json")
        finally:
            os.chdir(old_cwd)
        self.assertEqual(config["samples_root"], self.root / "data/fixed_samples")
        self.assertEqual(config["samples_manifest"], self.root / "data/samples.csv")
        self.assertEqual(config["prediction_root"], self.root / "data/predictions")
        self.assertEqual(config["output_root"], self.root / "outputs")
        self.assertEqual(config["prediction_path_template"], "{dataset}/{model}/{sample_id}")
        self.assertEqual(encoding_for(config, "D", "M"), ("binary_0255", "binary_0255"))
        self.assertEqual(effective_config(config)["samples_root"], str(self.root / "data/fixed_samples"))

    def test_default_config_prefers_local_override(self):
        configs = self.root / "configs"
        configs.mkdir()
        example = dict(self.base, gt_encoding="binary_01")
        (configs / "fixed_samples.example.json").write_text(json.dumps(example), encoding="utf-8")
        with patch("cd_error_maps.config.PROJECT_ROOT", self.root):
            self.assertEqual(load_config()["gt_encoding"], "binary_01")
        (configs / "local.fixed100.json").write_text(json.dumps(self.base), encoding="utf-8")
        with patch("cd_error_maps.config.PROJECT_ROOT", self.root):
            self.assertEqual(load_config()["gt_encoding"], "binary_0255")

    def test_malformed_types_are_actionable(self):
        malformed = {
            "schema_version": [None, True, 2, "1"],
            "samples_root": [None, "", 1, []],
            "prediction_root": [None, "", {}, 0],
            "output_root": [None, "", [], False],
            "datasets": [None, [], "D", ["D", "d"], [1], ["D/sub"]],
            "models": [None, [], "M", ["M", "M"], [False]],
            "combinations": [None, {}, "D:M", [None], [{"dataset": "D"}],
                             [{"dataset": [], "model": "M"}]],
            "prediction_path_template": [False, 1, {}, [], ""],
            "encodings": [None, [], "D", {"D": []}, {"D": {"M": None}},
                          {"D": {"M": {"gt_encoding": []}}}],
            "expected_counts": [None, [], "D", {"D": True}, {"D": 0}, {"D": "1"}],
            "expected_size": [True, [], [2], [2, 0], [True, 2]],
            "palette": [None, [], {}, {"TN": [0, 0, 0]}],
            "ignore": [[], True, {"gt_value": True}, {"gt_value": 256}],
            "model_aliases": [None, [], {"unknown": ["M1"]}, {"M": "Alias"}, {"M": [1]}],
        }
        for key, values in malformed.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(CDMapError):
                    self.load(dict(self.base, **{key: value}))
        for root in (None, [], "config", {"schema_version": 1, "unknown": True}):
            with self.subTest(root=root), self.assertRaises(CDMapError):
                self.load(root)

    def test_duplicates_unknown_pairs_and_alias_ambiguity(self):
        pair = {"dataset": "D", "model": "M"}
        with self.assertRaisesRegex(CDMapError, "Duplicate combinations"):
            self.load(dict(self.base, combinations=[pair, pair]))
        with self.assertRaisesRegex(CDMapError, "Invalid combinations"):
            self.load(dict(self.base, combinations=[{"dataset": "d", "model": "M"}]))
        with self.assertRaisesRegex(CDMapError, "ambiguous"):
            self.load(dict(self.base, models=["M", "N"], model_aliases={"M": ["n"]}))
        self.load(dict(self.base, model_aliases={"M": ["m", "Model"]}))

    def test_encoding_overrides_ignore_and_invalid_hashes(self):
        config = self.load(dict(self.base, encodings={"D": {"M": {"gt_encoding": "binary_01"}}},
                                ignore={"gt_value": 255}, expected_size=[3, 2], expected_counts={"D": 1}))
        self.assertEqual(encoding_for(config, "D", "M"), ("binary_01", "binary_0255"))
        for encoding in ("auto", None, 0):
            with self.subTest(encoding=encoding), self.assertRaises(CDMapError):
                self.load(dict(self.base, gt_encoding=encoding))
        for value in (0, 255):
            with self.subTest(ignore=value), self.assertRaisesRegex(CDMapError, "overlaps"):
                self.load(dict(self.base, ignore={"gt_value": value}))
        for key in ("expected_selection_sha256", "expected_manifest_sha256", "expected_prediction_manifest_sha256"):
            for invalid in ("A" * 64, "a" * 63, [], 0):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(CDMapError):
                    self.load(dict(self.base, **{key: invalid}))

    def test_manifest_selection_and_prediction_modes_are_exclusive(self):
        for changed in ({"samples_manifest": None}, {"selection_file": "selection.json"},
                        {"prediction_manifest": "prediction.csv", "prediction_path_template": "{dataset}/{model}/{sample_id}"},
                        {"prediction_path_template": None}):
            with self.subTest(changed=changed), self.assertRaisesRegex(CDMapError, "exactly one"):
                self.load(dict(self.base, **changed))
        config = self.load(dict(self.base, prediction_manifest="prediction.csv"))
        self.assertIsNone(config["prediction_path_template"])

    def test_invalid_templates_and_output_overlap(self):
        for template in ("{dataset}/{model}/{sample_id}/{sample_id}", "../{dataset}/{model}/{sample_id}",
                         "{dataset}/{model}/{sample_id!r}", "{dataset}/{model}/{sample_id.stem}",
                         "{dataset}/{model}/{sample_id:>20}", "{dataset}/{model}/{missing}",
                         "{dataset}/{model}/{sample_id"):
            with self.subTest(template=template), self.assertRaises(CDMapError):
                self.load(dict(self.base, prediction_path_template=template))
        for output in ("data", "data/fixed_samples", "data/fixed_samples/generated", "DATA/FIXED_SAMPLES/generated",
                       "data/predictions/runs"):
            with self.subTest(output=output), self.assertRaisesRegex(CDMapError, "overlaps"):
                self.load(dict(self.base, output_root=output))


class PathTests(unittest.TestCase):
    def test_preserves_case_zero_extension_subdirectory(self):
        value = "Sub/0001.PNG"
        self.assertEqual(validate_relative(value), value)
        self.assertEqual(output_id(value), value)
        self.assertEqual(output_id("Sub/0001.jpg"), "Sub/0001.jpg.png")

    def test_rejects_traversal_absolute_reserved_and_nonportable(self):
        for value in (None, 0, "", "/A.png", "../A.png", "a/../b", "a//b", "a/./b", "a\\b",
                      "C:/A.png", "a:b", "a?b", "CON", "NUL.png", "COM1.txt", "a. ", "a.", "a\x00b"):
            with self.subTest(path=value), self.assertRaises(CDMapError):
                validate_relative(value)

    def test_collisions_casefold_ancestors_file_conflict_and_generator(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            conflicts = [([root / "A.png", root / "a.png"], "collision"),
                         ([root / "Group/a.png", root / "group/b.png"], "casing collision"),
                         ([root / "group", root / "group/b.png"], "file/directory")]
            for paths, message in conflicts:
                with self.subTest(paths=paths), self.assertRaisesRegex(CDMapError, message):
                    check_unique_paths((path for path in paths), "Synthetic")
            check_unique_paths([root / "Group/a.png", root / "Group/b.png"], "Synthetic")

    def test_symlink_escape_is_rejected_when_platform_permits(self):
        with temporary_directory() as temporary:
            base = Path(temporary)
            root = base / "root"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "link").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest("Symlink creation unavailable: {}".format(exc))
            with self.assertRaisesRegex(CDMapError, "escapes"):
                safe_join(root, "link/mask.png")

    def test_resolved_escape_rejection_without_symlink_privilege(self):
        # Explicitly exercise the containment check on all platforms; a
        # separate test above creates a real symlink when OS policy permits.
        with temporary_directory() as temporary:
            base = Path(temporary)
            root = base / "root"
            escaped = base / "outside/mask.png"
            original_resolve = Path.resolve

            def resolve(path, *args, **kwargs):
                if path == root / "link/mask.png":
                    return escaped
                return original_resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", resolve), self.assertRaisesRegex(CDMapError, "escapes"):
                safe_join(root, "link/mask.png")


if __name__ == "__main__":
    unittest.main()
