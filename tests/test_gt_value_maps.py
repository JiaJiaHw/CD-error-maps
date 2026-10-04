"""Synthetic regressions for explicit, dataset-scoped GT value maps.

All PNG inputs, including Predictions, are created beneath this copy's .tmp.
No real dataset or model output is used by these tests.
"""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from cd_error_maps import pipeline
from cd_error_maps.common import CDMapError, sha256
from cd_error_maps.config import PALETTE, load_config
from cd_error_maps.masks import read_gt_mask, read_mask


GT_MAP = {"background": [0], "change": [254, 255]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class GTValueMapTests(unittest.TestCase):
    """Independent fixture; deliberately does not inherit PipelineTests."""

    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=str(temporary_root))
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.samples = self.base / "synthetic_samples"
        self.predictions = self.base / "synthetic_predictions"
        self.run = self.base / "results" / "run"
        self.combinations = [("D1", "M1"), ("D2", "M1")]
        self.sample_id = "synthetic.png"
        self.mask(self.gt_path("D1"), [[0, 0], [255, 255]])
        self.mask(self.pred_path("D1"), [[0, 255], [255, 0]])
        self.mask(self.gt_path("D2"), [[0, 255], [0, 255]])
        self.mask(self.pred_path("D2"), [[0, 255], [0, 255]])
        self.samples_manifest = self.base / "synthetic_samples.csv"
        with self.samples_manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "dataset", "sample_id", "rank", "gt_relative_path"],
                lineterminator="\n")
            writer.writeheader()
            for dataset, _ in self.combinations:
                writer.writerow({"dataset": dataset, "sample_id": self.sample_id,
                                 "rank": 1,
                                 "gt_relative_path": dataset + "/GT/" + self.sample_id})
        self.config_file = self.base / "synthetic_config.json"
        self.raw_config = {
            "schema_version": 1, "samples_root": str(self.samples),
            "samples_manifest": str(self.samples_manifest),
            "prediction_root": str(self.predictions),
            "output_root": str(self.base / "results"),
            "datasets": ["D1", "D2"], "models": ["M1"],
            "expected_counts": {"D1": 1, "D2": 1}, "expected_size": [2, 2],
            "gt_encoding": "binary_0255", "prediction_encoding": "binary_0255",
            "combinations": [{"dataset": d, "model": m} for d, m in self.combinations],
            "palette": copy.deepcopy(PALETTE)}
        self.configure()
        version_patch = mock.patch.object(
            pipeline, "_version_record", return_value={"test_fixture": "synthetic only"})
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def gt_path(self, dataset):
        return self.samples / dataset / "GT" / self.sample_id

    def pred_path(self, dataset):
        return self.predictions / dataset / "M1" / self.sample_id

    def mask(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array(values, dtype=np.uint8)).save(path, format="PNG")

    def configure(self, **changes):
        self.raw_config.update(copy.deepcopy(changes))
        write_json(self.config_file, self.raw_config)
        self.config = load_config(self.config_file)

    def mapped_fixture(self):
        # 254 is TP at [0, 1] and FN at [1, 0]; 255 is also TP.
        self.mask(self.gt_path("D1"), [[0, 254], [254, 255]])
        self.mask(self.pred_path("D1"), [[0, 255], [0, 255]])
        self.configure(gt_value_maps={"D1": GT_MAP})

    def generate(self, out=None):
        return pipeline.generate(self.config, self.combinations,
                                 out=self.run if out is None else out)

    def report(self, name="run_manifest.json"):
        return json.loads((self.run / name).read_text(encoding="utf-8"))

    def reseal(self, artifacts=(), manifest=None):
        manifest = self.report() if manifest is None else manifest
        for relative in artifacts:
            manifest["artifact_hashes"][relative] = sha256(self.run / relative)
        write_json(self.run / "run_manifest.json", manifest)
        seal = self.report("COMPLETE.json")
        seal["run_manifest_sha256"] = sha256(self.run / "run_manifest.json")
        write_json(self.run / "COMPLETE.json", seal)

    def tamper_effective_mapping(self):
        effective = self.report("effective_config.json")
        # An unused, legal value leaves pixel truth unchanged. Verification must
        # still reject a changed mapping declaration against the saved snapshot.
        effective["gt_value_maps"]["D1"]["change"].append(253)
        write_json(self.run / "effective_config.json", effective)

    def test_read_gt_maps_254_and_255_without_modifying_source_bytes(self):
        path = self.gt_path("D1")
        self.mask(path, [[0, 254], [255, 0]])
        before = path.read_bytes()
        before_hash = sha256(path)
        change, valid = read_gt_mask(path, "binary_0255", value_map=GT_MAP)
        self.assertEqual(change.dtype, np.dtype(np.bool_))
        self.assertEqual(valid.dtype, np.dtype(np.bool_))
        self.assertEqual(change.tolist(), [[False, True], [True, False]])
        self.assertEqual(valid.tolist(), [[True, True], [True, True]])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(sha256(path), before_hash)

    def test_mapping_can_declare_multiple_background_values_and_override_gt_encoding(self):
        path = self.gt_path("D1")
        self.mask(path, [[2, 254], [255, 0]])
        mapping = {"background": [0, 2], "change": [254, 255]}
        change, valid = read_gt_mask(path, "binary_01", value_map=mapping)
        self.assertEqual(change.tolist(), [[False, True], [True, False]])
        self.assertTrue(valid.all())

    def test_mapped_generation_has_hand_counted_tp_fn_and_sealed_provenance(self):
        self.mapped_fixture()
        before = {path: path.read_bytes() for path in self.samples.rglob("*.png")}
        self.generate()
        manifest = self.report()
        validation = self.report("validation_report.json")
        effective = self.report("effective_config.json")
        pair = manifest["pairs"][0]
        with Image.open(self.run / pair["error_class"]) as image:
            self.assertEqual(np.array(image).tolist(), [[0, 1], [3, 1]])
        self.assertEqual(pair["counts"], {"TN": 1, "TP": 2, "FP": 0, "FN": 1,
                                          "ignored": 0, "valid": 4,
                                          "width": 2, "height": 2})
        self.assertEqual(pair["gt_value_map"], GT_MAP)
        self.assertEqual(validation["gt_value_maps"], {"D1": GT_MAP})
        self.assertEqual(validation["pairs"][0]["gt_value_map"], GT_MAP)
        self.assertEqual(effective["gt_value_maps"], {"D1": GT_MAP})
        for relative in ("effective_config.json", "validation_report.json"):
            self.assertEqual(manifest["artifact_hashes"][relative], sha256(self.run / relative))
        self.assertEqual(self.report("COMPLETE.json")["run_manifest_sha256"],
                         sha256(self.run / "run_manifest.json"))
        self.assertEqual(pipeline.verify(self.run)["status"], "verified")
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_unknown_gt_values_are_rejected_without_implicit_encoding_union(self):
        for unknown, mapping in ((253, GT_MAP),
                                 (255, {"background": [0], "change": [254]}),
                                 (1, GT_MAP)):
            with self.subTest(unknown=unknown, mapping=mapping):
                self.mask(self.gt_path("D1"), [[0, unknown], [254, 0]])
                self.configure(gt_value_maps={"D1": mapping})
                with self.assertRaisesRegex(CDMapError, "Illegal values.*" + str(unknown)):
                    read_gt_mask(self.gt_path("D1"), "binary_0255", value_map=mapping)
                with self.assertRaisesRegex(CDMapError, "Illegal values.*" + str(unknown)):
                    self.generate()
                self.assertFalse(self.run.exists())

    def test_default_gt_254_is_rejected_for_both_binary_encodings(self):
        self.mask(self.gt_path("D1"), [[0, 254], [0, 0]])
        for encoding in ("binary_01", "binary_0255"):
            with self.subTest(encoding=encoding):
                with self.assertRaisesRegex(CDMapError, "Illegal values.*254"):
                    read_gt_mask(self.gt_path("D1"), encoding)
        with self.assertRaisesRegex(CDMapError, "Illegal values.*254"):
            self.generate()
        self.assertFalse(self.run.exists())

    def test_no_mapping_preserves_default_01_and_0255_decoding(self):
        path = self.gt_path("D1")
        for encoding, foreground in (("binary_01", 1), ("binary_0255", 255)):
            with self.subTest(encoding=encoding):
                self.mask(path, [[0, foreground], [foreground, 0]])
                gt, valid = read_gt_mask(path, encoding)
                strict, strict_valid = read_mask(path, encoding)
                self.assertEqual(gt.tolist(), [[False, True], [True, False]])
                self.assertTrue(np.array_equal(gt, strict))
                self.assertTrue(np.array_equal(valid, strict_valid))
                self.assertTrue(valid.all())

    def test_prediction_254_rejected_with_mapping_even_at_ignored_gt_pixel(self):
        self.mask(self.gt_path("D1"), [[0, 254], [255, 128]])
        self.configure(gt_value_maps={"D1": GT_MAP}, ignore={"gt_value": 128})
        for location in ((0, 0), (1, 1)):
            with self.subTest(location=location):
                prediction = np.array([[0, 255], [255, 0]], dtype=np.uint8)
                prediction[location] = 254
                self.mask(self.pred_path("D1"), prediction)
                with self.assertRaisesRegex(CDMapError, "Illegal values.*254"):
                    read_mask(self.pred_path("D1"), "binary_0255")
                with self.assertRaisesRegex(CDMapError, "Illegal values.*254"):
                    self.generate()
                self.assertFalse(self.run.exists())

    def test_prediction_reader_api_does_not_accept_gt_mapping(self):
        with self.assertRaises(TypeError):
            read_mask(self.pred_path("D1"), "binary_0255", value_map=GT_MAP)

    def test_mapping_applies_only_to_declared_dataset(self):
        self.mapped_fixture()
        self.generate()
        validation_pairs = {row["dataset"]: row for row in self.report("validation_report.json")["pairs"]}
        output_pairs = {row["dataset"]: row for row in self.report()["pairs"]}
        self.assertEqual(validation_pairs["D1"]["gt_value_map"], GT_MAP)
        self.assertNotIn("gt_value_map", validation_pairs["D2"])
        self.assertNotIn("gt_value_map", output_pairs["D2"])
        self.assertEqual(output_pairs["D2"]["counts"],
                         {"TN": 2, "TP": 2, "FP": 0, "FN": 0,
                          "ignored": 0, "valid": 4, "width": 2, "height": 2})
        self.assertEqual(pipeline.verify(self.run)["status"], "verified")
        self.mask(self.gt_path("D2"), [[0, 254], [0, 255]])
        rejected_run = self.base / "results" / "undeclared-dataset"
        with self.assertRaisesRegex(CDMapError, "Illegal values.*254"):
            self.generate(out=rejected_run)
        self.assertFalse(rejected_run.exists())

    def test_malformed_and_overlapping_gt_value_maps_are_configuration_errors(self):
        invalid = [
            ("null maps", None), ("array maps", []), ("string maps", "D1"),
            ("boolean maps", True), ("unknown dataset", {"D3": GT_MAP}),
            ("wrong dataset case", {"d1": GT_MAP}),
            ("null map", {"D1": None}), ("array map", {"D1": []}),
            ("scalar map", {"D1": 254}), ("empty map", {"D1": {}}),
            ("missing background", {"D1": {"change": [254]}}),
            ("missing change", {"D1": {"background": [0]}}),
            ("extra map key", {"D1": dict(GT_MAP, ignore=[128])}),
            ("empty background", {"D1": {"background": [], "change": [254]}}),
            ("empty change", {"D1": {"background": [0], "change": []}}),
            ("scalar background", {"D1": {"background": 0, "change": [254]}}),
            ("scalar change", {"D1": {"background": [0], "change": 254}}),
            ("duplicate background", {"D1": {"background": [0, 0], "change": [254]}}),
            ("duplicate change", {"D1": {"background": [0], "change": [254, 254]}}),
            ("class overlap", {"D1": {"background": [0, 254], "change": [254, 255]}})]
        for role in ("background", "change"):
            for label, value in (("boolean", True), ("float", 254.0), ("string", "254"),
                                 ("negative", -1), ("above uint8", 256),
                                 ("null", None), ("nested list", [254])):
                mapping = copy.deepcopy(GT_MAP)
                mapping[role] = [value]
                invalid.append((role + " " + label, {"D1": mapping}))
        for label, maps in invalid:
            with self.subTest(label=label):
                raw = copy.deepcopy(self.raw_config)
                raw["gt_value_maps"] = maps
                write_json(self.config_file, raw)
                with self.assertRaises(CDMapError):
                    load_config(self.config_file)

    def test_ignore_cannot_overlap_either_mapped_class(self):
        cases = [(GT_MAP, 0), (GT_MAP, 254), (GT_MAP, 255),
                 ({"background": [7], "change": [254]}, 7)]
        for mapping, ignore_value in cases:
            with self.subTest(mapping=mapping, ignore_value=ignore_value):
                raw = copy.deepcopy(self.raw_config)
                raw.update(gt_value_maps={"D1": mapping}, ignore={"gt_value": ignore_value})
                write_json(self.config_file, raw)
                with self.assertRaisesRegex(CDMapError, "ignore value overlaps GT value map"):
                    load_config(self.config_file)
                with self.assertRaises(CDMapError):
                    read_gt_mask(self.gt_path("D1"), "binary_0255",
                                 ignore_value=ignore_value, value_map=mapping)

    def test_independent_ignore_is_allowed_and_counted_separately(self):
        self.mask(self.gt_path("D1"), [[0, 128], [254, 255]])
        self.configure(gt_value_maps={"D1": GT_MAP}, ignore={"gt_value": 128})
        gt, valid = read_gt_mask(self.gt_path("D1"), "binary_0255", 128, GT_MAP)
        self.assertEqual(gt.tolist(), [[False, False], [True, True]])
        self.assertEqual(valid.tolist(), [[True, False], [True, True]])
        self.generate()
        pair = self.report()["pairs"][0]
        with Image.open(self.run / pair["error_class"]) as image:
            self.assertEqual(np.array(image).tolist(), [[0, 255], [1, 3]])
        self.assertEqual(pair["counts"], {"TN": 1, "TP": 1, "FP": 0, "FN": 1,
                                          "ignored": 1, "valid": 3,
                                          "width": 2, "height": 2})
        self.assertEqual(pipeline.verify(self.run)["status"], "verified")

    def test_samples_only_validates_mapping_without_predictions(self):
        self.mapped_fixture()
        self.predictions.rename(self.base / "withheld_synthetic_predictions")
        report = pipeline.validate_config_inputs(self.config, self.combinations, samples_only=True)
        self.assertTrue(report["samples_only"])
        self.assertFalse(report["prediction_checked"])
        self.assertEqual(report["gt_value_maps"], {"D1": GT_MAP})
        pairs = {row["dataset"]: row for row in report["pairs"]}
        self.assertEqual(pairs["D1"]["gt_value_map"], GT_MAP)
        self.assertNotIn("gt_value_map", pairs["D2"])
        self.assertTrue(all("prediction_path" not in pair for pair in report["pairs"]))
        self.assertFalse(self.run.exists())
        self.mask(self.gt_path("D1"), [[0, 253], [254, 255]])
        with self.assertRaisesRegex(CDMapError, "Illegal values.*253"):
            pipeline.validate_config_inputs(self.config, self.combinations, samples_only=True)

    def test_legacy_run_without_gt_value_maps_still_verifies(self):
        self.generate()
        validation = self.report("validation_report.json")
        self.assertNotIn("gt_value_maps", validation)
        self.assertTrue(all("gt_value_map" not in pair for pair in validation["pairs"]))
        effective = self.report("effective_config.json")
        effective.pop("gt_value_maps", None)
        write_json(self.run / "effective_config.json", effective)
        self.reseal(["effective_config.json"])
        self.assertEqual(pipeline.verify(self.run)["status"], "verified")

    def test_effective_mapping_tamper_is_rejected_by_artifact_hash(self):
        self.mapped_fixture()
        self.generate()
        self.tamper_effective_mapping()
        with self.assertRaisesRegex(CDMapError, "Output hash mismatch.*effective_config.json"):
            pipeline.verify(self.run)

    def test_resealed_effective_mapping_tamper_is_rejected_by_snapshot(self):
        self.mapped_fixture()
        self.generate()
        self.tamper_effective_mapping()
        self.reseal(["effective_config.json"])
        self.assertEqual(self.report()["artifact_hashes"]["effective_config.json"],
                         sha256(self.run / "effective_config.json"))
        self.assertEqual(self.report("COMPLETE.json")["run_manifest_sha256"],
                         sha256(self.run / "run_manifest.json"))
        with self.assertRaisesRegex(CDMapError, "Input snapshot or mapping changed"):
            pipeline.verify(self.run)

    def test_validation_mapping_tamper_is_rejected_before_and_after_resealing(self):
        self.mapped_fixture()
        self.generate()
        validation = self.report("validation_report.json")
        validation["gt_value_maps"]["D1"]["change"].append(253)
        write_json(self.run / "validation_report.json", validation)
        with self.assertRaisesRegex(CDMapError, "Output hash mismatch.*validation_report.json"):
            pipeline.verify(self.run)
        self.reseal(["validation_report.json"])
        with self.assertRaisesRegex(CDMapError, "Input snapshot or mapping changed"):
            pipeline.verify(self.run)

    def test_resealed_pair_mapping_tamper_is_rejected(self):
        self.mapped_fixture()
        self.generate()
        manifest = self.report()
        manifest["pairs"][0]["gt_value_map"]["change"].append(253)
        self.reseal(manifest=manifest)
        with self.assertRaisesRegex(CDMapError, "Pair mapping metadata mismatch"):
            pipeline.verify(self.run)


if __name__ == "__main__":
    unittest.main()
