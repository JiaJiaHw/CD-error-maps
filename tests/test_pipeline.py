"""End-to-end synthetic runs, hand-counted pixels, and integrity failures."""

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


def csv_file(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(temporary_root))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.samples = self.base / "samples"
        self.predictions = self.base / "predictions"
        self.run = self.base / "results" / "run"
        self.combinations = [("D1", "M1"), ("D2", "M2")]
        self.ids = ["sub/0001.PNG", "002.png"]
        self.rows = []
        self.arrays = {
            ("D1", "sub/0001.PNG"): ([[0, 0], [255, 255]], [[0, 255], [255, 0]]),
            ("D1", "002.png"): ([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
            ("D2", "sub/0001.PNG"): ([[255, 255], [255, 255]], [[255, 255], [255, 255]]),
            ("D2", "002.png"): ([[255, 255], [255, 255]], [[0, 0], [0, 0]])}
        for dataset, model in self.combinations:
            for rank, sample_id in enumerate(self.ids, 1):
                gt, prediction = self.arrays[dataset, sample_id]
                gt_path = self.samples / dataset / "GT" / sample_id
                pred_path = self.predictions / dataset / model / sample_id
                self.mask(gt_path, gt)
                self.mask(pred_path, prediction)
                self.rows.append({"dataset": dataset, "sample_id": sample_id, "rank": rank,
                                  "gt_relative_path": dataset + "/GT/" + sample_id})
        self.manifest = self.base / "samples.csv"
        csv_file(self.manifest, ["dataset", "sample_id", "rank", "gt_relative_path"], self.rows)
        self.config_file = self.base / "config.json"
        self.raw_config = {"schema_version": 1, "samples_root": str(self.samples),
                           "samples_manifest": str(self.manifest), "prediction_root": str(self.predictions),
                           "output_root": str(self.base / "results"), "datasets": ["D1", "D2"],
                           "models": ["M1", "M2"], "expected_counts": {"D1": 2, "D2": 2},
                           "expected_size": [2, 2], "gt_encoding": "binary_0255", "prediction_encoding": "binary_0255",
                           "combinations": [{"dataset": d, "model": m} for d, m in self.combinations], "palette": PALETTE}
        self.configure()
        version_patch = mock.patch.object(pipeline, "_version_record", return_value={"test_fixture": "synthetic only"})
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def configure(self, **changes):
        self.raw_config.update(changes)
        json_file(self.config_file, self.raw_config)
        self.config = load_config(self.config_file)

    def mask(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array(values, dtype=np.uint8)).save(path, format="PNG")

    def generate(self, **kwargs):
        return pipeline.generate(self.config, self.combinations, out=self.run, **kwargs)

    def report(self):
        return json.loads((self.run / "run_manifest.json").read_text(encoding="utf-8"))

    def reseal(self, artifacts=(), pair_hashes=False, manifest=None):
        manifest = manifest or self.report()
        for relative in artifacts:
            manifest["artifact_hashes"][relative] = sha256(self.run / relative)
        if pair_hashes:
            for pair in manifest["pairs"]:
                for kind in ("error_class", "error_rgb"):
                    pair[kind + "_sha256"] = sha256(self.run / pair[kind])
        json_file(self.run / "run_manifest.json", manifest)
        seal = json.loads((self.run / "COMPLETE.json").read_text())
        seal["run_manifest_sha256"] = sha256(self.run / "run_manifest.json")
        json_file(self.run / "COMPLETE.json", seal)

    def assert_failed_run(self, status="failed"):
        self.assertTrue(self.run.is_dir())
        self.assertFalse((self.run / "COMPLETE.json").exists())
        self.assertEqual(self.report()["status"], status)
        with self.assertRaisesRegex(CDMapError, "incomplete or failed"):
            pipeline.verify(self.run)

    def test_full_multigroup_manual_class_rgb_counts_and_micro_summary(self):
        report = self.generate()
        self.assertEqual(report["generated_pair_count"], 4)
        self.assertFalse(report["preview"])
        verified = pipeline.verify(self.run)
        self.assertEqual(verified["status"], "verified")
        manifest = self.report()
        first = manifest["pairs"][0]
        with Image.open(self.run / first["error_class"]) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(np.array(image).tolist(), [[0, 2], [1, 3]])
        with Image.open(self.run / first["error_rgb"]) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(np.array(image).tolist(), [[[0, 0, 0], [255, 0, 0]], [[255, 255, 255], [0, 255, 0]]])
        self.assertEqual(first["counts"], {"TN": 1, "TP": 1, "FP": 1, "FN": 1, "ignored": 0, "valid": 4, "width": 2, "height": 2})
        summaries = {(row["dataset"], row["model"]): row for row in manifest["summary"]}
        a = summaries["D1", "M1"]
        self.assertEqual([a[k] for k in ("images", "TN", "TP", "FP", "FN", "ignored", "valid")], [2, 5, 1, 1, 1, 0, 8])
        self.assertEqual([a[k] for k in ("precision", "recall", "f1", "accuracy")], [0.5, 0.5, 0.5, 0.75])
        self.assertAlmostEqual(a["iou"], 1 / 3)
        b = summaries["D2", "M2"]
        self.assertEqual([b[k] for k in ("images", "TN", "TP", "FP", "FN", "valid")], [2, 0, 4, 0, 4, 8])
        self.assertEqual([b[k] for k in ("precision", "recall", "iou", "accuracy")], [1, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(b["f1"], 2 / 3)

    def test_preview_is_first_per_group_and_validates_full_scope(self):
        report = self.generate(limit=1)
        self.assertEqual(report["generated_pair_count"], 2)
        self.assertEqual(report["validated_pair_count"], 4)
        self.assertTrue(report["preview"])
        self.assertEqual([pair["sample_id"] for pair in self.report()["pairs"]], ["sub/0001.PNG", "sub/0001.PNG"])
        self.assertTrue(pipeline.verify(self.run)["preview"])

    def test_preview_cannot_skip_missing_later_prediction(self):
        (self.predictions / "D1/M1/002.png").unlink()
        with self.assertRaisesRegex(CDMapError, "Prediction not ready"):
            self.generate(limit=1)
        self.assertFalse(self.run.exists())

    def test_missing_gt_size_illegal_encoding_and_manifest_hash_fail_before_output(self):
        original = (self.samples / "D1/GT/002.png").read_bytes()
        (self.samples / "D1/GT/002.png").unlink()
        with self.assertRaises((CDMapError, OSError)):
            self.generate()
        self.assertFalse(self.run.exists())
        (self.samples / "D1/GT/002.png").write_bytes(original)
        self.mask(self.predictions / "D1/M1/002.png", [[0, 0, 0], [0, 0, 0]])
        with self.assertRaisesRegex(CDMapError, "size mismatch"):
            self.generate()
        self.assertFalse(self.run.exists())
        self.mask(self.predictions / "D1/M1/002.png", [[0, 8], [0, 0]])
        with self.assertRaisesRegex(CDMapError, "Illegal values"):
            self.generate()
        self.assertFalse(self.run.exists())
        self.mask(self.predictions / "D1/M1/002.png", [[0, 0], [0, 0]])
        self.configure(expected_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch"):
            self.generate()
        self.assertFalse(self.run.exists())

    def test_declared_mask_hash_mismatch_fails_before_output(self):
        rows = [dict(row, gt_sha256="0" * 64) for row in self.rows]
        csv_file(self.manifest, ["dataset", "sample_id", "rank", "gt_relative_path", "gt_sha256"], rows)
        with self.assertRaisesRegex(CDMapError, "GT hash mismatch"):
            self.generate()
        self.assertFalse(self.run.exists())

    def test_samples_only_accepts_empty_predictions_and_is_explicit(self):
        import shutil
        shutil.rmtree(self.predictions)
        report = pipeline.validate_config_inputs(self.config, self.combinations, samples_only=True)
        self.assertTrue(report["samples_only"])
        self.assertFalse(report["prediction_checked"])
        self.assertEqual(report["pair_count"], 4)
        self.assertFalse(self.run.exists())
        with self.assertRaisesRegex(CDMapError, "Prediction not ready"):
            pipeline.validate_config_inputs(self.config, self.combinations)

    def test_explicit_mapping_distinct_names_hashes_and_provenance(self):
        rows = []
        for dataset, model in self.combinations:
            for index, sample_id in enumerate(self.ids):
                old = self.predictions / dataset / model / sample_id
                relative = "named/" + dataset + "/" + model + "/result_" + str(index) + ".png"
                new = self.predictions / relative
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
                rows.append({"dataset": dataset, "model": model, "sample_id": sample_id,
                             "prediction_relative_path": relative, "prediction_sha256": sha256(new),
                             "provenance_reference": "synthetic fixture only"})
        manifest = self.base / "predictions.csv"
        csv_file(manifest, list(rows[0]), rows)
        self.configure(prediction_manifest=str(manifest), prediction_path_template=None,
                       expected_prediction_manifest_sha256=sha256(manifest))
        self.generate()
        pipeline.verify(self.run)
        report = self.report()
        validation = json.loads((self.run / "validation_report.json").read_text())
        self.assertEqual(validation["manifest_hashes"][str(manifest)], sha256(manifest))
        self.assertEqual(report["pairs"][0]["provenance"]["provenance_reference"], "synthetic fixture only")

    def test_binary_01_gt_and_0255_prediction_produce_manual_answer(self):
        for dataset, model in self.combinations:
            for sample_id in self.ids:
                values = np.array(self.arrays[dataset, sample_id][0], dtype=np.uint8) // 255
                self.mask(self.samples / dataset / "GT" / sample_id, values)
        self.configure(gt_encoding="binary_01")
        self.generate()
        pair = self.report()["pairs"][0]
        with Image.open(self.run / pair["error_class"]) as image:
            self.assertEqual(np.array(image).tolist(), [[0, 2], [1, 3]])
        self.assertEqual(pair["gt_encoding"], "binary_01")
        self.assertEqual(pair["prediction_encoding"], "binary_0255")

    def test_ignore_semantics_and_statistics(self):
        self.mask(self.samples / "D1/GT/sub/0001.PNG", [[0, 128], [255, 128]])
        self.mask(self.samples / "D1/GT/002.png", [[128, 128], [128, 128]])
        self.configure(ignore={"gt_value": 128})
        self.generate()
        pair = self.report()["pairs"][0]
        with Image.open(self.run / pair["error_class"]) as image:
            self.assertEqual(np.array(image).tolist(), [[0, 255], [1, 255]])
        summary = self.report()["summary"][0]
        self.assertEqual([summary[k] for k in ("TN", "TP", "FP", "FN", "ignored", "valid")], [1, 1, 0, 0, 6, 2])
        pipeline.verify(self.run)

    def test_prediction_illegal_values_at_ignored_gt_remain_errors(self):
        self.mask(self.samples / "D1/GT/sub/0001.PNG", [[128, 128], [128, 128]])
        self.mask(self.predictions / "D1/M1/sub/0001.PNG", [[128, 128], [128, 128]])
        self.configure(ignore={"gt_value": 128})
        with self.assertRaisesRegex(CDMapError, "Illegal values"):
            self.generate()
        self.assertFalse(self.run.exists())

    def test_zero_denominator_metrics_are_null_and_csv_empty(self):
        for sample_id in self.ids:
            self.mask(self.samples / "D1/GT" / sample_id, [[0, 0], [0, 0]])
            self.mask(self.predictions / "D1/M1" / sample_id, [[0, 0], [0, 0]])
        self.generate()
        summary = self.report()["summary"][0]
        self.assertIsNone(summary["precision"])
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["f1"])
        self.assertIsNone(summary["iou"])
        self.assertEqual(summary["accuracy"], 1)
        self.assertEqual(summary["undefined_metrics"], "precision,recall,f1,iou")
        with (self.run / "summary.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["precision"], "")
        pipeline.verify(self.run)

    def test_refuses_existing_run_and_input_overlap(self):
        self.generate()
        before = sha256(self.run / "run_manifest.json")
        with self.assertRaisesRegex(CDMapError, "refusing overwrite"):
            self.generate()
        self.assertEqual(sha256(self.run / "run_manifest.json"), before)
        with self.assertRaisesRegex(CDMapError, "overlaps input"):
            pipeline.generate(self.config, self.combinations, out=self.samples / "new-run")
        self.assertFalse((self.samples / "new-run").exists())

    def test_casefold_input_roots_and_manifest_overlap_rejected(self):
        alternate_samples = self.samples.with_name(self.samples.name.upper())
        with self.assertRaisesRegex(CDMapError, "overlaps input"):
            pipeline.generate(self.config, self.combinations, out=alternate_samples / "new-run")
        with self.assertRaisesRegex(CDMapError, "overlaps input manifest"):
            pipeline._check_run_location(self.manifest.with_name(self.manifest.name.upper()), self.config)

    def test_malformed_manifest_and_nested_config_rejected_with_domain_error(self):
        self.generate()
        original = self.report()
        for changed in (dict(original, pairs=None), dict(original, pairs=[None] * 4),
                        dict(original, scope=[["D1"]]), dict(original, generated_pair_count="4")):
            self.reseal(manifest=changed)
            with self.subTest(changed=changed), self.assertRaises(CDMapError):
                pipeline.verify(self.run)
        self.reseal(manifest=original)
        effective_path = self.run / "effective_config.json"
        effective = json.loads(effective_path.read_text())
        effective["combinations"] = None
        json_file(effective_path, effective)
        self.reseal(["effective_config.json"])
        with self.assertRaisesRegex(CDMapError, "combinations must be a list"):
            pipeline.verify(self.run)

    def test_interruption_during_version_or_generation_is_incomplete(self):
        with mock.patch.object(pipeline, "_version_record", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.generate()
        self.assert_failed_run("incomplete")
        self.run = self.base / "results" / "interrupted-after-status"
        with mock.patch.object(pipeline, "write_legend", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.generate()
        self.assert_failed_run("incomplete")

    def test_forced_io_failure_preserves_partial_outputs_without_seal(self):
        with mock.patch.object(pipeline, "_write_csv", side_effect=OSError("forced disk failure")):
            with self.assertRaisesRegex(OSError, "forced disk failure"):
                self.generate()
        self.assert_failed_run()
        self.assertTrue((self.run / "error_class/D1/M1/sub/0001.PNG").exists())

    def test_partial_completion_write_failure_removes_marker(self):
        original_json = pipeline._json

        def fail_completion(path, value, exclusive=False):
            if Path(path).name == "COMPLETE.json":
                Path(path).write_text('{"status":"complete",', encoding="utf-8")
                raise OSError("completion write failure")
            return original_json(path, value, exclusive)

        with mock.patch.object(pipeline, "_json", side_effect=fail_completion):
            with self.assertRaisesRegex(OSError, "completion write failure"):
                self.generate()
        self.assert_failed_run()

    def test_original_io_error_is_retained_if_failure_status_write_fails(self):
        original_json = pipeline._json

        def fail_status(path, value, exclusive=False):
            if value.get("status") == "failed":
                raise OSError("secondary failure")
            return original_json(path, value, exclusive)

        with mock.patch.object(pipeline, "_json", side_effect=fail_status), mock.patch.object(pipeline, "write_legend", side_effect=OSError("original failure")):
            with self.assertRaisesRegex(OSError, "original failure"):
                self.generate()
        self.assertFalse((self.run / "COMPLETE.json").exists())
        with self.assertRaises(CDMapError):
            pipeline.verify(self.run)

    def test_input_changed_after_validation_fails_run(self):
        original_validation = pipeline.validate_config_inputs

        def mutate_after_validation(config, combinations, samples_only=False):
            report = original_validation(config, combinations, samples_only)
            self.mask(self.predictions / "D1/M1/sub/0001.PNG", [[0, 0], [0, 0]])
            return report

        with mock.patch.object(pipeline, "validate_config_inputs", side_effect=mutate_after_validation):
            with self.assertRaisesRegex(CDMapError, "Input changed after validation"):
                self.generate()
        self.assert_failed_run()

    def test_preview_unused_input_changed_after_validation_still_fails(self):
        original_write = pipeline._write_csv

        def mutate_before_csv(path, fields, rows):
            self.mask(self.predictions / "D1/M1/002.png", [[255, 0], [0, 0]])
            return original_write(path, fields, rows)

        with mock.patch.object(pipeline, "_write_csv", side_effect=mutate_before_csv):
            with self.assertRaises(CDMapError):
                self.generate(limit=1)
        self.assert_failed_run()

    def test_seal_missing_or_modified_and_extra_file_are_rejected(self):
        self.generate()
        seal_path = self.run / "COMPLETE.json"
        original = seal_path.read_bytes()
        seal_path.unlink()
        with self.assertRaises(CDMapError):
            pipeline.verify(self.run)
        seal_path.write_bytes(original)
        seal = json.loads(original)
        seal["generated_pair_count"] = 99
        json_file(seal_path, seal)
        with self.assertRaisesRegex(CDMapError, "Completion seal mismatch"):
            pipeline.verify(self.run)
        seal_path.write_bytes(original)
        (self.run / "unexpected.txt").write_text("tamper")
        with self.assertRaisesRegex(CDMapError, "Unexpected or missing"):
            pipeline.verify(self.run)

    def test_output_hash_tampering_is_rejected(self):
        self.generate()
        path = self.run / self.report()["pairs"][0]["error_class"]
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(CDMapError, "Output hash mismatch"):
            pipeline.verify(self.run)

    def test_resealed_class_tampering_is_rejected_from_input_truth(self):
        self.generate()
        relative = self.report()["pairs"][0]["error_class"]
        self.mask(self.run / relative, [[0, 0], [0, 0]])
        self.reseal([relative], pair_hashes=True)
        with self.assertRaisesRegex(CDMapError, "class pixels differ"):
            pipeline.verify(self.run)

    def test_resealed_rgb_tampering_is_rejected_from_index_and_palette(self):
        self.generate()
        relative = self.report()["pairs"][0]["error_rgb"]
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(self.run / relative)
        self.reseal([relative], pair_hashes=True)
        with self.assertRaisesRegex(CDMapError, "RGB differs"):
            pipeline.verify(self.run)

    def test_resealed_csv_tampering_is_rejected_semantically(self):
        self.generate()
        for relative in ("pairs.csv", "pixel_counts.csv", "summary.csv"):
            path = self.run / relative
            original = path.read_bytes()
            path.write_bytes(original + b"tampered row\n")
            self.reseal([relative])
            with self.subTest(relative=relative), self.assertRaisesRegex(CDMapError, "CSV contents mismatch"):
                pipeline.verify(self.run)
            path.write_bytes(original)
            self.reseal([relative])

    def test_resealed_counts_summary_scope_and_validation_metadata_rejected(self):
        self.generate()
        original = self.report()
        changed = self.report()
        changed["pairs"][0]["counts"]["TN"] = 88
        self.reseal(manifest=changed)
        with self.assertRaisesRegex(CDMapError, "Pixel counts differ"):
            pipeline.verify(self.run)
        self.reseal(manifest=original)
        changed = self.report()
        changed["summary"][0]["TN"] = 88
        self.reseal(manifest=changed)
        with self.assertRaisesRegex(CDMapError, "Aggregate statistics mismatch"):
            pipeline.verify(self.run)
        self.reseal(manifest=original)
        changed = self.report()
        changed["scope"] = ["wrong"]
        self.reseal(manifest=changed)
        with self.assertRaisesRegex(CDMapError, "Invalid run scope"):
            pipeline.verify(self.run)
        self.reseal(manifest=original)
        validation_path = self.run / "validation_report.json"
        validation = json.loads(validation_path.read_text())
        validation["prediction_checked"] = False
        json_file(validation_path, validation)
        self.reseal(["validation_report.json"])
        with self.assertRaisesRegex(CDMapError, "Input snapshot or mapping changed"):
            pipeline.verify(self.run)

    def test_source_input_and_manifest_modified_after_run_rejected(self):
        self.generate()
        path = self.predictions / "D1/M1/002.png"
        original = path.read_bytes()
        self.mask(path, [[255, 0], [0, 0]])
        with self.assertRaisesRegex(CDMapError, "Input snapshot or mapping changed"):
            pipeline.verify(self.run)
        path.write_bytes(original)
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(CDMapError, "Input snapshot or mapping changed"):
            pipeline.verify(self.run)

    def test_output_change_during_verify_detected_by_final_hash(self):
        self.generate()
        original_image_array = pipeline._image_array
        target = self.run / self.report()["pairs"][0]["error_rgb"]

        def mutate_after_decode(path, mode):
            array = original_image_array(path, mode)
            if Path(path) == target:
                Path(path).write_bytes(Path(path).read_bytes() + b"changed after decode")
            return array

        with mock.patch.object(pipeline, "_image_array", side_effect=mutate_after_decode):
            with self.assertRaisesRegex(CDMapError, "Output changed during verification"):
                pipeline.verify(self.run)

    def test_selection_arguments_choose_exact_combinations(self):
        self.assertEqual(pipeline.choose_combinations(self.config), self.combinations)
        self.assertEqual(pipeline.choose_combinations(self.config, explicit=["D2:M1", "D1:M2"]), [("D2", "M1"), ("D1", "M2")])
        self.assertEqual(pipeline.choose_combinations(self.config, datasets=["D1"], models=["M2"]), [("D1", "M2")])
        for kwargs in ({"explicit": ["D1:M1", "D1:M1"]}, {"explicit": ["D1:M1"], "datasets": ["D1"]},
                       {"explicit": ["unknown:M1"]}, {"explicit": ["broken"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(CDMapError):
                pipeline.choose_combinations(self.config, **kwargs)


if __name__ == "__main__":
    unittest.main()
