"""Independent synthetic frozen-list packages; no private server exporter import."""

import csv
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from cd_error_maps import pipeline
from cd_error_maps.common import CDMapError, sha256
from cd_error_maps.config import load_config
from cd_error_maps.import_fixed import import_archive
from cd_error_maps.manifests import load_selection_records


class FrozenTestListTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(temporary_root))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "SYNTHETIC-random-export"
        self.root.mkdir()
        self.ids = ["subset/0007.PNG", "Case_02.png"]
        self.selection = {"schema_version": 1, "sample_count_per_dataset": 2,
                          "sampling": {"method": "seeded_sampling_without_replacement"},
                          "datasets": {"SYNTHETIC": {"selection": []}}}
        self.data = self.selection["datasets"]["SYNTHETIC"]
        self.predictions = self.base / "SYNTHETIC-predictions"
        for rank, sample_id in enumerate(self.ids, 1):
            record = {"sample_id": sample_id, "rank": rank, "files": {}}
            for role in ("T1", "T2", "GT"):
                relative = "SYNTHETIC/" + role + "/" + sample_id
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if role == "GT":
                    values = np.array([[0, 0], [255, 255]], dtype=np.uint8)
                else:
                    values = np.full((2, 2, 3), 17 * rank, dtype=np.uint8)
                Image.fromarray(values).save(path)
                record["files"][role] = {"target": relative, "source": "/SYNTHETIC-only/" + relative,
                                          "bytes": path.stat().st_size, "sha256": sha256(path),
                                          "width": 2, "height": 2, "mode": "L" if role == "GT" else "RGB"}
            self.data["selection"].append(record)
            pred_path = self.predictions / "SYNTHETIC" / "SYNTHETIC-MODEL" / sample_id
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(pred_path)
        csv_path = self.root / "SYNTHETIC/samples.csv"
        fields = ["dataset", "rank", "sample_id", "width", "height"]
        for role in ("T1", "T2", "GT"):
            fields.extend(role + "_" + field for field in ("source", "target", "bytes", "sha256", "mode"))
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in self.data["selection"]:
                row = {"dataset": "SYNTHETIC", "sample_id": record["sample_id"],
                       "rank": record["rank"], "width": 2, "height": 2}
                for role in ("T1", "T2", "GT"):
                    for field in ("source", "target", "bytes", "sha256", "mode"):
                        row[role + "_" + field] = record["files"][role][field]
                writer.writerow(row)
        self.data["samples_csv_sha256"] = sha256(csv_path)
        self.test_path = self.root / "SYNTHETIC/test.txt"
        self.test_path.write_bytes(("\n".join(self.ids) + "\n").encode("utf-8"))
        self.data["test_txt_sha256"] = sha256(self.test_path)
        self.seal()
        self.scope = [("SYNTHETIC", "SYNTHETIC-MODEL")]
        self.run = self.base / "results" / "run"
        self.config_path = self.base / "config.json"
        self.config_path.write_text(json.dumps({"schema_version": 1, "samples_root": str(self.root),
                                                "selection_file": str(self.root / "selection.json"),
                                                "prediction_root": str(self.predictions),
                                                "output_root": str(self.base / "results"),
                                                "datasets": ["SYNTHETIC"], "models": ["SYNTHETIC-MODEL"],
                                                "gt_encoding": "binary_0255", "prediction_encoding": "binary_0255"}),
                                    encoding="utf-8")
        self.config = load_config(self.config_path)
        version_patch = mock.patch.object(pipeline, "_version_record", return_value={"fixture": "SYNTHETIC ONLY"})
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def seal(self):
        selection_path = self.root / "selection.json"
        selection_path.write_text(json.dumps(self.selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        digest = sha256(selection_path)
        (self.root / "selection.sha256").write_text(digest + "  selection.json\n", encoding="ascii")
        complete = {"status": "complete", "selection_sha256": digest,
                    "counts": {"SYNTHETIC": {"T1": 2, "T2": 2, "GT": 2}}, "total_images": 6,
                    "all_sha256_match_frozen_sources": True, "all_images_fully_decoded": True,
                    "all_triplet_dimensions_match": True, "original_names_and_bytes_preserved": True}
        (self.root / "EXPORT_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")

    def test_valid_list_preserves_exact_ids_order_and_hash(self):
        records, hashes, _ = load_selection_records(self.root)
        self.assertEqual([record["sample_id"] for record in records], self.ids)
        self.assertEqual(hashes[str(self.test_path)], sha256(self.test_path))

    def test_legacy_selection_without_list_seal_remains_compatible(self):
        del self.data["test_txt_sha256"]
        self.test_path.unlink()
        self.seal()
        records, hashes, _ = load_selection_records(self.root)
        self.assertEqual([record["sample_id"] for record in records], self.ids)
        self.assertNotIn(str(self.test_path), hashes)

    def test_declared_list_missing_rejected(self):
        self.test_path.unlink()
        with self.assertRaisesRegex(CDMapError, "Missing test.txt"):
            load_selection_records(self.root)

    def test_invalid_declared_digest_rejected(self):
        for value in (None, "", "0" * 63, "g" * 64, 123):
            self.data["test_txt_sha256"] = value
            self.seal()
            with self.subTest(digest=value), self.assertRaisesRegex(CDMapError, "Invalid SHA-256"):
                load_selection_records(self.root)

    def test_raw_list_tamper_rejected_even_with_same_ids(self):
        self.test_path.write_bytes(self.test_path.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch for test.txt"):
            load_selection_records(self.root)

    def test_resealed_reordered_duplicate_or_changed_id_rejected(self):
        for ids in (list(reversed(self.ids)), [self.ids[0], self.ids[0]],
                    ["subset/0007.png", self.ids[1]], ["subset/7.PNG", self.ids[1]], self.ids[:1]):
            self.test_path.write_bytes(("\n".join(ids) + "\n").encode("utf-8"))
            self.data["test_txt_sha256"] = sha256(self.test_path)
            self.seal()
            with self.subTest(ids=ids), self.assertRaisesRegex(CDMapError, "ID/order mismatch"):
                load_selection_records(self.root)

    def test_resealed_non_utf8_list_rejected(self):
        self.test_path.write_bytes(b"\xff\n")
        self.data["test_txt_sha256"] = sha256(self.test_path)
        self.seal()
        with self.assertRaisesRegex(CDMapError, "Cannot read test.txt"):
            load_selection_records(self.root)

    def test_list_path_escape_rejected(self):
        from cd_error_maps import manifests
        from cd_error_maps.common import safe_join

        def attempted_escape(root, relative):
            return safe_join(root, "../outside/test.txt" if relative.endswith("/test.txt") else relative)

        with mock.patch.object(manifests, "safe_join", side_effect=attempted_escape):
            with self.assertRaisesRegex(CDMapError, "Unsafe or nonportable"):
                load_selection_records(self.root)

    def test_synthetic_random_package_import_validate_generate_verify(self):
        archive = self.base / "SYNTHETIC-random.tar.gz"
        checksum = self.base / "SYNTHETIC-random.tar.gz.sha256"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(self.root, arcname="SYNTHETIC-one-outer-directory")
        checksum.write_text(sha256(archive) + "  " + archive.name + "\n", encoding="ascii")
        destination = self.base / "imported"
        imported = import_archive(archive, checksum, destination)
        self.assertEqual(imported["total_images"], 6)
        self.assertEqual((destination / "SYNTHETIC/test.txt").read_bytes(), self.test_path.read_bytes())
        self.config["samples_root"] = destination
        self.config["selection_file"] = destination / "selection.json"
        validation = pipeline.validate_config_inputs(self.config, self.scope)
        self.assertIn(str(destination / "SYNTHETIC/test.txt"), validation["manifest_hashes"])
        generated = pipeline.generate(self.config, self.scope, out=self.run)
        self.assertEqual(generated["generated_pair_count"], 2)
        self.assertEqual(pipeline.verify(self.run)["status"], "verified")
        report = json.loads((self.run / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([row["sample_id"] for row in report["pairs"]], self.ids)
        self.assertEqual([report["summary"][0][key] for key in ("TN", "TP", "FP", "FN", "valid")], [2, 2, 2, 2, 8])
        with Image.open(self.run / report["pairs"][0]["error_class"]) as image:
            self.assertEqual(np.array(image).tolist(), [[0, 2], [1, 3]])

    def test_list_change_during_generation_fails_without_complete_marker(self):
        original_read_pair = pipeline._read_pair

        def change_list(config, pair, expected=None):
            result = original_read_pair(config, pair, expected)
            if expected:
                self.test_path.write_bytes(self.test_path.read_bytes() + b"\n")
            return result

        with mock.patch.object(pipeline, "_read_pair", side_effect=change_list):
            with self.assertRaisesRegex(CDMapError, "Manifest changed after validation"):
                pipeline.generate(self.config, self.scope, out=self.run)
        self.assertFalse((self.run / "COMPLETE.json").exists())
        report = json.loads((self.run / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_verify_rejects_changed_source_list(self):
        pipeline.generate(self.config, self.scope, out=self.run)
        self.test_path.write_bytes(self.test_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch for test.txt"):
            pipeline.verify(self.run)


if __name__ == "__main__":
    unittest.main()
