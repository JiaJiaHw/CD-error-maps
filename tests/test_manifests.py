"""Small independent manifest fixtures, with no real Prediction files."""

import csv
import tempfile
import unittest
from pathlib import Path

from cd_error_maps.common import CDMapError, sha256
from cd_error_maps.manifests import load_samples, pair_samples


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"samples_root": self.root / "samples", "samples_manifest": self.root / "samples.csv",
                       "selection_file": None, "prediction_root": self.root / "predictions",
                       "prediction_manifest": None, "prediction_path_template": "{dataset}/{model}/{sample_id}"}
        self.rows = [{"dataset": "D", "sample_id": "sub/0001.PNG", "gt_relative_path": "D/GT/sub/0001.PNG"},
                     {"dataset": "D", "sample_id": "002.png", "gt_relative_path": "D/GT/002.png"}]
        self.samples()

    def samples(self, rows=None):
        write_csv(self.config["samples_manifest"], ["dataset", "sample_id", "gt_relative_path"], rows if rows is not None else self.rows)
        return load_samples(self.config)[0]

    def prediction_files(self, model="M"):
        for row in self.rows:
            path = self.config["prediction_root"] / "D" / model / row["sample_id"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic path fixture; not a mask")

    def mapping(self, rows):
        self.config["prediction_manifest"] = self.root / "predictions.csv"
        self.config["prediction_path_template"] = None
        fields = ["dataset", "model", "sample_id", "prediction_relative_path", "prediction_sha256", "provenance_reference"]
        write_csv(self.config["prediction_manifest"], fields, rows)

    def mapping_rows(self):
        return [{"dataset": "D", "model": "M", "sample_id": row["sample_id"],
                 "prediction_relative_path": "D/M/" + row["sample_id"]} for row in self.rows]

    def test_preserves_order_case_zero_extension_and_subdirectory(self):
        samples, hashes = load_samples(self.config)
        self.assertEqual([row["sample_id"] for row in samples], ["sub/0001.PNG", "002.png"])
        self.assertEqual(hashes[str(self.config["samples_manifest"])], sha256(self.config["samples_manifest"]))
        self.prediction_files()
        pairs = pair_samples(self.config, samples, [("D", "M")])
        self.assertEqual(pairs[0]["prediction_relative_path"], "D/M/sub/0001.PNG")

    def test_duplicate_sample_id_rejected(self):
        with self.assertRaisesRegex(CDMapError, "Duplicate sample"):
            self.samples(self.rows + [self.rows[0]])

    def test_case_collision_and_output_suffix_collision_rejected(self):
        for ids in [("A.png", "a.png"), ("A.jpg", "A.jpg.png")]:
            rows = [{"dataset": "D", "sample_id": value, "gt_relative_path": "GT/" + str(index) + ".png"} for index, value in enumerate(ids)]
            with self.subTest(ids=ids), self.assertRaisesRegex(CDMapError, "collision"):
                self.samples(rows)

    def test_gt_path_duplicate_and_parent_conflict_rejected(self):
        for paths in [("same.png", "same.png"), ("dir", "dir/child.png")]:
            rows = [dict(row, gt_relative_path=path) for row, path in zip(self.rows, paths)]
            with self.subTest(paths=paths), self.assertRaises(CDMapError):
                self.samples(rows)

    def test_traversal_absolute_nonportable_rejected(self):
        for path in ("../escape.png", "/escape.png", "C:/escape.png", "sub\\bad.png"):
            with self.subTest(path=path), self.assertRaises(CDMapError):
                self.samples([dict(self.rows[0], gt_relative_path=path)])

    def test_samples_hash_mismatch(self):
        self.config["expected_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch"):
            load_samples(self.config)

    def test_count_dimension_and_csv_column_checks(self):
        self.config["expected_counts"] = {"D": 3}
        with self.assertRaisesRegex(CDMapError, "count mismatch"):
            load_samples(self.config)
        del self.config["expected_counts"]
        write_csv(self.config["samples_manifest"], ["dataset", "sample_id", "gt_relative_path", "width"], [dict(self.rows[0], width=2)])
        with self.assertRaisesRegex(CDMapError, "width and height"):
            load_samples(self.config)
        self.config["samples_manifest"].write_text("dataset,dataset,sample_id,gt_relative_path\nD,D,a.png,a.png\n", encoding="utf-8")
        with self.assertRaisesRegex(CDMapError, "Duplicate CSV"):
            load_samples(self.config)

    def test_empty_and_partial_predictions_actionable(self):
        for prepare in (False, True):
            if prepare:
                path = self.config["prediction_root"] / "D/M" / self.rows[0]["sample_id"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")
            with self.subTest(partial=prepare), self.assertRaisesRegex(CDMapError, "Prediction not ready"):
                pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])

    def test_extra_prediction_image_and_metadata_allowed(self):
        self.prediction_files()
        group = self.config["prediction_root"] / "D/M"
        (group / "provenance.json").write_text("{}")
        pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])
        (group / "extra.jpeg").write_bytes(b"synthetic")
        with self.assertRaisesRegex(CDMapError, "Extra Prediction"):
            pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])

    def test_explicit_mapping_preserves_provenance_and_scope(self):
        self.prediction_files()
        rows = self.mapping_rows()
        rows[0]["prediction_sha256"] = "a" * 64
        rows[0]["provenance_reference"] = "synthetic fixture only"
        rows.append({"dataset": "unselected", "model": "other", "sample_id": "unrelated.png", "prediction_relative_path": "elsewhere.png"})
        self.mapping(rows)
        pairs = pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])
        self.assertEqual(pairs[0]["prediction_sha256"], "a" * 64)
        self.assertEqual(pairs[0]["provenance"]["provenance_reference"], "synthetic fixture only")

    def test_explicit_missing_duplicate_unknown_mapping(self):
        self.prediction_files()
        rows = self.mapping_rows()
        for changed, message in [(rows[:1], "missing mapping"), (rows + [rows[0]], "Duplicate Prediction"),
                                 (rows + [dict(rows[0], sample_id="unknown.png")], "Unknown Prediction")]:
            self.mapping(changed)
            with self.subTest(message=message), self.assertRaisesRegex(CDMapError, message):
                pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])

    def test_mapping_hash_mutual_exclusion_and_prediction_path_collision(self):
        self.prediction_files()
        self.mapping(self.mapping_rows())
        self.config["expected_prediction_manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch"):
            pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])
        del self.config["expected_prediction_manifest_sha256"]
        self.config["prediction_path_template"] = "{dataset}/{model}/{sample_id}"
        with self.assertRaisesRegex(CDMapError, "mutually exclusive"):
            pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])
        self.config["prediction_path_template"] = None
        rows = self.mapping_rows()
        rows[1]["prediction_relative_path"] = rows[0]["prediction_relative_path"]
        self.mapping(rows)
        # Remove canonical extra so that the duplicate mapping itself is examined.
        (self.config["prediction_root"] / "D/M/002.png").unlink()
        with self.assertRaisesRegex(CDMapError, "collision"):
            pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])

    def test_templates_reject_unknown_fields_formats_and_traversal(self):
        samples = load_samples(self.config)[0]
        for template in ("{sample_id.stem}", "{sample_id!r}", "../{sample_id}", "{sample_id:>20}", "{missing}"):
            self.config["prediction_path_template"] = template
            with self.subTest(template=template), self.assertRaises(CDMapError):
                pair_samples(self.config, samples, [("D", "M")])

    def test_prediction_cannot_alias_gt(self):
        self.prediction_files()
        self.config["samples_root"] = self.config["prediction_root"]
        self.samples([dict(row, gt_relative_path="D/M/" + row["sample_id"]) for row in self.rows])
        with self.assertRaisesRegex(CDMapError, "conflicts with GT"):
            pair_samples(self.config, load_samples(self.config)[0], [("D", "M")])


if __name__ == "__main__":
    unittest.main()
