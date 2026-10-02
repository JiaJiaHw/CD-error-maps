"""Byte-preservation and archive safety tests using tiny synthetic exports."""

import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cd_error_maps.common import CDMapError, sha256
from cd_error_maps.import_fixed import import_archive, verify_fixed_samples
from cd_error_maps.manifests import load_samples


def save_metadata(root, selection):
    fields = ["dataset", "rank", "sample_id", "source_list_entry", "sort_bytes", "width", "height"]
    for role in ("T1", "T2", "GT"):
        fields.extend(role + "_" + field for field in ("source", "target", "bytes", "sha256", "mode"))
    for dataset, data in selection["datasets"].items():
        csv_path = root / dataset / "samples.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in data["selection"]:
                first = record["files"]["T1"]
                row = {"dataset": dataset, "rank": record["rank"], "sample_id": record["sample_id"],
                       "source_list_entry": record["source_list_entry"], "sort_bytes": record["sort_bytes"],
                       "width": first["width"], "height": first["height"]}
                for role in ("T1", "T2", "GT"):
                    for field in ("source", "target", "bytes", "sha256", "mode"):
                        row[role + "_" + field] = record["files"][role][field]
                writer.writerow(row)
        data["samples_csv_sha256"] = sha256(csv_path)
    selection_path = root / "selection.json"
    selection_path.write_bytes((json.dumps(selection, indent=2) + "\n").encode("utf-8"))
    digest = sha256(selection_path)
    (root / "selection.sha256").write_text(digest + "  selection.json\n", encoding="ascii")
    counts = {dataset: {role: len(data["selection"]) for role in ("T1", "T2", "GT")} for dataset, data in selection["datasets"].items()}
    complete = {"status": "complete", "selection_sha256": digest, "counts": counts,
                "total_images": sum(sum(group.values()) for group in counts.values()),
                "all_sha256_match_frozen_sources": True, "all_images_fully_decoded": True,
                "all_triplet_dimensions_match": True, "original_names_and_bytes_preserved": True}
    (root / "EXPORT_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    return digest


def synthetic_export(root):
    root.mkdir()
    records = []
    for rank, sample_id in enumerate(("sub/0001.png", "Abc.png"), 1):
        record = {"sample_id": sample_id, "rank": rank, "source_list_entry": "", "files": {}}
        for role in ("T1", "T2", "GT"):
            relative = "D/" + role + "/" + sample_id
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if role == "GT":
                array = np.array([[0, 255], [255, 0]], dtype=np.uint8)
            else:
                array = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
            Image.fromarray(array).save(path)
            with Image.open(path) as image:
                record["files"][role] = {"source": "/historical/server/" + relative, "target": relative,
                                          "bytes": path.stat().st_size, "sha256": sha256(path),
                                          "width": image.width, "height": image.height, "mode": image.mode}
        record["sort_bytes"] = record["files"]["T1"]["bytes"] + record["files"]["T2"]["bytes"]
        records.append(record)
    selection = {"schema_version": 1, "sample_count_per_dataset": 2, "stage": "selection_frozen_no_images_copied",
                 "datasets": {"D": {"selection": records}}}
    save_metadata(root, selection)
    return selection


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "original-export"
        self.selection = synthetic_export(self.source)
        self.archive = self.base / "synthetic-export.tar.gz"
        self.checksum = self.base / "synthetic-export.tar.gz.sha256"
        self.destination = self.base / "imported"
        self.pack()

    def pack(self):
        with tarfile.open(self.archive, "w:gz") as handle:
            handle.add(self.source, arcname="one-outer-folder")
        self.checksum.write_text(sha256(self.archive) + "  " + self.archive.name + "\n", encoding="ascii")

    def run_import(self, **kwargs):
        return import_archive(self.archive, self.checksum, self.destination, **kwargs)

    def malformed_archive(self, entries):
        with tarfile.open(self.archive, "w:gz") as handle:
            for name, kind in entries:
                member = tarfile.TarInfo(name)
                if kind == "file":
                    payload = b"synthetic only"
                    member.size = len(payload)
                    handle.addfile(member, io.BytesIO(payload))
                else:
                    member.type = {"dir": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE,
                                   "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                    member.linkname = "../../escape"
                    handle.addfile(member)
        self.checksum.write_text(sha256(self.archive) + "  " + self.archive.name + "\n", encoding="ascii")

    def test_full_import_preserves_raw_bytes_hashes_and_original_archive(self):
        archive_bytes = self.archive.read_bytes()
        checksum_bytes = self.checksum.read_bytes()
        self.destination.mkdir()
        (self.destination / ".gitkeep").write_bytes(b"")
        report = self.run_import(expected_archive_sha256=sha256(self.archive),
                                 expected_selection_sha256=sha256(self.source / "selection.json"),
                                 expected_counts={"D": 2}, expected_size=[2, 2])
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["counts"], {"D": {"T1": 2, "T2": 2, "GT": 2}})
        self.assertEqual(report["total_images"], 6)
        self.assertEqual(report["gt_pixel_values"], {"D": [0, 255]})
        self.assertEqual(report["dimensions"], {"D": [[2, 2]]})
        for path in self.source.rglob("*"):
            if path.is_file():
                self.assertEqual((self.destination / path.relative_to(self.source)).read_bytes(), path.read_bytes())
        self.assertEqual(self.archive.read_bytes(), archive_bytes)
        self.assertEqual(self.checksum.read_bytes(), checksum_bytes)
        self.assertEqual(list(self.base.glob(".fixed-import-*")), [])
        self.assertTrue((self.destination / ".gitkeep").exists())

    def test_complete_data_is_reused_without_rewriting(self):
        first = self.run_import()
        before = {path: path.stat().st_mtime_ns for path in self.destination.rglob("*") if path.is_file()}
        second = self.run_import()
        self.assertGreater(first["copied_files"], 0)
        self.assertEqual(second["copied_files"], 0)
        self.assertEqual(second["reused_files"], len(before))
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})

    def test_partial_identical_data_merges_safely(self):
        path = self.destination / "D/GT/sub/0001.png"
        path.parent.mkdir(parents=True)
        path.write_bytes((self.source / "D/GT/sub/0001.png").read_bytes())
        old_mtime = path.stat().st_mtime_ns
        report = self.run_import()
        self.assertEqual(report["reused_files"], 1)
        self.assertEqual(path.stat().st_mtime_ns, old_mtime)

    def test_inconsistent_existing_file_prevents_all_copies(self):
        path = self.destination / "D/GT/sub/0001.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"existing personal data")
        with self.assertRaisesRegex(CDMapError, "refusing to overwrite"):
            self.run_import()
        self.assertEqual(path.read_bytes(), b"existing personal data")
        self.assertFalse((self.destination / "selection.json").exists())
        self.assertEqual(list(self.base.glob(".fixed-import-*")), [])

    def test_existing_extra_image_not_deleted_or_silently_accepted(self):
        path = self.destination / "extra.png"
        path.parent.mkdir()
        path.write_bytes(b"existing image")
        with self.assertRaisesRegex(CDMapError, "Existing extra image"):
            self.run_import()
        self.assertEqual(path.read_bytes(), b"existing image")

    def test_checksum_and_expected_hash_fail_before_destination_writes(self):
        for options in ({"expected_archive_sha256": "0" * 64}, {"expected_selection_sha256": "1" * 64},
                        {"expected_counts": {"D": 3}}, {"expected_size": [3, 2]}):
            with self.subTest(options=options), self.assertRaises(CDMapError):
                self.run_import(**options)
            self.assertFalse(self.destination.exists())
        self.archive.write_bytes(self.archive.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(CDMapError, "Archive SHA-256 mismatch"):
            self.run_import()

    def test_checksum_record_filename_must_match(self):
        self.checksum.write_text(sha256(self.archive) + "  wrong.tar.gz\n", encoding="ascii")
        with self.assertRaisesRegex(CDMapError, "filename mismatch"):
            self.run_import()

    def test_archive_traversal_links_absolute_collision_and_specials(self):
        cases = [[("outer/../../escape", "file")], [("/absolute/file", "file")],
                 [("outer/bad\\path", "file")], [("outer/link", "symlink")], [("outer/link", "hardlink")],
                 [("outer/fifo", "fifo")], [("outer/a", "file"), ("outer/a", "file")],
                 [("outer/A.png", "file"), ("outer/a.png", "file")],
                 [("outer/A/x", "file"), ("outer/a/y", "file")],
                 [("outer/a", "file"), ("outer/a/b", "file")],
                 [("first/file", "file"), ("second/file", "file")], [("top-level-file", "file")]]
        for entries in cases:
            self.malformed_archive(entries)
            with self.subTest(entries=entries), self.assertRaises(CDMapError):
                self.run_import()
            self.assertFalse(self.destination.exists())
            self.assertEqual(list(self.base.glob(".fixed-import-*")), [])

    def test_missing_extra_and_image_hash_mismatch(self):
        path = self.source / "D/GT/Abc.png"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(CDMapError, "Missing"):
            verify_fixed_samples(self.source)
        path.write_bytes(original)
        extra = self.source / "D/GT/extra.png"
        extra.write_bytes(original)
        with self.assertRaisesRegex(CDMapError, "extra="):
            verify_fixed_samples(self.source)
        extra.unlink()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(CDMapError, "Image SHA-256 mismatch"):
            verify_fixed_samples(self.source)

    def test_image_decode_mode_and_dimensions_against_frozen_metadata(self):
        record = self.selection["datasets"]["D"]["selection"][0]
        path = self.source / record["files"]["GT"]["target"]
        path.write_bytes(b"not a PNG image")
        item = record["files"]["GT"]
        item.update(bytes=path.stat().st_size, sha256=sha256(path))
        save_metadata(self.source, self.selection)
        with self.assertRaisesRegex(CDMapError, "Cannot fully decode"):
            verify_fixed_samples(self.source)
        Image.fromarray(np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)).save(path)
        item.update(bytes=path.stat().st_size, sha256=sha256(path))
        save_metadata(self.source, self.selection)
        with self.assertRaisesRegex(CDMapError, "dimensions differ"):
            verify_fixed_samples(self.source)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
        item.update(bytes=path.stat().st_size, sha256=sha256(path))
        save_metadata(self.source, self.selection)
        with self.assertRaisesRegex(CDMapError, "mode differs"):
            verify_fixed_samples(self.source)

    def test_selection_and_csv_raw_hashes_and_semantics(self):
        config = {"samples_root": self.source, "selection_file": self.source / "selection.json", "samples_manifest": None}
        records, hashes = load_samples(config)
        self.assertEqual(records[0]["gt_relative_path"], "D/GT/sub/0001.png")
        self.assertEqual(records[0]["provenance"]["gt_source"], "/historical/server/D/GT/sub/0001.png")
        self.assertEqual(hashes[str(self.source / "selection.json")], sha256(self.source / "selection.json"))
        config["expected_selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch"):
            load_samples(config)
        del config["expected_selection_sha256"]
        csv_path = self.source / "D/samples.csv"
        original = csv_path.read_bytes()
        csv_path.write_bytes(original.replace(b"sub/0001.png", b"sub/9999.png"))
        with self.assertRaisesRegex(CDMapError, "SHA-256 mismatch"):
            load_samples(config)
        self.selection["datasets"]["D"]["samples_csv_sha256"] = sha256(csv_path)
        (self.source / "selection.json").write_text(json.dumps(self.selection), encoding="utf-8")
        with self.assertRaisesRegex(CDMapError, "CSV ID/rank mismatch"):
            load_samples(config)

    def test_duplicate_and_changed_frozen_ids_rejected(self):
        data = self.selection["datasets"]["D"]
        data["selection"][1]["sample_id"] = data["selection"][0]["sample_id"]
        save_metadata(self.source, self.selection)
        with self.assertRaisesRegex(CDMapError, "Duplicate sample"):
            verify_fixed_samples(self.source)

    def test_completion_counts_hashes_and_flags_checked(self):
        path = self.source / "EXPORT_COMPLETE.json"
        original = json.loads(path.read_text())
        for field, value in (("status", "incomplete"), ("selection_sha256", "0" * 64),
                             ("counts", {}), ("total_images", 1), ("all_images_fully_decoded", False)):
            path.write_text(json.dumps(dict(original, **{field: value})), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(CDMapError):
                verify_fixed_samples(self.source)


if __name__ == "__main__":
    unittest.main()
