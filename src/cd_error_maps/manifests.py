"""Manifest-driven samples and exact Prediction pairing (never infer IDs)."""

import csv
import json
import re
import string
from pathlib import Path

from .common import CDMapError, check_unique_paths, output_id, safe_join, sha256, validate_relative


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def _require(condition, message):
    if not condition:
        raise CDMapError(message)


def _name(value, label):
    _require(isinstance(value, str) and bool(value), "Missing " + label)
    validate_relative(value)
    _require("/" not in value, label + " must be one directory component: " + value)
    return value


def _digest(value, label, optional=True):
    if value in (None, "") and optional:
        return None
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
             "Invalid SHA-256 for " + label)
    return value.lower()


def _integer(value, label, optional=False):
    if value in (None, "") and optional:
        return None
    _require(not isinstance(value, bool), "Invalid integer for " + label)
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise CDMapError("Invalid integer for " + label)
    _require(str(result) == str(value) and result > 0, "Expected positive integer for " + label)
    return result


def _csv_rows(path, required):
    path = Path(path)
    _require(path.is_file(), "Missing manifest: " + str(path))
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            _require(len(fields) == len(set(fields)), "Duplicate CSV columns: " + str(path))
            _require(set(required) <= set(fields), "Missing CSV columns " + str(sorted(set(required) - set(fields))) + ": " + str(path))
            rows = []
            for line, row in enumerate(reader, 2):
                _require(None not in row and all(value is not None for value in row.values()),
                         "Malformed CSV row " + str(line) + ": " + str(path))
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CDMapError("Cannot read CSV " + str(path) + ": " + str(exc))


def _check_hash(path, expected, label):
    _require(Path(path).is_file(), "Missing " + label + ": " + str(path))
    digest = sha256(path)
    expected = _digest(expected, label)
    _require(expected is None or digest == expected, "SHA-256 mismatch for " + label + ": " + str(path))
    return digest


def _check_count(dataset, count, expected_counts):
    if expected_counts and dataset in expected_counts:
        expected = expected_counts[dataset]
        if isinstance(expected, dict):
            expected = expected.get("GT")
        _require(count == _integer(expected, "expected_counts." + dataset),
                 "Sample count mismatch for " + dataset + ": " + str(count) + " (expected " + str(expected) + ")")


def load_selection_records(root, selection_file=None, expected_sha=None, expected_counts=None):
    """Read frozen JSON/CSV metadata unchanged; source paths are provenance only."""
    root = Path(root)
    selection_path = Path(selection_file) if selection_file else root / "selection.json"
    hashes = {str(selection_path): _check_hash(selection_path, expected_sha, "selection.json")}
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise CDMapError("Cannot read selection.json: " + str(exc))
    _require(isinstance(selection, dict) and selection.get("schema_version") == 1,
             "Unsupported selection schema_version")
    datasets = selection.get("datasets")
    _require(isinstance(datasets, dict) and bool(datasets), "selection datasets must be a nonempty object")
    records = []
    all_targets = []
    seen = set()
    for dataset, data in datasets.items():
        _name(dataset, "dataset")
        _require(isinstance(data, dict) and isinstance(data.get("selection"), list) and bool(data["selection"]),
                 "Missing selection records for " + dataset)
        selected = data["selection"]
        _check_count(dataset, len(selected), expected_counts)
        if "sample_count_per_dataset" in selection:
            _require(len(selected) == _integer(selection["sample_count_per_dataset"], "sample_count_per_dataset"),
                     "Frozen selection count inconsistent for " + dataset)
        csv_path = safe_join(root, dataset + "/samples.csv")
        hashes[str(csv_path)] = _check_hash(csv_path, data.get("samples_csv_sha256"), "samples.csv")
        _digest(data.get("samples_csv_sha256"), dataset + " samples.csv", optional=False)
        required = ["dataset", "rank", "sample_id", "width", "height"]
        for role in ("T1", "T2", "GT"):
            required.extend(role + "_" + field for field in ("source", "target", "bytes", "sha256", "mode"))
        rows = _csv_rows(csv_path, required)
        _require(len(rows) == len(selected), "CSV ID count differs from selection for " + dataset)
        for position, (record, row) in enumerate(zip(selected, rows), 1):
            _require(isinstance(record, dict), "Invalid selection record for " + dataset)
            sample_id = record.get("sample_id")
            _require(isinstance(sample_id, str) and bool(sample_id), "Missing sample_id for " + dataset)
            validate_relative(sample_id)
            key = (dataset, sample_id)
            _require(key not in seen, "Duplicate sample ID: " + str(key))
            seen.add(key)
            rank = _integer(record.get("rank"), "rank " + str(key))
            _require(rank == position, "Selection ranks must be consecutive in manifest order: " + str(key))
            _require(row["dataset"] == dataset and row["sample_id"] == sample_id and row["rank"] == str(rank),
                     "CSV ID/rank mismatch with selection: " + str(key))
            files = record.get("files")
            _require(isinstance(files, dict) and set(files) == {"T1", "T2", "GT"}, "Invalid triplet roles: " + str(key))
            sizes = set()
            for role in ("T1", "T2", "GT"):
                item = files[role]
                _require(isinstance(item, dict), "Invalid file metadata: " + str(key) + " " + role)
                target = item.get("target")
                _require(target == dataset + "/" + role + "/" + sample_id,
                         "Target path must preserve dataset/role/sample ID: " + str(key) + " " + role)
                all_targets.append(safe_join(root, target))
                _digest(item.get("sha256"), target, optional=False)
                byte_count = item.get("bytes")
                _require(isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count >= 0,
                         "Invalid byte count: " + target)
                width = _integer(item.get("width"), "width " + target)
                height = _integer(item.get("height"), "height " + target)
                _require(isinstance(item.get("mode"), str) and bool(item["mode"]), "Missing image mode: " + target)
                _require(isinstance(item.get("source"), str), "Missing source provenance: " + target)
                sizes.add((width, height))
                for field in ("source", "target", "bytes", "sha256", "mode"):
                    _require(row[role + "_" + field] == str(item[field]),
                             "CSV metadata mismatch: " + target + " " + field)
            _require(len(sizes) == 1, "Frozen triplet dimensions mismatch: " + str(key))
            width, height = next(iter(sizes))
            _require(row["width"] == str(width) and row["height"] == str(height), "CSV dimensions mismatch: " + str(key))
            for field in ("source_list_entry", "sort_bytes"):
                if field in row and field in record:
                    _require(row[field] == str(record[field]), "CSV metadata mismatch: " + str(key) + " " + field)
            gt = files["GT"]
            records.append({"dataset": dataset, "sample_id": sample_id, "rank": rank,
                            "gt_relative_path": gt["target"], "gt_sha256": gt["sha256"].lower(),
                            "width": width, "height": height,
                            "provenance": {"gt_source": gt["source"], "source_list_entry": record.get("source_list_entry", "")},
                            "files": files})
        if "test_txt_sha256" in data:
            # Upstream may supply a reusable frozen inference list. Never
            # reconstruct its order or normalize the original sample IDs.
            test_path = safe_join(root, dataset + "/test.txt")
            expected_test_hash = _digest(data["test_txt_sha256"], dataset + " test.txt", optional=False)
            hashes[str(test_path)] = _check_hash(test_path, expected_test_hash, "test.txt")
            try:
                test_ids = test_path.read_bytes().decode("utf-8").splitlines()
            except (OSError, UnicodeError) as exc:
                raise CDMapError("Cannot read test.txt for " + dataset + ": " + str(exc))
            _require(test_ids == [record["sample_id"] for record in selected],
                     "test.txt ID/order mismatch with selection for " + dataset)
    check_unique_paths(all_targets, "Frozen input")
    if expected_counts:
        _require(set(expected_counts) <= set(datasets), "Expected datasets missing from selection")
    for path, digest in hashes.items():
        _require(sha256(path) == digest, "Manifest changed while reading: " + path)
    return records, hashes, selection


def load_samples(config):
    """Return normalized rows in unchanged manifest order and raw manifest hashes."""
    root = Path(config["samples_root"])
    selection_file = config.get("selection_file")
    manifest = config.get("samples_manifest")
    _require(not (selection_file and manifest), "selection_file and samples_manifest are mutually exclusive")
    if selection_file:
        records, hashes, _ = load_selection_records(root, selection_file, config.get("expected_selection_sha256"), config.get("expected_counts"))
    else:
        _require(manifest is not None, "Set selection_file or samples_manifest")
        manifest = Path(manifest)
        hashes = {str(manifest): _check_hash(manifest, config.get("expected_manifest_sha256"), "samples manifest")}
        rows = _csv_rows(manifest, ["dataset", "sample_id", "gt_relative_path"])
        records = []
        seen = set()
        for row in rows:
            dataset = _name(row["dataset"], "dataset")
            sample_id = validate_relative(row["sample_id"])
            key = (dataset, sample_id)
            _require(key not in seen, "Duplicate sample ID: " + str(key))
            seen.add(key)
            relative = validate_relative(row["gt_relative_path"])
            safe_join(root, relative)
            width = _integer(row.get("width"), "width " + str(key), optional=True)
            height = _integer(row.get("height"), "height " + str(key), optional=True)
            _require((width is None) == (height is None), "Declare width and height together: " + str(key))
            records.append({"dataset": dataset, "sample_id": sample_id,
                            "rank": _integer(row.get("rank"), "rank " + str(key), optional=True),
                            "gt_relative_path": relative, "gt_sha256": _digest(row.get("gt_sha256"), str(key)),
                            "width": width, "height": height})
        _require(bool(records), "Samples manifest has no rows")
        for dataset in {record["dataset"] for record in records}:
            _check_count(dataset, sum(record["dataset"] == dataset for record in records), config.get("expected_counts"))
        if config.get("expected_counts"):
            _require(set(config["expected_counts"]) <= {record["dataset"] for record in records}, "Expected datasets missing from samples manifest")
        check_unique_paths([safe_join(root, record["gt_relative_path"]) for record in records], "GT input")
    check_unique_paths([safe_join(root, record["dataset"] + "/" + output_id(record["sample_id"])) for record in records], "Sample output")
    for path, digest in hashes.items():
        _require(sha256(path) == digest, "Manifest changed while reading: " + path)
    return records, hashes


def _template_path(template, dataset, model, sample_id):
    _require(isinstance(template, str) and bool(template), "Missing prediction_path_template")
    try:
        for _, field, spec, conversion in string.Formatter().parse(template):
            if field is not None:
                _require(field in {"dataset", "model", "sample_id"} and not spec and not conversion,
                         "Prediction template accepts only {dataset}, {model}, {sample_id}")
        value = template.format(dataset=dataset, model=model, sample_id=sample_id)
    except (ValueError, KeyError, IndexError) as exc:
        raise CDMapError("Invalid prediction_path_template: " + str(exc))
    return validate_relative(value)


def pair_samples(config, samples, combinations):
    """Match every selected key exactly, rejecting missing/extra/ambiguous paths."""
    combinations = list(combinations)
    _require(bool(combinations) and len(combinations) == len(set(combinations)), "Select distinct dataset/model combinations")
    scope = set(combinations)
    dataset_ids = {}
    for sample in samples:
        dataset_ids.setdefault(sample["dataset"], set()).add(sample["sample_id"])
    for dataset, model in combinations:
        _name(dataset, "dataset")
        _name(model, "model")
        _require(dataset in dataset_ids, "Unknown dataset: " + dataset)
    manifest = config.get("prediction_manifest")
    template = config.get("prediction_path_template")
    _require(not (manifest and template), "prediction_manifest and prediction_path_template are mutually exclusive")
    mapping = {}
    if manifest:
        manifest_digest = _check_hash(manifest, config.get("expected_prediction_manifest_sha256"), "Prediction manifest")
        rows = _csv_rows(manifest, ["dataset", "model", "sample_id", "prediction_relative_path"])
        for row in rows:
            if (row["dataset"], row["model"]) not in scope:
                continue
            key = (row["dataset"], row["model"], row["sample_id"])
            validate_relative(row["sample_id"])
            _require(row["sample_id"] in dataset_ids[row["dataset"]], "Unknown Prediction sample ID: " + str(key))
            _require(key not in mapping, "Duplicate Prediction ID: " + str(key))
            relative = validate_relative(row["prediction_relative_path"])
            mapping[key] = {"prediction_relative_path": relative,
                            "prediction_sha256": _digest(row.get("prediction_sha256"), str(key)),
                            "provenance": {k: v for k, v in row.items() if k not in {"dataset", "model", "sample_id", "prediction_relative_path", "prediction_sha256"} and v}}
    else:
        _require(template is not None, "Set Prediction manifest or explicit same-name template")
    root = Path(config["prediction_root"])
    result = []
    paths = []
    for dataset, model in combinations:
        group_paths = set()
        for sample in samples:
            if sample["dataset"] != dataset:
                continue
            key = (dataset, model, sample["sample_id"])
            if manifest:
                _require(key in mapping, "Prediction not ready: missing mapping for " + str(key))
                prediction = mapping[key]
            else:
                prediction = {"prediction_relative_path": _template_path(template, *key), "prediction_sha256": None, "provenance": {}}
            path = safe_join(root, prediction["prediction_relative_path"])
            _require(path.is_file(), "Prediction not ready: missing " + str(key) + " at " + str(path))
            paths.append(path)
            group_paths.add(path.resolve())
            pair = dict(sample)
            pair.update({"model": model, "prediction_relative_path": prediction["prediction_relative_path"],
                         "prediction_sha256": prediction["prediction_sha256"],
                         "provenance": dict(sample.get("provenance", {}), **prediction["provenance"])})
            result.append(pair)
        group_root = safe_join(root, dataset + "/" + model)
        if group_root.exists():
            _require(group_root.is_dir(), "Prediction group is not a directory: " + str(group_root))
            extras = []
            for candidate in group_root.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    checked = safe_join(root, candidate.relative_to(root).as_posix())
                    if checked.resolve() not in group_paths:
                        extras.append(candidate.relative_to(root).as_posix())
            _require(not extras, "Extra Prediction images in " + dataset + "/" + model + ": " + ", ".join(sorted(extras)))
    check_unique_paths(paths, "Prediction input")
    gt_paths = {str(safe_join(config["samples_root"], sample["gt_relative_path"])).casefold() for sample in samples}
    _require(not ({str(path.resolve()).casefold() for path in paths} & gt_paths), "Prediction path conflicts with GT input")
    check_unique_paths([safe_join(root, pair["dataset"] + "/" + pair["model"] + "/" + output_id(pair["sample_id"])) for pair in result], "Error map output")
    if manifest:
        _require(sha256(manifest) == manifest_digest, "Prediction manifest changed while reading: " + str(manifest))
    return result
