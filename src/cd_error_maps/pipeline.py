"""Read-only validation, isolated runs, and independent output read-back."""
import csv
import io
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from . import __version__
from .common import CDMapError, check_unique_paths, output_id, safe_join, sha256
from .config import PROJECT_ROOT, effective_config, encoding_for, load_config
from .core import CLASS_INDEX, classify, counts, metrics, render, validate_palette, write_legend
from .manifests import load_samples, pair_samples
from .masks import read_mask

COUNT_FIELDS = ["TN", "TP", "FP", "FN", "ignored", "valid", "width", "height"]
PAIR_FIELDS = ["dataset", "model", "sample_id", "rank", "gt_path", "prediction_path", "gt_sha256",
               "prediction_sha256", "gt_encoding", "prediction_encoding", "error_rgb", "error_class",
               "error_rgb_sha256", "error_class_sha256", "provenance"]
PIXEL_FIELDS = ["dataset", "model", "sample_id"] + COUNT_FIELDS
SUMMARY_FIELDS = ["dataset", "model", "images", "TN", "TP", "FP", "FN", "ignored", "valid",
                  "precision", "recall", "f1", "iou", "accuracy", "undefined_metrics"]


def _require(condition, message):
    if not condition:
        raise CDMapError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(path, value, exclusive=False):
    # Serialization must finish before an existing status file is opened.
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CDMapError("Invalid JSON {}: {}".format(path, exc))


def _csv_text(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        if isinstance(normalized.get("provenance"), dict):
            normalized["provenance"] = json.dumps(normalized["provenance"], ensure_ascii=False, sort_keys=True)
        writer.writerow(normalized)
    return stream.getvalue()


def _write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(_csv_text(fields, rows))


def choose_combinations(config, datasets=None, models=None, explicit=None):
    if explicit:
        _require(not datasets and not models, "--pair cannot be combined with --dataset/--model")
        result = []
        for value in explicit:
            parts = value.split(":")
            _require(len(parts) == 2, "--pair must be DATASET:MODEL")
            result.append(tuple(parts))
    elif datasets or models:
        result = [(d, m) for d in (datasets or config["datasets"]) for m in (models or config["models"])]
    elif config.get("combinations"):
        result = [(row["dataset"], row["model"]) for row in config["combinations"]]
    else:
        result = [(d, m) for d in config["datasets"] for m in config["models"]]
    _require(result and len(result) == len(set(result)), "Select nonduplicate dataset/model combinations")
    for d, m in result:
        _require(d in config["datasets"] and m in config["models"], "Unknown configured combination: {}:{}".format(d, m))
    return result


def _ignore_value(config):
    return config["ignore"]["gt_value"] if config.get("ignore") else None


def _read_pair(config, pair, expected=None):
    key = (pair["dataset"], pair["model"], pair["sample_id"])
    gt_path = safe_join(config["samples_root"], pair["gt_relative_path"])
    pred_path = safe_join(config["prediction_root"], pair["prediction_relative_path"])
    gt_hash, pred_hash = sha256(gt_path), sha256(pred_path)
    if expected:
        _require(gt_hash == expected["gt_sha256"] and pred_hash == expected["prediction_sha256"], "Input changed after validation: {}".format(key))
    for label, actual, wanted in (("GT", gt_hash, pair.get("gt_sha256")),
                                   ("Prediction", pred_hash, pair.get("prediction_sha256"))):
        _require(not wanted or actual == wanted, "{} hash mismatch {} at {}".format(label, key, gt_path if label == "GT" else pred_path))
    genc, penc = encoding_for(config, pair["dataset"], pair["model"])
    gt, valid = read_mask(gt_path, genc, _ignore_value(config))
    pred, _ = read_mask(pred_path, penc)
    _require(gt.shape == pred.shape, "GT/Prediction size mismatch {}: {} vs {}".format(key, gt.shape, pred.shape))
    size = [gt.shape[1], gt.shape[0]]
    _require(not config.get("expected_size") or size == config["expected_size"], "Configured size mismatch {}: {}".format(key, size))
    _require(not pair.get("width") or size == [pair["width"], pair["height"]], "Manifest size mismatch {}: {}".format(key, size))
    _require(sha256(gt_path) == gt_hash and sha256(pred_path) == pred_hash, "Input changed while decoding: {}".format(key))
    return gt, pred, valid, {"gt_path": str(gt_path), "prediction_path": str(pred_path),
                            "gt_sha256": gt_hash, "prediction_sha256": pred_hash,
                            "gt_encoding": genc, "prediction_encoding": penc,
                            "width": size[0], "height": size[1]}


def _snapshot_unchanged(report):
    for path, digest in report["manifest_hashes"].items():
        _require(sha256(path) == digest, "Manifest changed after validation: " + path)
    for pair in report["pairs"]:
        _require(sha256(pair["gt_path"]) == pair["gt_sha256"], "GT changed after validation: " + pair["gt_path"])
        if "prediction_path" in pair:
            _require(sha256(pair["prediction_path"]) == pair["prediction_sha256"], "Prediction changed after validation: " + pair["prediction_path"])


def validate_config_inputs(config, combinations, samples_only=False):
    palette = validate_palette(config["palette"])
    samples, hashes = load_samples(config)
    if config.get("prediction_manifest") and not samples_only:
        hashes[str(config["prediction_manifest"])] = sha256(config["prediction_manifest"])
    datasets = {d for d, _ in combinations}
    _require(datasets <= {row["dataset"] for row in samples}, "Selected dataset missing from samples manifest")
    normalized = []
    if samples_only:
        for sample in samples:
            if sample["dataset"] not in datasets:
                continue
            path = safe_join(config["samples_root"], sample["gt_relative_path"])
            digest = sha256(path)
            _require(not sample.get("gt_sha256") or digest == sample["gt_sha256"], "GT hash mismatch: " + str(path))
            encoding_list = sorted({encoding_for(config, d, m)[0] for d, m in combinations if d == sample["dataset"]})
            for encoding in encoding_list:
                gt, _ = read_mask(path, encoding, _ignore_value(config))
                size = [gt.shape[1], gt.shape[0]]
                _require(not sample.get("width") or size == [sample["width"], sample["height"]], "GT manifest size mismatch: " + str(path))
                _require(not config.get("expected_size") or size == config["expected_size"], "GT configured size mismatch: " + str(path))
            normalized.append(dict(sample, gt_path=str(path), gt_sha256=digest, width=size[0], height=size[1]))
    else:
        for pair in pair_samples(config, samples, combinations):
            _, _, _, info = _read_pair(config, pair)
            normalized.append(dict(pair, **info))
    report = {"schema_version": 1, "status": "validated", "samples_only": samples_only,
              "prediction_checked": not samples_only, "scope": [list(x) for x in combinations],
              "pair_count": len(normalized), "manifest_hashes": hashes, "palette": palette,
              "pairs": normalized, "validated_at_utc": _now()}
    _snapshot_unchanged(report)
    return report


def _summary(pairs):
    groups = {}
    for pair in pairs:
        key = (pair["dataset"], pair["model"])
        entry = groups.setdefault(key, {"dataset": key[0], "model": key[1], "images": 0,
                                      **{x: 0 for x in COUNT_FIELDS if x not in ("width", "height")}})
        entry["images"] += 1
        for name in ("TN", "TP", "FP", "FN", "ignored", "valid"):
            entry[name] += pair["counts"][name]
    for row in groups.values():
        row.update(metrics(row))
        row["undefined_metrics"] = ",".join(key for key in ("precision", "recall", "f1", "iou", "accuracy") if row[key] is None)
    return list(groups.values())


def _check_run_location(run, config):
    def within(path, root):
        parts = tuple(part.casefold() for part in Path(path).resolve().parts)
        root_parts = tuple(part.casefold() for part in Path(root).resolve().parts)
        return parts[:len(root_parts)] == root_parts

    for root in (config["samples_root"], config["prediction_root"]):
        _require(not (within(run, root) or within(root, run)), "Run path overlaps input root: " + str(root))
    for key in ("selection_file", "samples_manifest", "prediction_manifest"):
        path = config.get(key)
        _require(path is None or not within(path, run), "Run path overlaps input manifest")


def _version_record():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"tool": __version__, "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(),
            "numpy": np.__version__, "pillow": pillow_version, "platform": platform.platform(), "command": sys.argv}


def generate(config, combinations, limit=None, out=None):
    _require(limit is None or (type(limit) is int and limit > 0), "--limit must be a positive integer per combination")
    validation = validate_config_inputs(config, combinations)
    if out is None:
        run = config["output_root"] / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    else:
        path = Path(out)
        run = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    run = run.resolve()
    _check_run_location(run, config)
    _require(not run.exists(), "Run directory already exists; refusing overwrite: " + str(run))
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "started_at_utc": _now(),
                "preview": limit is not None, "limit_per_combination": limit,
                "scope": validation["scope"], "validated_pair_count": validation["pair_count"],
                "versions": {}, "pairs": []}
    seal_attempted = False
    try:
        manifest["versions"] = _version_record()
        _json(run / "run_manifest.json", manifest)
        _json(run / "effective_config.json", effective_config(config), True)
        _json(run / "validation_report.json", validation, True)
        _json(run / "palette.json", {"schema_version": 1, "indices": CLASS_INDEX, "colors": validation["palette"]}, True)
        write_legend(run / "legend.png", validation["palette"])
        group_counts = {}
        for pair in validation["pairs"]:
            key = (pair["dataset"], pair["model"])
            ordinal = group_counts.get(key, 0)
            group_counts[key] = ordinal + 1
            if limit is not None and ordinal >= limit:
                continue
            gt, pred, valid, _ = _read_pair(config, pair, expected=pair)
            index = classify(gt, pred, valid)
            relative = pair["dataset"] + "/" + pair["model"] + "/" + output_id(pair["sample_id"])
            rgb_relative, class_relative = "error_rgb/" + relative, "error_class/" + relative
            rgb_path, class_path = safe_join(run, rgb_relative), safe_join(run, class_relative)
            for target in (rgb_path, class_path):
                target.parent.mkdir(parents=True, exist_ok=True)
                _require(not target.exists(), "Output path conflict: " + str(target))
            with class_path.open("xb") as handle:
                Image.fromarray(index).save(handle, format="PNG")
            with rgb_path.open("xb") as handle:
                Image.fromarray(render(index, validation["palette"])).save(handle, format="PNG")
            item = dict(pair, error_rgb=rgb_relative, error_class=class_relative,
                        error_rgb_sha256=sha256(rgb_path), error_class_sha256=sha256(class_path), counts=counts(index))
            manifest["pairs"].append(item)
        _snapshot_unchanged(validation)
        pairs = manifest["pairs"]
        _write_csv(run / "pairs.csv", PAIR_FIELDS, pairs)
        _write_csv(run / "pixel_counts.csv", PIXEL_FIELDS, [dict({k: p[k] for k in ("dataset", "model", "sample_id")}, **p["counts"]) for p in pairs])
        manifest["summary"] = _summary(pairs)
        _write_csv(run / "summary.csv", SUMMARY_FIELDS, manifest["summary"])
        manifest.update(status="complete", generated_pair_count=len(pairs), finished_at_utc=_now(),
                        artifact_hashes={p.relative_to(run).as_posix(): sha256(p) for p in run.rglob("*") if p.is_file() and p.name != "run_manifest.json"})
        _json(run / "run_manifest.json", manifest)
        verify(run, require_complete=False)
        # Last write only after every output has passed full round-trip checks.
        seal_attempted = True
        _json(run / "COMPLETE.json", {"schema_version": 1, "status": "complete", "run_manifest_sha256": sha256(run / "run_manifest.json"),
                                    "generated_pair_count": len(pairs), "preview": manifest["preview"], "verified_at_utc": _now()}, True)
        return {"status": "complete", "run": str(run), "preview": manifest["preview"], "generated_pair_count": len(pairs),
                "validated_pair_count": validation["pair_count"]}
    except BaseException as exc:
        manifest.update(status="incomplete" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", error=str(exc), failed_at_utc=_now())
        # A failed run retains its partial files but never gains a completion seal.
        if seal_attempted:
            try:
                (run / "COMPLETE.json").unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # If the filesystem also blocks cleanup, the status update and
                # completion hash mismatch still prevent verify from succeeding.
                pass
        try:
            _json(run / "run_manifest.json", manifest)
        except (OSError, ValueError, TypeError):
            # Retain the original error when the filesystem cannot save status.
            pass
        raise


def _image_array(path, mode):
    try:
        with Image.open(path) as image:
            image.load()
            _require(image.format == "PNG" and image.mode == mode, "Invalid output PNG mode: " + str(path))
            return np.asarray(image).copy()
    except (OSError, ValueError, SyntaxError) as exc:
        raise CDMapError("Cannot decode output {}: {}".format(path, exc))


def _verify_run(run, require_complete=True):
    run = Path(run).resolve()
    manifest_hash = sha256(run / "run_manifest.json")
    manifest = _read_json(run / "run_manifest.json")
    _require(isinstance(manifest, dict), "Run manifest must be an object")
    _require(manifest.get("schema_version") == 1 and manifest.get("status") == "complete", "Run is incomplete or failed: " + str(run))
    _require(isinstance(manifest.get("pairs"), list) and isinstance(manifest.get("summary"), list), "Missing run pairs or summaries")
    scope = manifest.get("scope")
    _require(isinstance(scope, list) and bool(scope) and all(isinstance(pair, list) and len(pair) == 2 and all(isinstance(value, str) for value in pair) for pair in scope), "Invalid run scope")
    _require(type(manifest.get("generated_pair_count")) is int and type(manifest.get("validated_pair_count")) is int, "Invalid run pair counts")
    if require_complete:
        seal = _read_json(run / "COMPLETE.json")
        seal_hash = sha256(run / "COMPLETE.json")
        _require(isinstance(seal, dict), "Completion seal must be an object")
        _require(seal.get("status") == "complete" and seal.get("schema_version") == 1
                 and seal.get("run_manifest_sha256") == manifest_hash
                 and seal.get("generated_pair_count") == manifest.get("generated_pair_count")
                 and seal.get("preview") == manifest.get("preview"), "Completion seal mismatch")
    hashes = manifest.get("artifact_hashes", {})
    _require(isinstance(hashes, dict) and hashes, "Missing artifact hashes")
    for relative, digest in hashes.items():
        path = safe_join(run, relative)
        _require(path.is_file() and sha256(path) == digest, "Output hash mismatch or missing: " + relative)
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    _require(actual == set(hashes) | {"run_manifest.json"} | ({"COMPLETE.json"} if require_complete else set()), "Unexpected or missing run files")
    config = load_config(run / "effective_config.json")
    palette = validate_palette(config["palette"])
    _require(_read_json(run / "palette.json") == {"schema_version": 1, "indices": CLASS_INDEX, "colors": palette}, "Palette/index schema mismatch")
    validation = _read_json(run / "validation_report.json")
    _require(isinstance(validation, dict), "Validation report must be an object")
    combinations = [tuple(x) for x in manifest["scope"]]
    _require(choose_combinations(config, explicit=[d + ":" + m for d, m in combinations]) == combinations, "Invalid run scope")
    fresh = validate_config_inputs(config, combinations)
    _require(set(validation) == set(fresh) and all(validation.get(key) == value for key, value in fresh.items() if key != "validated_at_utc"), "Input snapshot or mapping changed")
    selected, groups = [], {}
    limit = manifest.get("limit_per_combination")
    _require(type(manifest.get("preview")) is bool and manifest["preview"] == (limit is not None), "Invalid preview declaration")
    _require(limit is None or (type(limit) is int and limit > 0), "Invalid preview limit")
    for pair in fresh["pairs"]:
        key = (pair["dataset"], pair["model"])
        groups[key] = groups.get(key, 0) + 1
        if limit is None or groups[key] <= limit:
            selected.append(pair)
    pairs = manifest["pairs"]
    _require(len(pairs) == len(selected) == manifest["generated_pair_count"] and fresh["pair_count"] == manifest["validated_pair_count"], "Run pair count mismatch")
    expected_artifacts = {"effective_config.json", "validation_report.json", "palette.json", "legend.png", "pairs.csv", "pixel_counts.csv", "summary.csv"}
    all_output_paths = []
    for pair, expected in zip(pairs, selected):
        _require(isinstance(pair, dict), "Run pair must be an object")
        _require({"error_rgb", "error_class", "error_rgb_sha256", "error_class_sha256", "counts"} <= set(pair), "Incomplete output pair metadata")
        _require(all(pair.get(k) == v for k, v in expected.items()), "Pair mapping metadata mismatch")
        relative = pair["dataset"] + "/" + pair["model"] + "/" + output_id(pair["sample_id"])
        _require(pair["error_rgb"] == "error_rgb/" + relative and pair["error_class"] == "error_class/" + relative, "Output mapping mismatch")
        cp, rp = safe_join(run, pair["error_class"]), safe_join(run, pair["error_rgb"])
        all_output_paths.extend([cp, rp])
        expected_artifacts.update([pair["error_class"], pair["error_rgb"]])
        _require(sha256(cp) == pair["error_class_sha256"] and sha256(rp) == pair["error_rgb_sha256"], "Pair output hash mismatch")
        index, rgb = _image_array(cp, "L"), _image_array(rp, "RGB")
        g, p, v, _ = _read_pair(config, pair, expected=expected)
        _require(np.array_equal(index, classify(g, p, v)), "Output class pixels differ from input truth")
        _require(np.array_equal(rgb, render(index, palette)), "Output RGB differs from class indices/palette")
        _require(counts(index) == pair["counts"], "Pixel counts differ from class image")
    check_unique_paths(all_output_paths, "Run output")
    _require(set(hashes) == expected_artifacts, "Artifact list does not match output mapping")
    summaries = _summary(pairs)
    _require(manifest["summary"] == summaries, "Aggregate statistics mismatch")
    csv_expectations = {"pairs.csv": (PAIR_FIELDS, pairs),
                        "pixel_counts.csv": (PIXEL_FIELDS, [dict({k: p[k] for k in ("dataset", "model", "sample_id")}, **p["counts"]) for p in pairs]),
                        "summary.csv": (SUMMARY_FIELDS, summaries)}
    for name, (fields, rows) in csv_expectations.items():
        _require((run / name).read_text(encoding="utf-8") == _csv_text(fields, rows), "CSV contents mismatch: " + name)
    _image_array(run / "legend.png", "RGB")
    # Recheck snapshots at the end too: a long read-back cannot certify files
    # which changed after their first hash or decoding check.
    _snapshot_unchanged(fresh)
    for relative, digest in hashes.items():
        _require(sha256(safe_join(run, relative)) == digest, "Output changed during verification: " + relative)
    _require(sha256(run / "run_manifest.json") == manifest_hash, "Run manifest changed during verification")
    if require_complete:
        _require(sha256(run / "COMPLETE.json") == seal_hash, "Completion seal changed during verification")
    return {"status": "verified", "run": str(run), "generated_pair_count": len(pairs), "preview": manifest["preview"],
            "input_hashes_checked": True, "output_hashes_checked": True, "pixel_statistics_checked": True}


def verify(run, require_complete=True):
    """Reject malformed run metadata with a stable user-facing error."""
    try:
        return _verify_run(run, require_complete)
    except CDMapError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise CDMapError("Invalid run metadata or schema: {}".format(exc)) from exc
