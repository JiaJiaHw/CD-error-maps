"""Create an independent synthetic workspace; no real experiment masks are used."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / "examples" / "synthetic_workspace"
MODELS = ("Synthetic-0255", "Synthetic-01")

# All answers below are written by hand. The tool's classifier is not imported.
CASES = (
    ("SYNTH-A", "0001.png", [[0, 0], [1, 1]],
     [[0, 1], [1, 0]], [[0, 1], [1, 0]],
     {"TN": 1, "TP": 1, "FP": 1, "FN": 1},
     {"TN": 1, "TP": 1, "FP": 1, "FN": 1}),
    ("SYNTH-A", "Subset/Case_02.png", [[0, 0], [0, 0]],
     [[0, 0], [0, 0]], [[1, 1], [1, 1]],
     {"TN": 4, "TP": 0, "FP": 0, "FN": 0},
     {"TN": 0, "TP": 0, "FP": 4, "FN": 0}),
    ("SYNTH-B", "0001.png", [[1, 1], [1, 1]],
     [[1, 1], [1, 1]], [[0, 0], [0, 0]],
     {"TN": 0, "TP": 4, "FP": 0, "FN": 0},
     {"TN": 0, "TP": 0, "FP": 0, "FN": 4}),
    ("SYNTH-B", "Subset/Case_02.png", [[1, 0], [0, 1]],
     [[1, 0], [1, 0]], [[1, 0], [1, 0]],
     {"TN": 1, "TP": 1, "FP": 1, "FN": 1},
     {"TN": 1, "TP": 1, "FP": 1, "FN": 1}),
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mask(path, values, foreground):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(values, dtype=np.uint8) * foreground
    Image.fromarray(pixels).save(path, format="PNG")


def create_workspace():
    if WORKSPACE.exists():
        raise ValueError("Synthetic workspace already exists: {}. Reuse it or move it before rerunning.".format(WORKSPACE))
    stage = Path(tempfile.mkdtemp(prefix=".synthetic-", dir=str(PROJECT_ROOT / "examples")))
    try:
        samples = []
        predictions = []
        answers = []
        ranks = {}
        for dataset, sample_id, gt, pred255, pred01, answer255, answer01 in CASES:
            ranks[dataset] = ranks.get(dataset, 0) + 1
            gt_relative = "{}/GT/{}".format(dataset, sample_id)
            gt_path = stage / "samples" / gt_relative
            write_mask(gt_path, gt, 255)
            samples.append({"dataset": dataset, "sample_id": sample_id,
                            "gt_relative_path": gt_relative, "rank": ranks[dataset],
                            "gt_sha256": file_hash(gt_path), "width": 2, "height": 2})
            for model, prediction, foreground, expected in (
                    (MODELS[0], pred255, 255, answer255),
                    (MODELS[1], pred01, 1, answer01)):
                relative = "{}/{}/{}".format(dataset, model, sample_id)
                prediction_path = stage / "predictions" / relative
                write_mask(prediction_path, prediction, foreground)
                predictions.append({"dataset": dataset, "model": model,
                                    "sample_id": sample_id, "prediction_relative_path": relative,
                                    "prediction_sha256": file_hash(prediction_path),
                                    "acquisition_method": "SYNTHETIC_HAND_WRITTEN_ARRAY"})
                answers.append(dict({"dataset": dataset, "model": model, "sample_id": sample_id,
                                     "width": 2, "height": 2, "valid": 4, "ignored": 0}, **expected))
        write_csv(stage / "samples.csv", ("dataset", "sample_id", "gt_relative_path", "rank",
                                         "gt_sha256", "width", "height"), samples)
        write_csv(stage / "predictions.csv", ("dataset", "model", "sample_id", "prediction_relative_path",
                                             "prediction_sha256", "acquisition_method"), predictions)
        root_relative = WORKSPACE.relative_to(PROJECT_ROOT).as_posix()
        config = {
            "schema_version": 1,
            "samples_root": root_relative + "/samples",
            "selection_file": None,
            "samples_manifest": root_relative + "/samples.csv",
            "prediction_root": root_relative + "/predictions",
            "prediction_manifest": None,
            "prediction_path_template": "{dataset}/{model}/{sample_id}",
            "output_root": root_relative + "/outputs",
            "datasets": ["SYNTH-A", "SYNTH-B"],
            "models": list(MODELS),
            "gt_encoding": "binary_0255",
            "prediction_encoding": "binary_0255",
            "ignore": None,
            "encodings": {dataset: {MODELS[1]: {"gt_encoding": "binary_0255",
                                               "prediction_encoding": "binary_01"}}
                          for dataset in ("SYNTH-A", "SYNTH-B")},
            "palette": {"TN": [0, 0, 0], "TP": [255, 255, 255],
                        "FP": [230, 159, 0], "FN": [0, 114, 178], "IGNORE": [128, 128, 128]},
        }
        write_json(stage / "config.json", config)
        explicit = dict(config)
        explicit["prediction_manifest"] = root_relative + "/predictions.csv"
        explicit["prediction_path_template"] = None
        write_json(stage / "config.explicit.json", explicit)
        write_json(stage / "expected_counts.json", {"kind": "SYNTHETIC_HAND_CALCULATED", "pairs": answers})
        write_json(stage / "SOURCE.json", {"kind": "SYNTHETIC_ONLY", "real_model_predictions": False,
                                          "description": "Hand-written 2x2 arrays; unrelated to frozen experiment data."})
        (stage / "outputs").mkdir()
        stage.rename(WORKSPACE)
        return WORKSPACE
    finally:
        # Only the temporary directory created by this call is eligible for cleanup.
        if stage.exists():
            if stage.resolve().parent != (PROJECT_ROOT / "examples").resolve() or not stage.name.startswith(".synthetic-"):
                raise ValueError("Refusing to remove an unexpected staging directory: {}".format(stage))
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        workspace = create_workspace()
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    print("SYNTHETIC ONLY: {}".format(workspace))
    print("4 samples, 2 synthetic models, 8 pairs. No real Prediction files were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
