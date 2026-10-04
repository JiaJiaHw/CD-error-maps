"""TP/TN/FP/FN classification, pixel statistics, and lossless colors."""

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

from .common import CDMapError


CLASS_INDEX = {"TN": 0, "TP": 1, "FP": 2, "FN": 3, "IGNORE": 255}
DEFAULT_PALETTE = {
    "TN": [0, 0, 0],
    "TP": [255, 255, 255],
    "FP": [255, 0, 0],
    "FN": [0, 255, 0],
    "IGNORE": [128, 128, 128],
}


def _boolean_image(value: np.ndarray, label: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.ndim != 2
        or not value.size
    ):
        raise CDMapError("{} must be a nonempty 2D boolean NumPy array".format(label))


def _index_image(index: np.ndarray) -> None:
    if (
        not isinstance(index, np.ndarray)
        or index.dtype != np.dtype(np.uint8)
        or index.ndim != 2
        or not index.size
    ):
        raise CDMapError("Class index must be a nonempty 2D uint8 NumPy array")
    unexpected = sorted(set(int(value) for value in np.unique(index)) - set(CLASS_INDEX.values()))
    if unexpected:
        raise CDMapError("Illegal class indices: {}".format(unexpected))


def classify(
    gt: np.ndarray, pred: np.ndarray, valid: Optional[np.ndarray] = None
) -> np.ndarray:
    """TN=0, TP=1, FP=2, FN=3, ignored GT=255; preserve input size."""
    _boolean_image(gt, "GT")
    _boolean_image(pred, "Prediction")
    if gt.shape != pred.shape:
        raise CDMapError("GT/Prediction dimensions differ: {} vs {}".format(gt.shape, pred.shape))
    if valid is None:
        valid = np.ones(gt.shape, dtype=np.bool_)
    else:
        _boolean_image(valid, "GT valid mask")
        if valid.shape != gt.shape:
            raise CDMapError("GT valid mask dimensions differ from GT")

    index = np.full(gt.shape, CLASS_INDEX["IGNORE"], dtype=np.uint8)
    index[valid & ~gt & ~pred] = CLASS_INDEX["TN"]
    index[valid & gt & pred] = CLASS_INDEX["TP"]
    index[valid & ~gt & pred] = CLASS_INDEX["FP"]
    index[valid & gt & ~pred] = CLASS_INDEX["FN"]
    return index


def counts(index: np.ndarray) -> Dict[str, int]:
    """Count pixels and retain width/height for per-image reports."""
    _index_image(index)
    result = {
        name: int(np.count_nonzero(index == CLASS_INDEX[name]))
        for name in ("TN", "TP", "FP", "FN")
    }
    result["ignored"] = int(np.count_nonzero(index == CLASS_INDEX["IGNORE"]))
    result["valid"] = sum(result[name] for name in ("TN", "TP", "FP", "FN"))
    result["width"] = int(index.shape[1])
    result["height"] = int(index.shape[0])
    return result


def metrics(pixel_counts: Mapping) -> Dict[str, Optional[float]]:
    """Metrics from summed pixel counts (also suitable for micro summaries).

    A zero denominator produces ``None`` (JSON null), including all-ignored
    images. A false-positive-only or false-negative-only F1/IoU is zero.
    """
    if not isinstance(pixel_counts, Mapping):
        raise CDMapError("Pixel counts must be a mapping")
    values = {}
    for name in ("TN", "TP", "FP", "FN"):
        value = pixel_counts.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise CDMapError("Pixel count {} must be a nonnegative integer".format(name))
        values[name] = int(value)
    tn, tp, fp, fn = (values[name] for name in ("TN", "TP", "FP", "FN"))
    total = tn + tp + fp + fn
    if "valid" in pixel_counts and pixel_counts["valid"] != total:
        raise CDMapError("Valid pixel count differs from TN+TP+FP+FN")

    def ratio(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "accuracy": ratio(tp + tn, total),
    }


def validate_palette(palette: Mapping) -> Dict[str, list]:
    """Require one distinct 8-bit RGB color for each stable class."""
    if not isinstance(palette, Mapping):
        raise CDMapError("Palette must be a mapping")
    expected = set(CLASS_INDEX)
    actual = set(palette)
    if actual != expected:
        raise CDMapError(
            "Palette keys must be TN, TP, FP, FN, IGNORE (missing {}, extra {})".format(
                sorted(expected - actual), sorted(str(key) for key in actual - expected)
            )
        )
    normalized = {}
    for name in CLASS_INDEX:
        value = palette[name]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise CDMapError("Palette {} must be a list of three RGB integers".format(name))
        if any(
            isinstance(channel, bool)
            or not isinstance(channel, Integral)
            or not 0 <= channel <= 255
            for channel in value
        ):
            raise CDMapError("Palette {} RGB channels must be integers in [0, 255]".format(name))
        normalized[name] = [int(channel) for channel in value]
    if len({tuple(color) for color in normalized.values()}) != len(normalized):
        raise CDMapError("Palette colors must be distinct for each class")
    return normalized


def render(index: np.ndarray, palette: Mapping) -> np.ndarray:
    """Return an RGB uint8 array; index PNGs remain the canonical classes."""
    _index_image(index)
    colors = validate_palette(palette)
    table = np.zeros((256, 3), dtype=np.uint8)
    for name, class_id in CLASS_INDEX.items():
        table[class_id] = colors[name]
    return table[index]


def write_legend(path: Path, palette: Mapping) -> None:
    """Write a standalone PNG with class names, meanings, indices and RGB.

    Use ASCII labels so the legend requires no installed Chinese font. The
    output path must be new: a caller can never silently overwrite a legend.
    """
    colors = validate_palette(palette)
    destination = Path(path)
    labels = {
        "TN": "True negative / unchanged",
        "TP": "True positive / change",
        "FP": "False positive / false alarm",
        "FN": "False negative / missed change",
        "IGNORE": "Excluded GT pixel",
    }
    image = Image.new("RGB", (600, 230), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Change-detection error map legend", fill="black")
    draw.text((16, 30), "Index PNG: TN=0, TP=1, FP=2, FN=3, IGNORE=255", fill="black")
    for row, name in enumerate(CLASS_INDEX):
        top = 56 + row * 32
        color = tuple(colors[name])
        draw.rectangle((16, top, 44, top + 22), fill=color, outline=(70, 70, 70))
        text = "{} [{}]  {}  RGB {}".format(name, CLASS_INDEX[name], labels[name], color)
        draw.text((56, top + 5), text, fill="black")
    try:
        with destination.open("xb") as handle:
            image.save(handle, format="PNG")
    except FileExistsError as exc:
        raise CDMapError("Refusing to overwrite existing legend: {}".format(destination)) from exc
    except OSError as exc:
        raise CDMapError("Cannot write legend {}: {}".format(destination, exc)) from exc
