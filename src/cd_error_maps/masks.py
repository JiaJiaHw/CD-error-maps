"""Strict binary mask decoding; no resizing or implicit ignore values."""

from numbers import Integral
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .common import CDMapError


ENCODINGS = {"binary_01": (0, 1), "binary_0255": (0, 255)}


def validate_gt_value_map(value_map, ignore_value=None):
    """Validate an explicit GT-only map; values outside it remain illegal."""
    if not isinstance(value_map, dict) or set(value_map) != {"background", "change"}:
        raise CDMapError("GT value map must contain only background and change lists")
    result = {}
    for role in ("background", "change"):
        values = value_map[role]
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or not 0 <= value <= 255 for value in values)):
            raise CDMapError("GT value map {} must be a nonempty list of integers in [0, 255]".format(role))
        if len(values) != len(set(values)):
            raise CDMapError("Duplicate GT value map {} values".format(role))
        result[role] = list(values)
    if set(result["background"]) & set(result["change"]):
        raise CDMapError("GT value map background and change values overlap")
    if ignore_value is not None and ignore_value in result["background"] + result["change"]:
        raise CDMapError("GT ignore value overlaps GT value map background/change values")
    return result


def read_mask(
    path: Path, encoding: str, ignore_value: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (change, valid) boolean arrays from an explicitly encoded PNG.

    Only 8-bit grayscale PNG (Pillow mode ``L``) is accepted. Palette, RGB,
    boolean, and 16-bit files must be converted explicitly by their producer.
    ``ignore_value`` is intended for GT only; prediction callers omit it so
    every prediction pixel is checked, including positions ignored by GT.
    """
    return _read_mask(path, encoding, ignore_value)


def read_gt_mask(path: Path, encoding: str, ignore_value: Optional[int] = None,
                 value_map=None) -> Tuple[np.ndarray, np.ndarray]:
    """Read GT with an optional explicit multi-value class map, without rewriting it."""
    return _read_mask(path, encoding, ignore_value, value_map)


def _read_mask(path, encoding, ignore_value=None, gt_value_map=None):
    if encoding not in ENCODINGS:
        raise CDMapError("Unknown mask encoding: {!r}".format(encoding))
    background, foreground = ENCODINGS[encoding]
    mapping = validate_gt_value_map(gt_value_map) if gt_value_map is not None else None
    background_values = mapping["background"] if mapping else [background]
    change_values = mapping["change"] if mapping else [foreground]
    if ignore_value is not None:
        if (
            isinstance(ignore_value, bool)
            or not isinstance(ignore_value, Integral)
            or not 0 <= ignore_value <= 255
        ):
            raise CDMapError("GT ignore value must be an integer in [0, 255]")
        if ignore_value in [background, foreground] + background_values + change_values:
            raise CDMapError("GT ignore value must differ from background and change values")

    source = Path(path)
    try:
        with Image.open(source) as image:
            if image.format != "PNG":
                raise CDMapError("Mask must be a PNG file: {}".format(source))
            if image.mode != "L":
                raise CDMapError(
                    "Mask must use 8-bit grayscale PNG mode L, got {}: {}".format(
                        image.mode, source
                    )
                )
            values = np.array(image, dtype=np.uint8, copy=True)
    except CDMapError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
        raise CDMapError("Cannot decode mask {}: {}".format(source, exc)) from exc

    allowed = set(background_values + change_values)
    if ignore_value is not None:
        allowed.add(int(ignore_value))
    unexpected = sorted(set(int(value) for value in np.unique(values)) - allowed)
    if unexpected:
        raise CDMapError(
            "Illegal values {} for {} (allowed {}): {}".format(
                unexpected, "GT value map" if mapping else encoding, sorted(allowed), source
            )
        )
    valid = np.ones(values.shape, dtype=np.bool_)
    if ignore_value is not None:
        valid = values != ignore_value
    return (np.isin(values, change_values) if mapping else values == foreground), valid
