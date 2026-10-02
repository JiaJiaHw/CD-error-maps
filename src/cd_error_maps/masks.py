"""Strict binary mask decoding; no resizing or implicit ignore values."""

from numbers import Integral
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .common import CDMapError


ENCODINGS = {"binary_01": (0, 1), "binary_0255": (0, 255)}


def read_mask(
    path: Path, encoding: str, ignore_value: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (change, valid) boolean arrays from an explicitly encoded PNG.

    Only 8-bit grayscale PNG (Pillow mode ``L``) is accepted. Palette, RGB,
    boolean, and 16-bit files must be converted explicitly by their producer.
    ``ignore_value`` is intended for GT only; prediction callers omit it so
    every prediction pixel is checked, including positions ignored by GT.
    """
    if encoding not in ENCODINGS:
        raise CDMapError("Unknown mask encoding: {!r}".format(encoding))
    background, foreground = ENCODINGS[encoding]
    if ignore_value is not None:
        if (
            isinstance(ignore_value, bool)
            or not isinstance(ignore_value, Integral)
            or not 0 <= ignore_value <= 255
        ):
            raise CDMapError("GT ignore value must be an integer in [0, 255]")
        if ignore_value in (background, foreground):
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

    allowed = {background, foreground}
    if ignore_value is not None:
        allowed.add(int(ignore_value))
    unexpected = sorted(set(int(value) for value in np.unique(values)) - allowed)
    if unexpected:
        raise CDMapError(
            "Illegal values {} for {} (allowed {}): {}".format(
                unexpected, encoding, sorted(allowed), source
            )
        )
    valid = np.ones(values.shape, dtype=np.bool_)
    if ignore_value is not None:
        valid = values != ignore_value
    return values == foreground, valid
