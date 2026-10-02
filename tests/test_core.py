"""Independent synthetic mask oracles; these are never model predictions."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cd_error_maps.common import CDMapError
from cd_error_maps.core import (
    CLASS_INDEX,
    DEFAULT_PALETTE,
    classify,
    counts,
    metrics,
    render,
    validate_palette,
    write_legend,
)
from cd_error_maps.masks import read_mask


def temporary_directory():
    # Keep synthetic scratch files in the ignored project directory. This
    # also works in restricted runners whose global temp directory is read-only.
    scratch = Path(__file__).resolve().parents[1] / ".tmp"
    scratch.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=scratch)


class CoreTests(unittest.TestCase):
    def test_hand_computed_all_four_directions(self):
        # GT 0/pred 0 -> TN; GT 1/pred 1 -> TP;
        # GT 0/pred 1 -> FP; GT 1/pred 0 -> FN.
        gt = np.array([[False, True], [False, True]], dtype=bool)
        prediction = np.array([[False, True], [True, False]], dtype=bool)
        actual = classify(gt, prediction)
        np.testing.assert_array_equal(actual, [[0, 1], [2, 3]])
        self.assertEqual(actual.dtype, np.dtype(np.uint8))
        self.assertEqual(
            counts(actual),
            {"TN": 1, "TP": 1, "FP": 1, "FN": 1, "ignored": 0,
             "valid": 4, "width": 2, "height": 2},
        )
        actual_metrics = metrics(counts(actual))
        self.assertEqual(actual_metrics["precision"], 0.5)
        self.assertEqual(actual_metrics["recall"], 0.5)
        self.assertEqual(actual_metrics["f1"], 0.5)
        self.assertEqual(actual_metrics["iou"], 1 / 3)
        self.assertEqual(actual_metrics["accuracy"], 0.5)

    def test_full_background_and_foreground(self):
        for foreground, expected in ((False, 0), (True, 1)):
            with self.subTest(foreground=foreground):
                data = np.full((3, 5), foreground, dtype=bool)
                np.testing.assert_array_equal(classify(data, data), np.full((3, 5), expected))
        background_counts = counts(np.zeros((3, 5), dtype=np.uint8))
        self.assertEqual(
            metrics(background_counts),
            {"precision": None, "recall": None, "f1": None, "iou": None, "accuracy": 1.0},
        )
        foreground_counts = counts(np.ones((3, 5), dtype=np.uint8))
        self.assertEqual(metrics(foreground_counts), dict.fromkeys(metrics(foreground_counts), 1.0))

    def test_ignore_and_all_ignored(self):
        gt = np.array([[False, True], [False, True]], dtype=bool)
        prediction = np.array([[False, True], [True, False]], dtype=bool)
        valid = np.array([[True, False], [True, False]], dtype=bool)
        np.testing.assert_array_equal(classify(gt, prediction, valid), [[0, 255], [2, 255]])
        result = classify(gt, prediction, np.zeros((2, 2), dtype=bool))
        np.testing.assert_array_equal(result, [[255, 255], [255, 255]])
        self.assertEqual(counts(result)["ignored"], 4)
        self.assertEqual(counts(result)["valid"], 0)
        self.assertEqual(metrics(counts(result)), dict.fromkeys(metrics(counts(result)), None))

    def test_metrics_micro_counts_and_zero_denominators(self):
        result = metrics({"TN": 6, "TP": 3, "FP": 1, "FN": 2})
        self.assertEqual(result["precision"], 3 / 4)
        self.assertEqual(result["recall"], 3 / 5)
        self.assertEqual(result["f1"], 2 / 3)
        self.assertEqual(result["iou"], 1 / 2)
        self.assertEqual(result["accuracy"], 3 / 4)
        self.assertEqual(metrics({"TN": 0, "TP": 0, "FP": 2, "FN": 0})["f1"], 0)
        self.assertIsNone(metrics({"TN": 0, "TP": 0, "FP": 2, "FN": 0})["recall"])
        self.assertIsNone(metrics({"TN": 0, "TP": 0, "FP": 0, "FN": 2})["precision"])
        for invalid in (
            {"TN": -1, "TP": 0, "FP": 0, "FN": 0},
            {"TN": 1.0, "TP": 0, "FP": 0, "FN": 0},
            {"TN": True, "TP": 0, "FP": 0, "FN": 0},
            {"TN": 0, "TP": 0, "FP": 0},
            {"TN": 0, "TP": 1, "FP": 0, "FN": 0, "valid": 5},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(CDMapError):
                metrics(invalid)

    def test_arrays_are_strict_and_dimensions_match(self):
        good = np.zeros((2, 3), dtype=bool)
        for bad in (np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 3, 1), dtype=bool),
                    np.zeros((0, 3), dtype=bool), [[False, False, False]]):
            with self.subTest(bad=repr(bad)), self.assertRaises(CDMapError):
                classify(bad, good)
        with self.assertRaisesRegex(CDMapError, "dimensions"):
            classify(good, np.zeros((3, 2), dtype=bool))
        with self.assertRaisesRegex(CDMapError, "dimensions"):
            classify(good, good, np.zeros((2, 2), dtype=bool))
        with self.assertRaises(CDMapError):
            classify(good, good, np.ones((2, 3), dtype=np.uint8))
        for bad in (np.array([[4]], dtype=np.uint8), np.array([[0]], dtype=np.int32)):
            with self.assertRaises(CDMapError):
                counts(bad)
            with self.assertRaises(CDMapError):
                render(bad, DEFAULT_PALETTE)

    def test_palette_and_lossless_output_roundtrip(self):
        index = np.array([[0, 1, 2, 3, 255]], dtype=np.uint8)
        rgb = render(index, DEFAULT_PALETTE)
        np.testing.assert_array_equal(
            rgb,
            [[[0, 0, 0], [255, 255, 255], [230, 159, 0], [0, 114, 178], [128, 128, 128]]],
        )
        self.assertEqual(rgb.dtype, np.dtype(np.uint8))
        with temporary_directory() as temporary:
            root = Path(temporary)
            Image.fromarray(index).save(root / "index.png")
            Image.fromarray(rgb).save(root / "rgb.png")
            with Image.open(root / "index.png") as image:
                self.assertEqual(image.mode, "L")
                np.testing.assert_array_equal(np.array(image), [[0, 1, 2, 3, 255]])
            with Image.open(root / "rgb.png") as image:
                self.assertEqual(image.mode, "RGB")
                np.testing.assert_array_equal(np.array(image), rgb)
            write_legend(root / "legend.png", DEFAULT_PALETTE)
            with Image.open(root / "legend.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((20, 60)), (0, 0, 0))
                self.assertEqual(image.getpixel((20, 92)), (255, 255, 255))
            with self.assertRaisesRegex(CDMapError, "overwrite"):
                write_legend(root / "legend.png", DEFAULT_PALETTE)

    def test_palette_validation(self):
        normalized = validate_palette(DEFAULT_PALETTE)
        self.assertEqual(normalized, DEFAULT_PALETTE)
        self.assertIsNot(normalized["TN"], DEFAULT_PALETTE["TN"])
        custom = dict(DEFAULT_PALETTE, TN=[10, 20, 30])
        np.testing.assert_array_equal(render(np.array([[0]], dtype=np.uint8), custom), [[[10, 20, 30]]])
        invalid_palettes = (
            {"TN": [0, 0, 0]},
            dict(DEFAULT_PALETTE, EXTRA=[2, 3, 4]),
            dict(DEFAULT_PALETTE, TN=[0, 0]),
            dict(DEFAULT_PALETTE, TN=[-1, 0, 0]),
            dict(DEFAULT_PALETTE, TN=[256, 0, 0]),
            dict(DEFAULT_PALETTE, TN=[False, 0, 0]),
            dict(DEFAULT_PALETTE, TN=[0.1, 0, 0]),
            dict(DEFAULT_PALETTE, TP=[0, 0, 0]),
        )
        for palette in invalid_palettes:
            with self.subTest(palette=palette), self.assertRaises(CDMapError):
                validate_palette(palette)


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, values, name="mask.png", mode=None):
        path = self.root / name
        image = Image.fromarray(np.array(values, dtype=np.uint8))
        if mode is not None:
            image = image.convert(mode)
        image.save(path)
        return path

    def test_both_encodings_and_255_is_change_by_default(self):
        for encoding, foreground in (("binary_01", 1), ("binary_0255", 255)):
            path = self.save([[0, foreground], [foreground, 0]])
            change, valid = read_mask(path, encoding)
            np.testing.assert_array_equal(change, [[False, True], [True, False]])
            np.testing.assert_array_equal(valid, [[True, True], [True, True]])

    def test_explicit_ignore_distinct_from_binary_values(self):
        path = self.save([[0, 1, 255]])
        change, valid = read_mask(path, "binary_01", ignore_value=255)
        np.testing.assert_array_equal(change, [[False, True, False]])
        np.testing.assert_array_equal(valid, [[True, True, False]])
        path = self.save([[0, 255, 128]])
        change, valid = read_mask(path, "binary_0255", ignore_value=128)
        np.testing.assert_array_equal(change, [[False, True, False]])
        np.testing.assert_array_equal(valid, [[True, True, False]])
        with self.assertRaises(CDMapError):
            read_mask(path, "binary_0255", ignore_value=255)
        with self.assertRaises(CDMapError):
            read_mask(path, "binary_0255", ignore_value=0)
        for invalid in (True, -1, 256, 128.0, "128"):
            with self.subTest(ignore=invalid), self.assertRaises(CDMapError):
                read_mask(path, "binary_0255", ignore_value=invalid)

    def test_invalid_prediction_on_ignored_gt_position_still_rejected(self):
        gt_path = self.save([[0, 255]], "gt.png")
        read_mask(gt_path, "binary_01", ignore_value=255)
        prediction_path = self.save([[0, 2]], "prediction.png")
        with self.assertRaisesRegex(CDMapError, "Illegal values"):
            read_mask(prediction_path, "binary_01")

    def test_wrong_encoding_illegal_values_or_unreadable(self):
        path = self.save([[0, 255]])
        with self.assertRaisesRegex(CDMapError, "Illegal values"):
            read_mask(path, "binary_01")
        path = self.save([[0, 1]])
        with self.assertRaisesRegex(CDMapError, "Illegal values"):
            read_mask(path, "binary_0255")
        with self.assertRaises(CDMapError):
            read_mask(path, "guess")
        with self.assertRaisesRegex(CDMapError, "Cannot decode"):
            read_mask(self.root / "missing.png", "binary_01")
        path.write_bytes(b"not a PNG")
        with self.assertRaisesRegex(CDMapError, "Cannot decode"):
            read_mask(path, "binary_01")

    def test_reject_ambiguous_modes_and_non_png(self):
        for mode in ("P", "RGB", "RGBA", "1"):
            with self.subTest(mode=mode):
                path = self.save([[0, 255]], mode=mode)
                with self.assertRaisesRegex(CDMapError, "mode L"):
                    read_mask(path, "binary_0255")
        path = self.root / "sixteen.png"
        Image.fromarray(np.array([[0, 255]], dtype=np.uint16)).save(path)
        with self.assertRaisesRegex(CDMapError, "mode L"):
            read_mask(path, "binary_0255")
        path = self.save([[0, 255]], name="mask.bmp")
        with self.assertRaisesRegex(CDMapError, "must be a PNG"):
            read_mask(path, "binary_0255")


if __name__ == "__main__":
    unittest.main()
