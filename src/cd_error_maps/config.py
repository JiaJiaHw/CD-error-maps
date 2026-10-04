"""Project-relative JSON configuration. Experiment constraints are local data."""
import json
import string
from pathlib import Path

from .common import CDMapError, validate_relative
from .core import validate_palette
from .masks import validate_gt_value_map

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PALETTE = {"TN": [0, 0, 0], "TP": [255, 255, 255], "FP": [255, 0, 0],
           "FN": [0, 255, 0], "IGNORE": [128, 128, 128]}
PATH_KEYS = ("samples_root", "selection_file", "samples_manifest", "prediction_root",
             "prediction_manifest", "output_root")
ALLOWED = set(PATH_KEYS) | {"schema_version", "datasets", "models", "combinations",
    "prediction_path_template", "expected_selection_sha256", "expected_manifest_sha256",
    "expected_prediction_manifest_sha256", "expected_counts", "expected_size",
    "gt_encoding", "prediction_encoding", "ignore", "palette", "encodings", "model_aliases", "gt_value_maps"}


def _path(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CDMapError("Configuration paths must be nonempty strings")
    try:
        path = Path(value)
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise CDMapError("Cannot resolve configuration path {!r}: {}".format(value, exc)) from exc


def _names(value, label):
    if not isinstance(value, list) or not value:
        raise CDMapError("{} must be a nonempty list".format(label))
    for name in value:
        validate_relative(name)
        if "/" in name:
            raise CDMapError("{} names must be single path components".format(label))
    if len(set(x.casefold() for x in value)) != len(value):
        raise CDMapError("Duplicate or case-conflicting {}".format(label))


def load_config(path=None):
    if path is None:
        local = PROJECT_ROOT / "configs/local.fixed100.json"
        path = local if local.exists() else PROJECT_ROOT / "configs/fixed_samples.example.json"
    else:
        path = _path(str(path))
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RuntimeError) as exc:
        raise CDMapError("Cannot read config {}: {}".format(path, exc))
    if not isinstance(raw, dict) or set(raw) - ALLOWED:
        raise CDMapError("Unknown configuration keys or invalid object: {}".format(
            sorted(set(raw) - ALLOWED) if isinstance(raw, dict) else type(raw).__name__))
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise CDMapError("schema_version must be 1")
    config = dict(raw)
    config.setdefault("samples_root", "data/fixed_samples")
    config.setdefault("prediction_root", "data/predictions")
    config.setdefault("output_root", "outputs")
    for key in PATH_KEYS:
        config[key] = _path(config.get(key))
    for key in ("samples_root", "prediction_root", "output_root"):
        if config[key] is None:
            raise CDMapError("{} must be a nonempty path string".format(key))
    if bool(config["selection_file"]) == bool(config["samples_manifest"]):
        raise CDMapError("Choose exactly one selection_file or samples_manifest")
    if "prediction_path_template" not in config:
        config["prediction_path_template"] = None if config["prediction_manifest"] else "{dataset}/{model}/{sample_id}"
    if bool(config["prediction_manifest"]) == bool(config["prediction_path_template"]):
        raise CDMapError("Choose exactly one prediction_manifest or prediction_path_template")
    template = config["prediction_path_template"]
    if template is not None and (not isinstance(template, str) or not template):
        raise CDMapError("prediction_path_template must be null or a nonempty string")
    if template:
        try:
            parsed = list(string.Formatter().parse(template))
            fields = [field for _, field, _, _ in parsed if field is not None]
            if len(fields) != 3 or set(fields) != {"dataset", "model", "sample_id"} or any(spec or conv for _, _, spec, conv in parsed):
                raise ValueError("requires dataset, model, sample_id fields without format specs")
            validate_relative(template.format(dataset="D", model="M", sample_id="S.png"))
        except (ValueError, KeyError, IndexError) as exc:
            raise CDMapError("Invalid prediction_path_template: {}".format(exc))
    _names(config.get("datasets"), "datasets")
    _names(config.get("models"), "models")
    config.setdefault("combinations", [])
    if not isinstance(config["combinations"], list):
        raise CDMapError("combinations must be a list of {dataset, model} objects")
    seen_combinations = set()
    for pair in config["combinations"]:
        if not isinstance(pair, dict) or set(pair) != {"dataset", "model"} or pair["dataset"] not in config["datasets"] or pair["model"] not in config["models"]:
            raise CDMapError("Invalid combinations entry: {}".format(pair))
        key = (pair["dataset"], pair["model"])
        if key in seen_combinations:
            raise CDMapError("Duplicate combinations entry: {}:{}".format(*key))
        seen_combinations.add(key)
    config.setdefault("gt_encoding", "binary_0255")
    config.setdefault("prediction_encoding", "binary_0255")
    config.setdefault("encodings", {})
    if not isinstance(config["encodings"], dict):
        raise CDMapError("encodings must be an object keyed by dataset and model")
    for dataset, rules in config["encodings"].items():
        if dataset not in config["datasets"] or not isinstance(rules, dict):
            raise CDMapError("Invalid encoding override dataset")
        for model, values in rules.items():
            if model not in config["models"] or not isinstance(values, dict) or set(values) - {"gt_encoding", "prediction_encoding"}:
                raise CDMapError("Invalid encoding override model or keys")
    for dataset in config["datasets"]:
        for model in config["models"]:
            for encoding in encoding_for(config, dataset, model):
                if encoding not in ("binary_01", "binary_0255"):
                    raise CDMapError("Unsupported binary encoding: {}".format(encoding))
    config.setdefault("gt_value_maps", {})
    if not isinstance(config["gt_value_maps"], dict):
        raise CDMapError("gt_value_maps must be an object keyed by dataset")
    for dataset, mapping in config["gt_value_maps"].items():
        if dataset not in config["datasets"]:
            raise CDMapError("Unknown GT value map dataset: {}".format(dataset))
        config["gt_value_maps"][dataset] = validate_gt_value_map(mapping)
    config.setdefault("ignore", None)
    if config["ignore"] is not None:
        ig = config["ignore"]
        if not isinstance(ig, dict) or set(ig) != {"gt_value"} or type(ig["gt_value"]) is not int or not 0 <= ig["gt_value"] <= 255:
            raise CDMapError("ignore must be null or {gt_value: integer 0..255}")
        for dataset in config["datasets"]:
            mapping = config["gt_value_maps"].get(dataset)
            if mapping is not None:
                validate_gt_value_map(mapping, ig["gt_value"])
            for model in config["models"]:
                fg = 1 if encoding_for(config, dataset, model)[0] == "binary_01" else 255
                if ig["gt_value"] in (0, fg):
                    raise CDMapError("GT ignore value overlaps background/foreground encoding")
    config.setdefault("palette", PALETTE)
    config["palette"] = validate_palette(config["palette"])
    config.setdefault("expected_counts", {})
    if not isinstance(config["expected_counts"], dict):
        raise CDMapError("expected_counts must be an object keyed by dataset")
    for dataset, count in config["expected_counts"].items():
        if dataset not in config["datasets"] or type(count) is not int or count <= 0:
            raise CDMapError("Invalid expected_counts")
    size = config.get("expected_size")
    if size is not None and (not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size)):
        raise CDMapError("expected_size must be [positive width, positive height]")
    for key in ("expected_selection_sha256", "expected_manifest_sha256", "expected_prediction_manifest_sha256"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            raise CDMapError("{} must be a lowercase SHA-256".format(key))
    config.setdefault("model_aliases", {})
    if not isinstance(config["model_aliases"], dict):
        raise CDMapError("model_aliases must be an object keyed by canonical model")
    seen_aliases = {}
    canonical_models = {model.casefold(): model for model in config["models"]}
    for model, aliases in config["model_aliases"].items():
        if model not in config["models"]:
            raise CDMapError("Unknown canonical model in model_aliases: {}".format(model))
        _names(aliases, "model_aliases for {}".format(model))
        for alias in aliases:
            alias_key = alias.casefold()
            owner = seen_aliases.get(alias_key, canonical_models.get(alias_key))
            if owner is not None and owner != model:
                raise CDMapError("Model alias is ambiguous: {}".format(alias))
            seen_aliases[alias_key] = model
    # An output root inside or enclosing an input root risks overwriting source data.
    # Compare component spellings portably, including Linux configurations whose
    # paths would otherwise alias when moved to a case-insensitive filesystem.
    for input_root in (config["samples_root"], config["prediction_root"]):
        out = config["output_root"]
        out_parts = tuple(part.casefold() for part in out.parts)
        input_parts = tuple(part.casefold() for part in input_root.parts)
        shorter = min(len(out_parts), len(input_parts))
        if out_parts[:shorter] == input_parts[:shorter]:
            raise CDMapError("output_root overlaps input root: {}".format(input_root))
    return config


def encoding_for(config, dataset, model):
    rule = config.get("encodings", {}).get(dataset, {}).get(model, {})
    return (rule.get("gt_encoding", config["gt_encoding"]),
            rule.get("prediction_encoding", config["prediction_encoding"]))


def effective_config(config):
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
