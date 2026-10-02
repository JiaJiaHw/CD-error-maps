"""Shared integrity and portable path helpers."""
import hashlib
import re
from pathlib import Path


class CDMapError(ValueError):
    """An actionable input, integrity or output-contract violation."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise CDMapError("Expected a nonempty POSIX relative path: {!r}".format(value))
    parts = value.split("/")
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    for part in parts:
        if (part in ("", ".", "..") or part.endswith((" ", "."))
                or any(ord(c) < 32 or c in ':*?"<>|' for c in part)
                or reserved.match(part)):
            raise CDMapError("Unsafe or nonportable relative path: {!r}".format(value))
    return value


def safe_join(root, relative):
    validate_relative(relative)
    root = Path(root).resolve()
    path = root.joinpath(*relative.split("/")).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise CDMapError("Path escapes root {}: {}".format(root, relative))
    return path


def output_id(sample_id):
    validate_relative(sample_id)
    return sample_id if sample_id.lower().endswith(".png") else sample_id + ".png"


def check_unique_paths(paths, label):
    seen = {}
    for path in paths:
        key = str(Path(path).resolve()).casefold()
        if key in seen:
            raise CDMapError("{} path collision: {} / {}".format(label, seen[key], path))
        seen[key] = path
    keys = set(seen)
    for path in paths:
        for parent in Path(path).resolve().parents:
            if str(parent).casefold() in keys:
                raise CDMapError("{} file/directory path conflict: {}".format(label, path))
