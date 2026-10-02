"""Safe, byte-preserving import and full verification of a frozen export."""

import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .common import CDMapError, check_unique_paths, safe_join, sha256, validate_relative
from .manifests import IMAGE_SUFFIXES, _digest, _require, load_selection_records


def _checksum(path, filename):
    path = Path(path)
    _require(path.is_file(), "Missing checksum file: " + str(path))
    try:
        lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    except (UnicodeError, OSError) as exc:
        raise CDMapError("Cannot read checksum file: " + str(exc))
    _require(len(lines) == 1, "Checksum file must contain one SHA-256 entry: " + str(path))
    parts = lines[0].split(maxsplit=1)
    digest = _digest(parts[0], str(path), optional=False)
    if len(parts) > 1:
        recorded = parts[1].lstrip("*")
        _require(recorded == filename, "Checksum filename mismatch: expected " + filename + ", recorded " + recorded)
    return digest


def _load_json(path, label):
    _require(Path(path).is_file(), "Missing " + label + ": " + str(path))
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise CDMapError("Cannot read " + label + ": " + str(exc))
    _require(isinstance(value, dict), label + " must be an object")
    return value


def verify_fixed_samples(root, expected_selection_sha256=None, expected_counts=None, expected_size=None):
    """Check original metadata, all image bytes, decoded modes/sizes and ID sets."""
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(), "Fixed sample root must be an ordinary directory: " + str(root))
    selection_path = safe_join(root, "selection.json")
    checksum_path = safe_join(root, "selection.sha256")
    original_hash = _checksum(checksum_path, "selection.json")
    if expected_selection_sha256 is not None:
        _require(original_hash == _digest(expected_selection_sha256, "expected selection", optional=False),
                 "selection.sha256 differs from expected frozen hash")
    records, hashes, selection = load_selection_records(root, selection_path, original_hash, expected_counts)
    hashes[str(checksum_path)] = sha256(checksum_path)
    complete_path = safe_join(root, "EXPORT_COMPLETE.json")
    complete = _load_json(complete_path, "EXPORT_COMPLETE.json")
    hashes[str(complete_path)] = sha256(complete_path)
    _require(complete.get("status") == "complete", "Frozen export is not complete")
    _require(complete.get("selection_sha256") == original_hash, "EXPORT_COMPLETE selection SHA-256 mismatch")
    if expected_size is not None:
        _require(isinstance(expected_size, (list, tuple)) and len(expected_size) == 2 and
                 all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in expected_size),
                 "expected_size must be [width, height] with positive integers")
        expected_size = tuple(expected_size)
    counts = {dataset: {"T1": 0, "T2": 0, "GT": 0} for dataset in selection["datasets"]}
    gt_values = {dataset: set() for dataset in selection["datasets"]}
    image_hashes = {}
    dimensions = {}
    targets = set()
    for record in records:
        for role, item in record["files"].items():
            target = item["target"]
            path = safe_join(root, target)
            _require(path.is_file() and not path.is_symlink(), "Missing or nonordinary frozen image: " + target)
            _require(path.stat().st_size == item["bytes"], "Image byte count mismatch: " + target)
            digest = sha256(path)
            _require(digest == item["sha256"].lower(), "Image SHA-256 mismatch: " + target)
            try:
                with Image.open(path) as image:
                    image.load()
                    _require(image.format == "PNG", "Frozen image is not PNG: " + target)
                    _require(image.size == (item["width"], item["height"]), "Image dimensions differ from frozen record: " + target)
                    _require(image.mode == item["mode"], "Image mode differs from frozen record: " + target)
                    _require(expected_size is None or image.size == expected_size, "Image dimensions differ from expected_size: " + target)
                    dimensions.setdefault(record["dataset"], set()).add(image.size)
                    if role == "GT":
                        gt_values[record["dataset"]].update(int(value) for value in np.unique(np.asarray(image)))
            except (OSError, ValueError, SyntaxError) as exc:
                raise CDMapError("Cannot fully decode frozen image " + target + ": " + str(exc))
            targets.add(target)
            image_hashes[target] = digest
            counts[record["dataset"]][role] += 1
    actual_images = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "Symlink inside fixed samples is forbidden: " + str(path))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(root).as_posix()
            safe_join(root, relative)
            actual_images.add(relative)
    _require(actual_images == targets, "Fixed image ID set mismatch; missing=" + str(sorted(targets - actual_images)) + "; extra=" + str(sorted(actual_images - targets)))
    total = len(targets)
    _require(complete.get("counts") == counts, "EXPORT_COMPLETE image counts mismatch")
    _require(complete.get("total_images") == total, "EXPORT_COMPLETE total_images mismatch")
    for field in ("all_sha256_match_frozen_sources", "all_images_fully_decoded", "all_triplet_dimensions_match", "original_names_and_bytes_preserved"):
        _require(complete.get(field) is True, "EXPORT_COMPLETE verification flag is not true: " + field)
    return {"status": "verified", "root": str(root.resolve()), "selection_sha256": original_hash,
            "counts": counts, "total_images": total, "manifest_hashes": hashes,
            "image_hashes": image_hashes,
            "gt_pixel_values": {dataset: sorted(values) for dataset, values in gt_values.items()},
            "dimensions": {dataset: [list(size) for size in sorted(sizes)] for dataset, sizes in dimensions.items()},
            "all_sha256_match_frozen_sources": True, "all_images_fully_decoded": True,
            "all_triplet_dimensions_match": True, "original_names_and_bytes_preserved": True}


def _inspect_archive(archive):
    """Inspect the complete tar tree before extraction, including ancestor clashes."""
    entries = []
    roots = set()
    seen = {}
    nodes = {}
    try:
        members = archive.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise CDMapError("Cannot inspect archive: " + str(exc))
    _require(bool(members), "Archive is empty")
    for member in members:
        _require(member.isfile() or member.isdir(), "Archive links/special files are forbidden: " + member.name)
        name = member.name.rstrip("/") if member.isdir() else member.name
        validate_relative(name)
        parts = PurePosixPath(name).parts
        roots.add(parts[0])
        folded = name.casefold()
        _require(folded not in seen, "Archive duplicate/case collision: " + member.name)
        seen[folded] = (name, member.isdir())
        for length in range(1, len(parts) + 1):
            node_name = "/".join(parts[:length])
            node_directory = length < len(parts) or member.isdir()
            node_key = node_name.casefold()
            if node_key in nodes:
                _require(nodes[node_key] == (node_name, node_directory),
                         "Archive case or file/directory path conflict: " + member.name)
            else:
                nodes[node_key] = (node_name, node_directory)
        entries.append((member, parts))
    _require(len(roots) == 1, "Archive must have exactly one outer directory")
    for member, parts in entries:
        _require(len(parts) > 1 or member.isdir(), "Archive top level must be a directory")
        for length in range(1, len(parts)):
            parent = "/".join(parts[:length]).casefold()
            _require(parent not in seen or seen[parent][1], "Archive file/directory path conflict: " + member.name)
    return [(member, "/".join(parts[1:])) for member, parts in entries if len(parts) > 1]


def _preflight_merge(stage, destination):
    _require(not destination.is_symlink(), "Destination symlink is forbidden: " + str(destination))
    _require(not destination.exists() or destination.is_dir(), "Destination must be a directory: " + str(destination))
    stage_files = [path for path in stage.rglob("*") if path.is_file()]
    staged_relatives = {path.relative_to(stage).as_posix() for path in stage_files}
    expected_images = {relative for relative in staged_relatives if Path(relative).suffix.lower() in IMAGE_SUFFIXES}
    overlay_nodes = {}
    existing_paths = []
    if destination.exists():
        existing_paths = list(destination.rglob("*"))
        check_unique_paths([path for path in existing_paths if path.is_file()], "Existing fixed sample")
        for existing in existing_paths:
            _require(not existing.is_symlink(), "Existing symlink is forbidden: " + str(existing))
            relative = existing.relative_to(destination).as_posix()
            safe_join(destination, relative)
            if existing.is_file() and existing.suffix.lower() in IMAGE_SUFFIXES:
                _require(relative in expected_images, "Existing extra image; refusing import: " + relative)
    # Include directories, even empty ones: D/ and d/ are a portability conflict
    # on a case-sensitive filesystem and alias each other on Windows.
    for tree_root, tree_paths in ((destination, existing_paths), (stage, list(stage.rglob("*")))):
        for path in tree_paths:
            relative = path.relative_to(tree_root).as_posix()
            node = (relative, path.is_dir())
            key = relative.casefold()
            if key in overlay_nodes:
                _require(overlay_nodes[key] == node,
                         "Existing case or file/directory conflict; refusing import: " + relative)
            else:
                overlay_nodes[key] = node
    for staged in stage.rglob("*"):
        relative = staged.relative_to(stage).as_posix()
        target = safe_join(destination, relative)
        if target.exists():
            _require(target.is_dir() == staged.is_dir(), "Existing file/directory conflict: " + relative)
            if staged.is_file():
                _require(target.stat().st_size == staged.stat().st_size and sha256(target) == sha256(staged),
                         "Existing file differs; refusing to overwrite: " + relative)
        for parent in target.parents:
            if parent == destination.parent:
                break
            _require(not parent.exists() or parent.is_dir(), "Existing ancestor is a file: " + str(parent))
    return stage_files


def _guard_destination(destination, resolved_destination):
    """Reject redirected destination roots before staging or each merge write."""
    for path in (destination,) + tuple(destination.parents):
        _require(not path.is_symlink(), "Destination ancestor symlink is forbidden: " + str(path))
        # pathlib.is_junction was introduced after the Python 3.9 target.
        if hasattr(path, "is_junction"):
            _require(not path.is_junction(), "Destination ancestor junction is forbidden: " + str(path))
    _require(str(destination.resolve()).casefold() == str(resolved_destination).casefold(),
             "Destination changed during import: " + str(destination))


def import_archive(archive, checksum_file, destination, expected_archive_sha256=None,
                   expected_selection_sha256=None, expected_counts=None, expected_size=None):
    """Strip one outer folder, validate staging, then add only absent identical files."""
    archive_path = Path(archive)
    destination = Path(destination).absolute()
    _require(archive_path.is_file(), "Missing archive: " + str(archive_path))
    expected = _digest(expected_archive_sha256, "expected archive")
    if checksum_file is not None:
        recorded = _checksum(checksum_file, archive_path.name)
        _require(expected is None or recorded == expected, "Archive checksum file differs from expected SHA-256")
        expected = recorded
    _require(expected is not None, "Provide checksum_file or expected_archive_sha256")
    archive_hash = sha256(archive_path)
    _require(archive_hash == expected, "Archive SHA-256 mismatch: " + str(archive_path))
    resolved_destination = destination.resolve()
    _guard_destination(destination, resolved_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = None
    try:
        with tarfile.open(archive_path, "r:gz") as handle:
            entries = _inspect_archive(handle)
            stage = Path(tempfile.mkdtemp(prefix=".fixed-import-", dir=str(destination.parent)))
            for member, relative in entries:
                target = safe_join(stage, relative)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    _require(target.stat().st_size == member.size, "Truncated archive member: " + member.name)
        staged_report = verify_fixed_samples(stage, expected_selection_sha256, expected_counts, expected_size)
        _require(sha256(archive_path) == archive_hash, "Source archive changed before merge")
        _guard_destination(destination, resolved_destination)
        files = _preflight_merge(stage, destination)
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        reused = 0
        # The export marker is placed last, after all other bytes are present.
        files.sort(key=lambda path: (path.relative_to(stage).as_posix() == "EXPORT_COMPLETE.json", path.relative_to(stage).as_posix()))
        for source in files:
            _guard_destination(destination, resolved_destination)
            target = safe_join(destination, source.relative_to(stage).as_posix())
            if target.exists():
                _require(target.is_file() and not target.is_symlink() and
                         target.stat().st_size == source.stat().st_size and sha256(target) == sha256(source),
                         "Existing file changed after preflight; refusing import: " + str(target))
                reused += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            copied += 1
        report = verify_fixed_samples(destination, expected_selection_sha256, expected_counts, expected_size)
        _require(report["selection_sha256"] == staged_report["selection_sha256"], "Imported selection changed")
        _require(sha256(archive_path) == archive_hash, "Source archive changed during import")
        report.update({"archive_sha256": archive_hash, "archive": str(archive_path.resolve()),
                       "copied_files": copied, "reused_files": reused})
        return report
    except (tarfile.TarError, OSError) as exc:
        raise CDMapError("Import failed; existing files retained: " + str(exc))
    finally:
        if stage is not None:
            _require(stage.resolve().parent == destination.parent.resolve() and stage.name.startswith(".fixed-import-"),
                     "Unsafe staging cleanup path")
            shutil.rmtree(stage)
