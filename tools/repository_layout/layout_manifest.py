"""Capture and verify the immutable repository-layout migration ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


_ALLOWED_CONFIG_KEYS = {
    "identity_pins",
    "prefix_moves",
    "exact_moves",
    "pattern_moves",
    "rewrite_allowlist",
    "regenerate_allowlist",
}
_ALLOWED_DISPOSITIONS = {
    "move-byte-identical",
    "same-byte-identical",
    "approved-rewrite",
    "proven-generated-regenerate",
}


class LayoutError(ValueError):
    """The mapping, baseline, or ledger violates migration invariants."""


@dataclass(frozen=True)
class LedgerRow:
    old_path: str
    new_path: str
    git_mode: str
    blob_oid: str
    byte_length: int
    sha256: str
    disposition: str


@dataclass(frozen=True)
class VerificationReport:
    input_manifest_rows: int
    migrated_legacy_rows: int
    unmapped: int
    collisions: int
    lost_blobs: int
    unexpected_rewrites: int
    mode_changes: int

    @property
    def ok(self) -> bool:
        return (
            self.input_manifest_rows == self.migrated_legacy_rows
            and self.unmapped == 0
            and self.collisions == 0
            and self.lost_blobs == 0
            and self.unexpected_rewrites == 0
            and self.mode_changes == 0
        )

@dataclass(frozen=True)
class ApplyReport:
    legacy_rows: int
    move_rows: int
    identity_rows: int
    pending_moves: int
    completed_moves: int
    conflicts: int
    missing: int
    moved: int
    dry_run: bool

    @property
    def ok(self) -> bool:
        return self.conflicts == 0 and self.missing == 0


@dataclass(frozen=True)
class LayoutMapping:
    identity_pins: tuple[str, ...]
    prefix_moves: tuple[tuple[str, str], ...]
    exact_moves: tuple[tuple[str, str], ...]
    pattern_moves: tuple[tuple[str, str], ...]
    rewrite_allowlist: frozenset[str]
    regenerate_allowlist: frozenset[str]

    @classmethod
    def from_file(cls, path: Path) -> "LayoutMapping":
        try:
            raw = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LayoutError(f"cannot load layout mapping {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise LayoutError("layout mapping must be a JSON object")
        unknown = set(raw) - _ALLOWED_CONFIG_KEYS
        if unknown:
            raise LayoutError(f"unknown layout mapping keys: {sorted(unknown)!r}")

        identity_pins = _string_list(raw.get("identity_pins", []), "identity_pins")
        prefix_moves = _string_mapping(raw.get("prefix_moves", {}), "prefix_moves")
        exact_moves = _string_mapping(raw.get("exact_moves", {}), "exact_moves")
        pattern_moves = _string_mapping(raw.get("pattern_moves", {}), "pattern_moves")
        rewrite = frozenset(
            _string_list(raw.get("rewrite_allowlist", []), "rewrite_allowlist")
        )
        regenerate = frozenset(
            _string_list(raw.get("regenerate_allowlist", []), "regenerate_allowlist")
        )

        for pattern in identity_pins:
            _validate_source_pattern(pattern, "identity pin")
        for source, destination in prefix_moves:
            if not source.endswith("/") or not destination.endswith("/"):
                raise LayoutError("prefix move sources and destinations must end in '/'")
            _validate_relative_path(source[:-1], "prefix source")
            _validate_destination(destination + "probe")
        for source, destination in exact_moves:
            _validate_relative_path(source, "exact source")
            _validate_destination(destination)
        for pattern, destination_prefix in pattern_moves:
            if not pattern.endswith("/**") or not destination_prefix.endswith("/"):
                raise LayoutError(
                    "pattern moves must use a '/**' source and a '/' destination"
                )
            _validate_relative_path(pattern[:-3], "pattern source")
            _validate_destination(destination_prefix + "probe")
        for path_value in rewrite | regenerate:
            _validate_relative_path(path_value, "allowlisted path")
        overlap = rewrite & regenerate
        if overlap:
            raise LayoutError(
                "paths cannot be both rewrite and regenerate allowlisted: "
                f"{sorted(overlap)!r}"
            )

        mapping = cls(
            identity_pins=identity_pins,
            prefix_moves=prefix_moves,
            exact_moves=exact_moves,
            pattern_moves=pattern_moves,
            rewrite_allowlist=rewrite,
            regenerate_allowlist=regenerate,
        )
        for path_value in rewrite | regenerate:
            if mapping.map_path(path_value) is None:
                raise LayoutError(f"allowlisted path is unmapped: {path_value}")
        return mapping

    def map_path(self, old_path: str) -> str | None:
        exact = dict(self.exact_moves).get(old_path)
        if exact is not None:
            return exact

        for pattern, destination_prefix in self.pattern_moves:
            source_prefix = pattern[:-2]
            if old_path.startswith(source_prefix):
                return destination_prefix + old_path[len(source_prefix):]

        for pattern in self.identity_pins:
            if _pattern_matches(pattern, old_path):
                return old_path

        for source_prefix, destination_prefix in self.prefix_moves:
            if old_path.startswith(source_prefix):
                return destination_prefix + old_path[len(source_prefix):]
        return None

    def validate_allowlists_against_baseline(self, baseline_paths: Iterable[str]) -> None:
        baseline = set(baseline_paths)
        missing = (self.rewrite_allowlist | self.regenerate_allowlist) - baseline
        if missing:
            raise LayoutError(
                "allowlisted path not present in baseline: " f"{sorted(missing)!r}"
            )


@dataclass(frozen=True)
class _IndexEntry:
    path: str
    git_mode: str
    blob_oid: str


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LayoutError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LayoutError(f"{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise LayoutError(f"{field} contains duplicates")
    return tuple(value)


def _string_mapping(value: object, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise LayoutError(f"{field} must be an object mapping strings to strings")
    return tuple(value.items())


def _validate_relative_path(path_value: str, label: str) -> None:
    path = PurePosixPath(path_value)
    if (
        not path_value
        or path_value.startswith("/")
        or path.is_absolute()
        or path_value in {".", ".."}
        or ".." in path.parts
        or "" in path.parts
    ):
        raise LayoutError(f"unsafe {label}: {path_value!r}")


def _validate_destination(path_value: str) -> None:
    try:
        _validate_relative_path(path_value, "destination")
    except LayoutError as exc:
        raise LayoutError(f"unsafe destination: {path_value!r}") from exc


def _validate_source_pattern(pattern: str, label: str) -> None:
    if pattern.endswith("/**"):
        _validate_relative_path(pattern[:-3], label)
    elif "*" in pattern or "?" in pattern or "[" in pattern:
        raise LayoutError(f"unsupported {label} glob: {pattern!r}")
    else:
        _validate_relative_path(pattern, label)


def _pattern_matches(pattern: str, path_value: str) -> bool:
    if pattern.endswith("/**"):
        return path_value.startswith(pattern[:-2])
    return path_value == pattern


def _git(repo: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", "replace").strip()
        raise LayoutError(f"git {' '.join(args)} failed: {detail or exc}") from exc
    return completed.stdout


def _baseline_index(repo: Path) -> list[_IndexEntry]:
    inventory = _git(repo, "ls-files", "-s", "-z")
    entries: list[_IndexEntry] = []
    seen_paths: set[str] = set()
    for record in inventory.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = metadata.split(b" ", 2)
            path_value = os.fsdecode(raw_path)
            git_mode = raw_mode.decode("ascii")
            blob_oid = raw_oid.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise LayoutError("malformed git ls-files -s -z output") from exc
        if stage != "0":
            raise LayoutError(f"unmerged baseline index entry: {path_value}")
        if path_value in seen_paths:
            raise LayoutError(f"duplicate baseline index path: {path_value}")
        _validate_relative_path(path_value, "baseline path")
        seen_paths.add(path_value)
        entries.append(_IndexEntry(path_value, git_mode, blob_oid))
    return entries


def _blob_metadata(repo: Path, object_ids: Iterable[str]) -> Mapping[str, tuple[int, str]]:
    unique_ids = tuple(dict.fromkeys(object_ids))
    if not unique_ids:
        return {}
    try:
        process = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise LayoutError(f"cannot start git cat-file --batch: {exc}") from exc
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    result: dict[str, tuple[int, str]] = {}
    error = b""
    return_code = 0
    with process:
        try:
            for requested_oid in unique_ids:
                process.stdin.write(requested_oid.encode("ascii") + b"\n")
                process.stdin.flush()
                header = process.stdout.readline()
                parts = header.rstrip(b"\n").split(b" ")
                if len(parts) != 3:
                    raise LayoutError(
                        f"cannot read baseline object {requested_oid}: "
                        f"{header.decode('utf-8', 'replace').strip()}"
                    )
                actual_oid, object_type, raw_size = parts
                if object_type != b"blob":
                    raise LayoutError(
                        f"baseline object {requested_oid} is not a blob: "
                        f"{object_type.decode('ascii', 'replace')}"
                    )
                size = int(raw_size)
                digest = hashlib.sha256()
                remaining = size
                while remaining:
                    chunk = process.stdout.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise LayoutError(f"truncated baseline blob {requested_oid}")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if process.stdout.read(1) != b"\n":
                    raise LayoutError(f"malformed git cat-file record for {requested_oid}")
                result[requested_oid] = (size, digest.hexdigest())
                if actual_oid.decode("ascii") != requested_oid:
                    raise LayoutError(f"git returned the wrong object for {requested_oid}")
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise LayoutError(f"cannot read baseline blobs: {exc}") from exc
        finally:
            process.stdin.close()
        error = process.stderr.read()
        return_code = process.wait()
    if return_code:
        detail = error.decode("utf-8", "replace").strip()
        raise LayoutError(f"git cat-file --batch failed: {detail}")
    return result


def _validate_destinations(
    old_to_new: Sequence[tuple[str, str]], baseline_paths: set[str]
) -> None:
    destinations: dict[str, str] = {}
    casefolded: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for old_path, new_path in old_to_new:
        _validate_destination(new_path)
        if new_path in destinations:
            raise LayoutError(
                "duplicate destination "
                f"{new_path!r} for {destinations[new_path]!r} and {old_path!r}"
            )
        folded = new_path.casefold()
        if folded in casefolded and casefolded[folded] != new_path:
            raise LayoutError(
                f"case-fold destination collision: {casefolded[folded]!r}, {new_path!r}"
            )
        nfc = unicodedata.normalize("NFC", new_path)
        if nfc in normalized and normalized[nfc] != new_path:
            raise LayoutError(
                f"NFC destination collision: {normalized[nfc]!r}, {new_path!r}"
            )
        destinations[new_path] = old_path
        casefolded[folded] = new_path
        normalized[nfc] = new_path

    destination_paths = set(destinations)
    for path_value in destination_paths:
        parent = PurePosixPath(path_value).parent
        while str(parent) != ".":
            if parent.as_posix() in destination_paths:
                raise LayoutError(
                    f"destination overlap: {parent.as_posix()!r} contains {path_value!r}"
                )
            parent = parent.parent

    mapped = dict(old_to_new)
    for old_path, new_path in old_to_new:
        if (
            new_path != old_path
            and new_path in baseline_paths
            and mapped.get(new_path) != new_path
        ):
            raise LayoutError(
                f"destination {new_path!r} for {old_path!r} overlaps baseline source"
            )


def build_baseline_ledger(
    repo: Path,
    mapping: LayoutMapping,
    *,
    require_complete_allowlists: bool = False,
) -> list[LedgerRow]:
    """Build immutable rows from the Git index, never from a working-tree scan."""
    repo = repo.resolve()
    entries = _baseline_index(repo)
    baseline_paths = {entry.path for entry in entries}
    if require_complete_allowlists:
        mapping.validate_allowlists_against_baseline(baseline_paths)

    mapped_paths: list[tuple[str, str]] = []
    unmapped: list[str] = []
    for entry in entries:
        destination = mapping.map_path(entry.path)
        if destination is None:
            unmapped.append(entry.path)
        else:
            mapped_paths.append((entry.path, destination))
    if unmapped:
        raise LayoutError(f"unmapped baseline paths: {unmapped!r}")
    _validate_destinations(mapped_paths, baseline_paths)

    metadata = _blob_metadata(repo, (entry.blob_oid for entry in entries))
    destinations = dict(mapped_paths)
    rows: list[LedgerRow] = []
    for entry in entries:
        destination = destinations[entry.path]
        if entry.path in mapping.rewrite_allowlist:
            disposition = "approved-rewrite"
        elif entry.path in mapping.regenerate_allowlist:
            disposition = "proven-generated-regenerate"
        elif destination == entry.path:
            disposition = "same-byte-identical"
        else:
            disposition = "move-byte-identical"
        byte_length, sha256 = metadata[entry.blob_oid]
        rows.append(LedgerRow(
            old_path=entry.path,
            new_path=destination,
            git_mode=entry.git_mode,
            blob_oid=entry.blob_oid,
            byte_length=byte_length,
            sha256=sha256,
            disposition=disposition,
        ))
    return rows


def _working_tree_metadata(path: Path) -> tuple[str, int, str] | None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(file_stat.st_mode):
        target = os.readlink(os.fsencode(path))
        data = target if isinstance(target, bytes) else os.fsencode(target)
        return "120000", len(data), hashlib.sha256(data).hexdigest()
    if not stat.S_ISREG(file_stat.st_mode):
        return None
    git_mode = "100755" if file_stat.st_mode & 0o111 else "100644"
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_length += len(chunk)
            digest.update(chunk)
    return git_mode, byte_length, digest.hexdigest()


def verify_final_tree(
    repo: Path,
    ledger: Sequence[LedgerRow],
    rewrite_allowlist: set[str],
    regenerate_allowlist: set[str],
) -> VerificationReport:
    """Verify each immutable ledger row at its final path without scanning the tree."""
    overlap = rewrite_allowlist & regenerate_allowlist
    if overlap:
        raise LayoutError(f"verification allowlists overlap: {sorted(overlap)!r}")
    ledger_paths = {row.old_path for row in ledger}
    unknown = (rewrite_allowlist | regenerate_allowlist) - ledger_paths
    if unknown:
        raise LayoutError(f"verification allowlist path not in ledger: {sorted(unknown)!r}")
    for row in ledger:
        if row.disposition not in _ALLOWED_DISPOSITIONS:
            raise LayoutError(
                f"invalid disposition {row.disposition!r} for {row.old_path!r}"
            )
        _validate_destination(row.new_path)

    collisions = len(ledger) - len({row.new_path for row in ledger})
    migrated = 0
    lost = 0
    unexpected_rewrites = 0
    mode_changes = 0
    for row in ledger:
        metadata = _working_tree_metadata(repo / row.new_path)
        if metadata is None:
            lost += 1
            continue
        migrated += 1
        git_mode, byte_length, sha256 = metadata
        if git_mode != row.git_mode:
            mode_changes += 1
        changed = byte_length != row.byte_length or sha256 != row.sha256
        approved = (
            row.old_path in rewrite_allowlist
            or row.old_path in regenerate_allowlist
        )
        if changed and not approved:
            unexpected_rewrites += 1

    return VerificationReport(
        input_manifest_rows=len(ledger),
        migrated_legacy_rows=migrated,
        unmapped=0,
        collisions=collisions,
        lost_blobs=lost,
        unexpected_rewrites=unexpected_rewrites,
        mode_changes=mode_changes,
    )

def load_ledger(path: Path) -> list[LedgerRow]:
    """Load and validate an immutable repository-layout ledger."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise LayoutError(f"cannot load layout ledger {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise LayoutError("layout ledger must use schema version 1")
    base_commit = payload.get("base_commit")
    if (
        not isinstance(base_commit, str)
        or path.name != f"{base_commit}-repository-layout.json"
    ):
        raise LayoutError("layout ledger filename does not match its base commit")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or payload.get("row_count") != len(raw_rows):
        raise LayoutError("layout ledger row count does not match its rows")

    expected_fields = set(LedgerRow.__annotations__)
    rows: list[LedgerRow] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or set(raw_row) != expected_fields:
            raise LayoutError("layout ledger row has invalid fields")
        if (
            not isinstance(raw_row["old_path"], str)
            or not isinstance(raw_row["new_path"], str)
            or raw_row["git_mode"] not in {"100644", "100755", "120000"}
            or not isinstance(raw_row["blob_oid"], str)
            or len(raw_row["blob_oid"]) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in raw_row["blob_oid"])
            or type(raw_row["byte_length"]) is not int
            or raw_row["byte_length"] < 0
            or not isinstance(raw_row["sha256"], str)
            or len(raw_row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in raw_row["sha256"])
            or raw_row["disposition"] not in _ALLOWED_DISPOSITIONS
        ):
            raise LayoutError("layout ledger row has invalid values")
        _validate_relative_path(raw_row["old_path"], "ledger source")
        _validate_destination(raw_row["new_path"])
        rows.append(LedgerRow(**raw_row))

    old_paths = [row.old_path for row in rows]
    if len(old_paths) != len(set(old_paths)):
        raise LayoutError("layout ledger contains duplicate sources")
    _validate_destinations(
        [(row.old_path, row.new_path) for row in rows],
        set(old_paths),
    )
    return rows


def _repository_ledger_path(repo: Path) -> Path:
    matches = sorted(
        (repo / "operations" / "migrations").glob("*-repository-layout.json")
    )
    if len(matches) != 1:
        raise LayoutError(
            "expected exactly one immutable repository-layout ledger, "
            f"found {len(matches)}"
        )
    return matches[0]


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def apply_layout(
    repo: Path,
    ledger: Sequence[LedgerRow],
    *,
    dry_run: bool,
) -> ApplyReport:
    """Inspect or apply pending ledger moves without rewriting file contents."""
    repo = repo.resolve()
    pending: list[LedgerRow] = []
    completed = 0
    conflicts = 0
    missing = 0
    identity_rows = 0
    for row in ledger:
        source_exists = _path_exists(repo / row.old_path)
        destination_exists = _path_exists(repo / row.new_path)
        if row.old_path == row.new_path:
            identity_rows += 1
            if not source_exists:
                missing += 1
        elif source_exists and not destination_exists:
            pending.append(row)
        elif not source_exists and destination_exists:
            completed += 1
        elif source_exists and destination_exists:
            conflicts += 1
        else:
            missing += 1

    move_rows = len(ledger) - identity_rows
    report = ApplyReport(
        legacy_rows=len(ledger),
        move_rows=move_rows,
        identity_rows=identity_rows,
        pending_moves=len(pending),
        completed_moves=completed,
        conflicts=conflicts,
        missing=missing,
        moved=0,
        dry_run=dry_run,
    )
    if dry_run or not report.ok:
        return report

    ordered = sorted(
        pending,
        key=lambda row: (-len(PurePosixPath(row.old_path).parts), row.old_path),
    )
    for row in ordered:
        (repo / row.new_path).parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "mv", "--", row.old_path, row.new_path)
    return ApplyReport(
        legacy_rows=report.legacy_rows,
        move_rows=report.move_rows,
        identity_rows=report.identity_rows,
        pending_moves=report.pending_moves,
        completed_moves=report.completed_moves,
        conflicts=report.conflicts,
        missing=report.missing,
        moved=len(ordered),
        dry_run=False,
    )


def verify_moves(repo: Path, ledger: Sequence[LedgerRow]) -> VerificationReport:
    """Verify destination bytes plus authoritative Git index blobs and modes."""
    repo = repo.resolve()
    index = {entry.path: entry for entry in _baseline_index(repo)}
    collisions = len(ledger) - len({row.new_path for row in ledger})
    migrated = 0
    lost = 0
    unexpected_rewrites = 0
    mode_changes = 0
    for row in ledger:
        metadata = _working_tree_metadata(repo / row.new_path)
        index_entry = index.get(row.new_path)
        if metadata is None or index_entry is None:
            lost += 1
            continue
        migrated += 1
        if index_entry.git_mode != row.git_mode:
            mode_changes += 1
        _, byte_length, sha256 = metadata
        changed = (
            byte_length != row.byte_length
            or sha256 != row.sha256
            or index_entry.blob_oid != row.blob_oid
        )
        approved = row.disposition in {
            "approved-rewrite",
            "proven-generated-regenerate",
        }
        if changed and not approved:
            unexpected_rewrites += 1

    return VerificationReport(
        input_manifest_rows=len(ledger),
        migrated_legacy_rows=migrated,
        unmapped=0,
        collisions=collisions,
        lost_blobs=lost,
        unexpected_rewrites=unexpected_rewrites,
        mode_changes=mode_changes,
    )


def capture_layout(repo: Path, mapping_path: Path) -> Path:
    repo = repo.resolve()
    mapping = LayoutMapping.from_file(mapping_path)
    rows = build_baseline_ledger(
        repo,
        mapping,
        require_complete_allowlists=True,
    )
    base_commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    adapter_row = next(
        (row for row in rows if row.old_path == "fastllm_adapter.py"),
        None,
    )
    if adapter_row is None:
        raise LayoutError("fastllm_adapter.py is absent from the baseline ledger")
    ledger_path = (
        repo
        / "operations"
        / "migrations"
        / f"{base_commit}-repository-layout.json"
    )
    payload = {
        "schema_version": 1,
        "base_commit": base_commit,
        "baseline_inventory": "git ls-files -s -z",
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "fastllm_adapter_sha256": adapter_row.sha256,
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ledger_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as exc:
        raise LayoutError(f"refusing to overwrite immutable ledger: {ledger_path}") from exc
    return ledger_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture-layout",
        help="capture the immutable baseline migration ledger",
    )
    capture.add_argument("--repo", type=Path, default=Path("."))
    capture.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).with_name("layout-map.json"),
    )
    apply = subparsers.add_parser(
        "apply",
        help="inspect or apply immutable repository-layout moves",
    )
    apply.add_argument("--repo", type=Path, default=Path("."))
    apply.add_argument("--dry-run", action="store_true")
    verify = subparsers.add_parser(
        "verify-moves",
        help="verify migrated destinations against the immutable ledger",
    )
    verify.add_argument("--repo", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture-layout":
        try:
            ledger_path = capture_layout(args.repo, args.mapping)
        except LayoutError as exc:
            raise SystemExit(f"capture-layout: {exc}") from exc
        try:
            display_path = ledger_path.relative_to(args.repo.resolve())
        except ValueError:
            display_path = ledger_path
        print("unmapped=0 collisions=0")
        print(f"ledger={display_path.as_posix()}")
        return 0
    if args.command == "apply":
        try:
            ledger = load_ledger(_repository_ledger_path(args.repo.resolve()))
            report = apply_layout(args.repo, ledger, dry_run=args.dry_run)
        except LayoutError as exc:
            raise SystemExit(f"apply: {exc}") from exc
        print(
            f"legacy_rows={report.legacy_rows} move_rows={report.move_rows} "
            f"identity_rows={report.identity_rows}"
        )
        print(
            f"pending_moves={report.pending_moves} "
            f"completed_moves={report.completed_moves} "
            f"conflicts={report.conflicts} missing={report.missing}"
        )
        print(f"dry_run={int(report.dry_run)} moved={report.moved}")
        return 0 if report.ok else 1
    if args.command == "verify-moves":
        try:
            ledger = load_ledger(_repository_ledger_path(args.repo.resolve()))
            report = verify_moves(args.repo, ledger)
        except LayoutError as exc:
            raise SystemExit(f"verify-moves: {exc}") from exc
        print(
            f"legacy_rows={report.input_manifest_rows} "
            f"joined={report.migrated_legacy_rows}"
        )
        print(
            f"lost_blobs={report.lost_blobs} "
            f"unexpected_rewrites={report.unexpected_rewrites} "
            f"mode_changes={report.mode_changes}"
        )
        return 0 if report.ok else 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
