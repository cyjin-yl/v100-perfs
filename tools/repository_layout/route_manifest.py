"""Freeze and validate the pre-migration documentation route surface."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
from urllib.parse import quote, unquote


_EXPECTED_CURATED_ROUTES = 9
_EXPECTED_READ_ROUTES = 413
_ALLOWED_EXTENSIONS = frozenset({
    ".astro",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".css",
    ".csv",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".ini",
    ".jinja",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".patch",
    ".py",
    ".sh",
    ".sse",
    ".toml",
    ".ts",
    ".tsv",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
})
_SKIPPED_DIRECTORIES = frozenset({
    ".git",
    ".astro",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "1Cat-vLLM",
    "1Cat-vLLM-1.2.0",
    "__pycache__",
    "build",
    "dist",
    "exllamav3",
    "ik_llama.cpp",
    "llama.cpp",
    "llama.cpp-turboquant",
    "logs",
    "models",
    "node_modules",
    "runtime",
    "runtime-artifacts",
    "server",
})
_MAX_LEGACY_BYTES = 1024 * 1024
_CURATED_CANONICAL_ROUTES = {
    "getting-started": "/docs/getting-started/overview",
    "architecture": "/docs/concepts/architecture/system-overview",
    "toolcall": "/docs/concepts/tool-calling/constraint-architecture",
    "performance": "/docs/performance/qwen38/sweep-matrix",
    "ops": "/archive/handoffs/2026-08-18-qwen38",
    "experience": "/archive/chronicles/v100-fastllm-experience",
    "optane": "/docs/research/storage/optane-l3",
    "prefill-plan": "/docs/research/performance/short-prefill-plan",
    "upstream": "/archive/assessments/2026-08-22-upstream-cherry-pick",
}


class RouteManifestError(ValueError):
    """The frozen baseline or route graph violates a route invariant."""


@dataclass(frozen=True)
class LegacyRoute:
    old_url: str
    encoded_url: str
    trailing_slash_url: str
    canonical_target: str


@dataclass(frozen=True)
class _BaselinePath:
    old_path: str
    new_path: str
    byte_length: int


@dataclass(frozen=True)
class _Capture:
    base_commit: str
    base_path: str
    layout_ledger: Path
    routes: tuple[LegacyRoute, ...]


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RouteManifestError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteManifestError(f"cannot load {path}: {exc}") from exc


def _layout_ledger(repo: Path) -> tuple[Path, str, list[_BaselinePath]]:
    candidates = sorted((repo / "operations" / "migrations").glob(
        "*-repository-layout.json"
    ))
    if len(candidates) != 1:
        raise RouteManifestError(
            "expected exactly one immutable repository-layout ledger, "
            f"found {len(candidates)}"
        )
    path = candidates[0]
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RouteManifestError(f"layout ledger must contain an object: {path}")
    base_commit = payload.get("base_commit")
    rows_data = payload.get("rows")
    if (
        not isinstance(base_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", base_commit)
        or path.name != f"{base_commit}-repository-layout.json"
    ):
        raise RouteManifestError("layout ledger base commit does not match its filename")
    if not isinstance(rows_data, list):
        raise RouteManifestError("layout ledger rows must be a list")

    rows: list[_BaselinePath] = []
    seen: set[str] = set()
    for index, value in enumerate(rows_data):
        if not isinstance(value, dict):
            raise RouteManifestError(f"layout row {index} must be an object")
        old_path = value.get("old_path")
        new_path = value.get("new_path")
        byte_length = value.get("byte_length")
        if (
            not isinstance(old_path, str)
            or not isinstance(new_path, str)
            or not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 0
        ):
            raise RouteManifestError(f"layout row {index} has invalid route fields")
        if old_path in seen:
            raise RouteManifestError(f"duplicate baseline path: {old_path!r}")
        seen.add(old_path)
        rows.append(_BaselinePath(old_path, new_path, byte_length))
    if payload.get("row_count") != len(rows):
        raise RouteManifestError("layout ledger row_count does not match rows")
    return path, base_commit, rows


def _git_text(repo: Path, base_commit: str, path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{base_commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RouteManifestError(
            f"cannot read baseline route definition {path!r} at {base_commit}"
        ) from exc
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RouteManifestError(
            f"baseline route definition is not UTF-8: {path!r}"
        ) from exc


def _base_path(repo: Path, base_commit: str) -> str:
    config = _git_text(repo, base_commit, "site/astro.config.mjs")
    matches = re.findall(r"\bbase\s*:\s*['\"]([^'\"]+)['\"]", config)
    if len(matches) != 1:
        raise RouteManifestError("baseline Astro config must define exactly one base path")
    base = matches[0].rstrip("/")
    if not base.startswith("/") or base == "":
        raise RouteManifestError(f"invalid baseline Astro base path: {matches[0]!r}")
    return base


def _defined_read_paths(repo: Path, base_commit: str) -> tuple[str, ...]:
    raw = _git_text(repo, base_commit, "site/src/data/repo-index.json")
    try:
        values = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise RouteManifestError(f"invalid baseline repo-index.json: {exc}") from exc
    if not isinstance(values, list):
        raise RouteManifestError("baseline repo-index.json must contain a list")
    paths: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise RouteManifestError(f"baseline repo-index row {index} has no path")
        paths.append(value["path"])
    if len(paths) != len(set(paths)):
        raise RouteManifestError("baseline repo-index contains duplicate paths")
    return tuple(paths)


def _defined_curated_slugs(repo: Path, base_commit: str) -> set[str]:
    source = _git_text(repo, base_commit, "site/src/pages/docs/[...slug].astro")
    return set(re.findall(r"\{\s*slug:\s*['\"]([^'\"]+)['\"]", source))


def _is_sensitive_file(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith(".env")
        or re.search(
            r"(?:^|[._-])(?:credentials?|secrets?)(?:[._-]|$)", lower
        ) is not None
        or re.search(r"\.(?:key|p12|pem|pfx)$", lower) is not None
    )


def _is_legacy_read_path(row: _BaselinePath) -> bool:
    path = PurePosixPath(row.old_path)
    if row.old_path == "site/src/data/repo-index.json":
        return False
    if any(
        part in _SKIPPED_DIRECTORIES or part.startswith(".venv")
        for part in path.parts[:-1]
    ):
        return False
    return (
        path.suffix.lower() in _ALLOWED_EXTENSIONS
        and not _is_sensitive_file(path.name)
        and row.byte_length <= _MAX_LEGACY_BYTES
    )


def _route_without_markdown_suffix(path: str) -> str:
    route = path
    suffix = PurePosixPath(route).suffix.lower()
    if suffix in {".md", ".txt"}:
        route = route[:-len(suffix)]
    if route == "index":
        return ""
    if route.endswith("/index"):
        return route[:-len("index")]
    return route


def _canonical_route(new_path: str) -> str:
    if new_path.startswith("docs/archive/"):
        relative = _route_without_markdown_suffix(new_path[len("docs/archive/"):])
        return f"/archive/{relative}"
    if new_path.startswith("docs/"):
        relative = _route_without_markdown_suffix(new_path[len("docs/"):])
        return f"/docs/{relative}"
    if new_path.startswith("evidence/"):
        return f"/evidence/{new_path[len('evidence/'):]}"
    return f"/repository/{new_path}"


def _absolute_url(base_path: str, route: str, *, trailing_slash: bool) -> str:
    raw = f"{base_path}{route}"
    encoded = quote(raw, safe="/")
    if trailing_slash and not encoded.endswith("/"):
        return f"{encoded}/"
    return encoded


def _legacy_route(base_path: str, route: str, canonical_route: str) -> LegacyRoute:
    old_url = f"{base_path}{route}"
    encoded_url = quote(old_url, safe="/")
    trailing_slash_url = encoded_url if encoded_url.endswith("/") else f"{encoded_url}/"
    canonical_target = _absolute_url(base_path, canonical_route, trailing_slash=True)
    return LegacyRoute(old_url, encoded_url, trailing_slash_url, canonical_target)


def _artifact_path(dist: Path, route: str) -> Path:
    relative = unquote(route).strip("/")
    return dist / relative / "index.html" if relative else dist / "index.html"


def _validate_artifacts(
    dist: Path,
    defined_read_paths: Sequence[str],
    curated_slugs: Sequence[str],
) -> None:
    expected_routes = ["/", "/repo"]
    expected_routes.extend(f"/docs/{slug}" for slug in curated_slugs)
    expected_routes.extend(f"/read/{path}" for path in defined_read_paths)
    missing = [route for route in expected_routes if not _artifact_path(dist, route).is_file()]
    if missing:
        raise RouteManifestError(
            f"baseline dist is missing {len(missing)} defined route artifacts: {missing[:5]!r}"
        )


def _validate_route_uniqueness(routes: Sequence[LegacyRoute]) -> None:
    seen: dict[str, str] = {}
    for route in routes:
        for form in {route.old_url, route.encoded_url, route.trailing_slash_url}:
            normalized = _normalized_url(form)
            owner = seen.get(normalized)
            if owner is not None and owner != route.old_url:
                raise RouteManifestError(
                    f"duplicate legacy URL {form!r} for {owner!r} and {route.old_url!r}"
                )
            seen[normalized] = route.old_url


def _capture_legacy_routes(repo: Path, dist: Path) -> _Capture:
    repo = repo.resolve()
    dist = dist.resolve() if dist.is_absolute() else (repo / dist).resolve()
    layout_path, base_commit, baseline_rows = _layout_ledger(repo)
    base_path = _base_path(repo, base_commit)
    defined_read_paths = _defined_read_paths(repo, base_commit)
    eligible_rows = [row for row in baseline_rows if _is_legacy_read_path(row)]
    eligible_paths = {row.old_path for row in eligible_rows}
    undefined = sorted(set(defined_read_paths) - eligible_paths)
    if undefined:
        raise RouteManifestError(
            f"baseline repo-index paths are absent or ineligible: {undefined[:5]!r}"
        )
    if len(eligible_rows) != _EXPECTED_READ_ROUTES:
        raise RouteManifestError(
            f"expected {_EXPECTED_READ_ROUTES} baseline read routes, "
            f"found {len(eligible_rows)}"
        )

    curated_slugs = _defined_curated_slugs(repo, base_commit)
    approved_slugs = set(_CURATED_CANONICAL_ROUTES)
    if curated_slugs != approved_slugs or len(curated_slugs) != _EXPECTED_CURATED_ROUTES:
        raise RouteManifestError(
            "baseline curated routes do not match the approved canonical mapping"
        )
    _validate_artifacts(dist, defined_read_paths, sorted(curated_slugs))

    routes = [
        _legacy_route(base_path, "/", "/"),
        _legacy_route(base_path, "/repo", "/repo"),
    ]
    routes.extend(
        _legacy_route(
            base_path,
            f"/docs/{slug}",
            _CURATED_CANONICAL_ROUTES[slug],
        )
        for slug in _CURATED_CANONICAL_ROUTES
    )
    routes.extend(
        _legacy_route(
            base_path,
            f"/read/{row.old_path}",
            _canonical_route(row.new_path),
        )
        for row in eligible_rows
    )
    expected_total = 2 + _EXPECTED_CURATED_ROUTES + _EXPECTED_READ_ROUTES
    if len(routes) != expected_total:
        raise RouteManifestError(
            f"expected {expected_total} legacy routes, found {len(routes)}"
        )
    _validate_route_uniqueness(routes)
    return _Capture(
        base_commit=base_commit,
        base_path=base_path,
        layout_ledger=layout_path,
        routes=tuple(routes),
    )


def capture_legacy_routes(repo: Path, dist: Path) -> list[LegacyRoute]:
    """Capture the immutable 424-route baseline and validate built definitions."""
    return list(_capture_legacy_routes(repo, dist).routes)


def _normalized_url(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"route must be an absolute URL path: {value!r}")
    decoded = unquote(value)
    return decoded if decoded == "/" else decoded.rstrip("/")


def validate_alias_targets(aliases: Mapping[str, str]) -> None:
    """Reject cycles and every alias whose target is another alias."""
    normalized: dict[str, str] = {}
    original: dict[str, str] = {}
    for alias, target in aliases.items():
        source_key = _normalized_url(alias)
        target_key = _normalized_url(target)
        if source_key in normalized:
            raise ValueError(f"duplicate normalized alias: {alias!r}")
        normalized[source_key] = target_key
        original[source_key] = alias

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(source: str, chain: list[str]) -> None:
        if source in visiting:
            start = chain.index(source)
            cycle = chain[start:] + [source]
            raise ValueError(f"alias cycle: {' -> '.join(cycle)}")
        if source in visited:
            return
        visiting.add(source)
        target = normalized[source]
        if target in normalized:
            visit(target, chain + [target])
        visiting.remove(source)
        visited.add(source)

    for source in normalized:
        visit(source, [source])
    for source, target in normalized.items():
        if target in normalized:
            raise ValueError(
                f"alias target is another alias: {original[source]!r} -> "
                f"{original[target]!r}"
            )


def capture_route_ledger(repo: Path, dist: Path) -> Path:
    """Write the immutable route ledger using the layout ledger's base commit."""
    repo = repo.resolve()
    capture = _capture_legacy_routes(repo, dist)
    try:
        layout_display = capture.layout_ledger.relative_to(repo).as_posix()
    except ValueError:
        layout_display = capture.layout_ledger.as_posix()
    ledger_path = (
        repo
        / "operations"
        / "migrations"
        / f"{capture.base_commit}-legacy-routes.json"
    )
    payload = {
        "schema_version": 1,
        "base_commit": capture.base_commit,
        "base_path": capture.base_path,
        "source_layout_ledger": layout_display,
        "baseline_route_definitions": [
            "site/astro.config.mjs",
            "site/src/pages/docs/[...slug].astro",
            "site/src/data/repo-index.json",
        ],
        "curated_route_count": _EXPECTED_CURATED_ROUTES,
        "read_route_count": _EXPECTED_READ_ROUTES,
        "row_count": len(capture.routes),
        "rows": [asdict(route) for route in capture.routes],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ledger_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as exc:
        raise RouteManifestError(
            f"refusing to overwrite immutable route ledger: {ledger_path}"
        ) from exc
    return ledger_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="freeze the legacy route ledger")
    capture.add_argument("--repo", type=Path, default=Path("."))
    capture.add_argument("--dist", type=Path, default=Path("site/dist"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture":
        try:
            ledger_path = capture_route_ledger(args.repo, args.dist)
        except RouteManifestError as exc:
            raise SystemExit(f"capture: {exc}") from exc
        print("legacy_routes=424 missing_artifacts=0 duplicate_urls=0")
        try:
            display_path = ledger_path.relative_to(args.repo.resolve())
        except ValueError:
            display_path = ledger_path
        print(f"ledger={display_path.as_posix()}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
