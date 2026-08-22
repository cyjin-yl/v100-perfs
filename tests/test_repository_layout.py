import gc
from collections import Counter
import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from tools.repository_layout.layout_manifest import (
    LayoutError,
    LayoutMapping,
    build_baseline_ledger,
    load_ledger,
    main as layout_main,
    rewrite_repository_reference,
    scan_internal_references,
    verify_final_tree,
)
from tools.repository_layout.route_manifest import (
    LegacyRoute,
    validate_alias_targets,
)


PRODUCTION_MAPPING = Path("tools/repository_layout/layout-map.json")
PRODUCTION_ROUTE_LEDGER = Path(
    "operations/migrations/"
    "67ef398f9da504e020db1349956e2990f2876171-legacy-routes.json"
)


def git_tracked_paths(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        path.decode("utf-8", "surrogateescape")
        for path in completed.stdout.split(b"\0")
        if path
    ]


def load_content_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(item, dict) for item in payload
    ):
        raise ValueError("content manifest must be a JSON array of objects")
    return payload



class LayoutManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def add_files(self, files):
        for rel, data in files.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True)

    def mapping(self, **overrides):
        data = {
            "identity_pins": [],
            "prefix_moves": {},
            "exact_moves": {},
            "pattern_moves": {},
            "rewrite_allowlist": [],
            "regenerate_allowlist": [],
        }
        data.update(overrides)
        path = self.repo / "layout-map.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return LayoutMapping.from_file(path)

    def write_ledger(self, rows):
        ledger_path = (
            self.repo
            / "operations"
            / "migrations"
            / f"{'a' * 40}-repository-layout.json"
        )
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps({
                "schema_version": 1,
                "base_commit": "a" * 40,
                "row_count": len(rows),
                "rows": [asdict(row) for row in rows],
            }),
            encoding="utf-8",
        )
        return ledger_path

    def run_layout_main(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = layout_main([*args, "--repo", str(self.repo)])
        return return_code, output.getvalue()

    def move_rows(self, rows):
        for row in rows:
            source = self.repo / row.old_path
            target = self.repo / row.new_path
            if source != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)

    def test_mapping_is_total_and_has_no_destination_collisions(self):
        self.add_files({
            "benchmarks/a.log": b"a",
            "results/r.json": b"{}",
            "fastllm_adapter.py": b"pinned",
        })
        mapping = LayoutMapping.from_file(PRODUCTION_MAPPING)
        rows = build_baseline_ledger(self.repo, mapping)
        self.assertEqual({row.new_path for row in rows}, {
            "evidence/benchmarks/a.log",
            "evidence/results/r.json",
            "fastllm_adapter.py",
        })

    def test_inventory_comes_from_git_index_and_preserves_blob_metadata(self):
        self.add_files({"tracked": b"indexed", "bin/run": b"#!/bin/sh\n"})
        executable = self.repo / "bin/run"
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        subprocess.run(["git", "-C", str(self.repo), "add", "bin/run"], check=True)
        (self.repo / "tracked").write_bytes(b"dirty working tree")
        (self.repo / "untracked").write_bytes(b"ignore me")

        rows = build_baseline_ledger(
            self.repo,
            self.mapping(identity_pins=["tracked", "bin/**"]),
        )

        self.assertEqual([row.old_path for row in rows], ["bin/run", "tracked"])
        by_path = {row.old_path: row for row in rows}
        self.assertEqual(by_path["tracked"].byte_length, 7)
        self.assertEqual(
            by_path["tracked"].sha256,
            hashlib.sha256(b"indexed").hexdigest(),
        )
        self.assertEqual(by_path["bin/run"].git_mode, "100755")
        self.assertEqual(by_path["tracked"].disposition, "same-byte-identical")

    def test_blob_reader_closes_all_subprocess_pipes(self):
        self.add_files({"tracked": b"indexed"})
        mapping = self.mapping(identity_pins=["tracked"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            build_baseline_ledger(self.repo, mapping)
            gc.collect()

        resource_warnings = [
            warning for warning in caught
            if issubclass(warning.category, ResourceWarning)
        ]
        self.assertEqual(resource_warnings, [])

    def test_rejects_unmapped_baseline_path(self):
        self.add_files({"unknown": b"x"})
        with self.assertRaisesRegex(LayoutError, "unmapped"):
            build_baseline_ledger(self.repo, self.mapping())

    def test_rejects_duplicate_casefold_and_nfc_destination_collisions(self):
        cases = [
            ({"a": "dest", "b": "dest"}, "duplicate"),
            ({"a": "Dest", "b": "dest"}, "case-fold"),
            ({"a": "caf\u00e9", "b": "cafe\u0301"}, "NFC"),
        ]
        for exact_moves, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temp:
                    repo = Path(temp)
                    subprocess.run(["git", "init", "-q", str(repo)], check=True)
                    for rel in exact_moves:
                        (repo / rel).write_bytes(rel.encode())
                    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
                    map_path = repo / "map.json"
                    map_path.write_text(json.dumps({
                        "identity_pins": [],
                        "prefix_moves": {},
                        "exact_moves": exact_moves,
                        "pattern_moves": {},
                        "rewrite_allowlist": [],
                        "regenerate_allowlist": [],
                    }), encoding="utf-8")
                    with self.assertRaisesRegex(LayoutError, message):
                        build_baseline_ledger(repo, LayoutMapping.from_file(map_path))

    def test_rejects_unsafe_destination_and_destination_overlap(self):
        self.add_files({"a": b"a", "b": b"b"})
        with self.assertRaisesRegex(LayoutError, "unsafe destination"):
            build_baseline_ledger(
                self.repo,
                self.mapping(exact_moves={"a": "../escape", "b": "b"}),
            )
        with self.assertRaisesRegex(LayoutError, "overlaps baseline source"):
            build_baseline_ledger(
                self.repo,
                self.mapping(exact_moves={"a": "b", "b": "elsewhere"}),
            )

    def test_allowlists_are_disjoint_present_and_set_dispositions(self):
        self.add_files({"rewrite": b"r", "generated": b"g", "plain": b"p"})
        mapping = self.mapping(
            identity_pins=["rewrite", "generated", "plain"],
            rewrite_allowlist=["rewrite"],
            regenerate_allowlist=["generated"],
        )
        mapping.validate_allowlists_against_baseline(
            {"rewrite", "generated", "plain"})
        rows = build_baseline_ledger(self.repo, mapping)
        dispositions = {row.old_path: row.disposition for row in rows}
        self.assertEqual(dispositions, {
            "generated": "proven-generated-regenerate",
            "plain": "same-byte-identical",
            "rewrite": "approved-rewrite",
        })

        with self.assertRaisesRegex(LayoutError, "not present in baseline"):
            mapping.validate_allowlists_against_baseline({"rewrite", "plain"})

    def test_loader_rejects_overlapping_allowlists(self):
        with self.assertRaisesRegex(LayoutError, "both rewrite and regenerate"):
            self.mapping(
                identity_pins=["x"],
                rewrite_allowlist=["x"],
                regenerate_allowlist=["x"],
            )

    def test_repository_reference_rewriter_preserves_suffixes(self):
        self.assertEqual(
            rewrite_repository_reference(
                "./docs/qwen38-sampling.md?plain=1#template"
            ),
            "./docs/reference/models/qwen38/sampling-and-template.md"
            "?plain=1#template",
        )
        self.assertEqual(
            rewrite_repository_reference("scripts/start_proxy.sh"),
            "tools/scripts/start_proxy.sh",
        )
        self.assertEqual(
            rewrite_repository_reference("docs/current.md"),
            "docs/current.md",
        )

    def test_reference_scanner_separates_active_from_historical_old_paths(self):
        self.add_files({
            "current.md": b"[legacy](./legacy/a.md)\n",
            "archive.md": b"Recorded legacy/a.md for history.\n",
            "legacy/a.md": b"target\n",
        })
        rows = build_baseline_ledger(
            self.repo,
            self.mapping(
                identity_pins=["current.md", "archive.md"],
                exact_moves={"legacy/a.md": "current/a.md"},
                rewrite_allowlist=["current.md"],
            ),
        )
        self.move_rows(rows)

        report = scan_internal_references(self.repo, rows)

        self.assertEqual(
            [
                (
                    finding.source_path,
                    finding.line_number,
                    finding.reference,
                    finding.resolved_path,
                )
                for finding in report.stale_old_paths
            ],
            [("current.md", 1, "legacy/a.md", "current/a.md")],
        )
        self.assertEqual(
            [
                (
                    finding.source_path,
                    finding.line_number,
                    finding.reference,
                    finding.resolved_path,
                )
                for finding in report.broken_internal_links
            ],
            [("current.md", 1, "./legacy/a.md", "legacy/a.md")],
        )
        self.assertEqual(
            {
                classification: len(findings)
                for classification, findings
                in report.historical_old_paths.items()
            },
            {"identity-pinned": 1},
        )
        (self.repo / "current/a.md").unlink()
        missing_target = scan_internal_references(self.repo, rows)
        self.assertIn(
            ("archive.md", 1, "legacy/a.md", "current/a.md"),
            {
                (
                    finding.source_path,
                    finding.line_number,
                    finding.reference,
                    finding.resolved_path,
                )
                for finding in missing_target.broken_internal_links
            },
        )
        (self.repo / "current/a.md").write_text("target\n", encoding="utf-8")

        (self.repo / "current.md").write_text(
            "[current](current/a.md)\n",
            encoding="utf-8",
        )
        repaired = scan_internal_references(self.repo, rows)
        self.assertEqual(repaired.stale_old_paths, [])
        self.assertEqual(repaired.broken_internal_links, [])
        self.assertEqual(
            {
                finding.resolved_path
                for finding in repaired.resolved_internal_links
            },
            {"current/a.md"},
        )


    def test_verifier_rejects_rewritten_byte_identical_blob(self):
        self.add_files({"benchmarks/a.log": b"a"})
        rows = build_baseline_ledger(
            self.repo,
            self.mapping(prefix_moves={"benchmarks/": "evidence/benchmarks/"}),
        )
        self.move_rows(rows)
        (self.repo / rows[0].new_path).write_bytes(b"changed")

        report = verify_final_tree(self.repo, rows, set(), set())

        self.assertEqual(report.lost_blobs, 0)
        self.assertEqual(report.unexpected_rewrites, 1)
        self.assertEqual(report.mode_changes, 0)

    def test_verifier_allows_approved_changes_but_reports_missing_and_mode_changes(self):
        self.add_files({"rewrite": b"r", "missing": b"m", "mode": b"x"})
        rows = build_baseline_ledger(
            self.repo,
            self.mapping(identity_pins=["rewrite", "missing", "mode"]),
        )
        (self.repo / "rewrite").write_bytes(b"approved")
        (self.repo / "missing").unlink()
        (self.repo / "mode").chmod((self.repo / "mode").stat().st_mode | stat.S_IXUSR)

        report = verify_final_tree(self.repo, rows, {"rewrite"}, set())

        self.assertEqual(report.input_manifest_rows, 3)
        self.assertEqual(report.migrated_legacy_rows, 2)
        self.assertEqual(report.lost_blobs, 1)
        self.assertEqual(report.unexpected_rewrites, 0)
        self.assertEqual(report.mode_changes, 1)

    def test_apply_dry_run_is_read_only_and_reports_ledger_counts(self):
        self.add_files({"old/nested/data": b"data", "keep": b"keep"})
        rows = build_baseline_ledger(
            self.repo,
            self.mapping(
                identity_pins=["keep"],
                prefix_moves={"old/": "new/"},
            ),
        )
        self.write_ledger(rows)
        status_before = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain=v1", "-z"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout

        return_code, output = self.run_layout_main("apply", "--dry-run")

        status_after = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain=v1", "-z"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        self.assertEqual(return_code, 0)
        self.assertEqual(status_after, status_before)
        self.assertTrue((self.repo / "old/nested/data").exists())
        self.assertFalse((self.repo / "new/nested/data").exists())
        self.assertEqual(output, (
            "legacy_rows=2 move_rows=1 identity_rows=1\n"
            "pending_moves=1 completed_moves=0 conflicts=0 missing=0\n"
            "dry_run=1 moved=0\n"
        ))

    def test_apply_and_verify_use_git_index_modes_and_working_tree_bytes(self):
        self.add_files({"old/data": b"data", "keep": b"keep"})
        rows = build_baseline_ledger(
            self.repo,
            self.mapping(
                identity_pins=["keep"],
                prefix_moves={"old/": "new/"},
            ),
        )
        self.write_ledger(rows)

        apply_code, apply_output = self.run_layout_main("apply")

        self.assertEqual(apply_code, 0)
        self.assertFalse((self.repo / "old/data").exists())
        self.assertEqual((self.repo / "new/data").read_bytes(), b"data")
        self.assertEqual(apply_output, (
            "legacy_rows=2 move_rows=1 identity_rows=1\n"
            "pending_moves=1 completed_moves=0 conflicts=0 missing=0\n"
            "dry_run=0 moved=1\n"
        ))

        subprocess.run(
            ["git", "-C", str(self.repo), "config", "core.filemode", "false"],
            check=True,
        )
        destination = self.repo / "new/data"
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR)

        verify_code, verify_output = self.run_layout_main("verify-moves")

        self.assertEqual(verify_code, 0)
        self.assertEqual(verify_output, (
            "legacy_rows=2 joined=2\n"
            "lost_blobs=0 unexpected_rewrites=0 mode_changes=0\n"
        ))

        destination.write_bytes(b"changed")
        changed_code, changed_output = self.run_layout_main("verify-moves")
        self.assertEqual(changed_code, 1)
        self.assertIn("unexpected_rewrites=1", changed_output)

    def test_every_tracked_internal_reference_resolves_after_mapping(self):
        ledger_path = next(
            Path("operations/migrations").glob("*-repository-layout.json")
        )
        report = scan_internal_references(
            Path.cwd(),
            load_ledger(ledger_path),
        )
        self.assertEqual(report.broken_internal_links, [])
        self.assertEqual(report.stale_old_paths, [])

        manifest = load_content_manifest(
            Path("site/src/data/content-manifest.json")
        )
        published_docs = {
            str(item["source_path"]): (
                str(item["route_kind"]),
                str(item["canonical_route"]),
            )
            for item in manifest
            if str(item["source_path"]).startswith("docs/")
        }
        self.assertEqual(published_docs, {
            "docs/archive/assessments/2026-08-22-upstream-cherry-pick.md": (
                "archive",
                "/archive/assessments/2026-08-22-upstream-cherry-pick",
            ),
            "docs/archive/chronicles/v100-fastllm-experience.md": (
                "archive",
                "/archive/chronicles/v100-fastllm-experience",
            ),
            "docs/archive/design-decisions/plans/2026-08-22-full-repository-docs-migration.md": (
                "archive",
                "/archive/design-decisions/plans/2026-08-22-full-repository-docs-migration",
            ),
            "docs/archive/design-decisions/plans/2026-08-22-reasoning-control-compatibility.md": (
                "archive",
                "/archive/design-decisions/plans/2026-08-22-reasoning-control-compatibility",
            ),
            "docs/archive/design-decisions/specs/2026-08-22-full-repository-docs-and-thinking-controls-design.md": (
                "archive",
                "/archive/design-decisions/specs/2026-08-22-full-repository-docs-and-thinking-controls-design",
            ),
            "docs/archive/handoffs/2026-07-28-fastllm.md": (
                "archive",
                "/archive/handoffs/2026-07-28-fastllm",
            ),
            "docs/archive/handoffs/2026-08-09-deepseek-v4.md": (
                "archive",
                "/archive/handoffs/2026-08-09-deepseek-v4",
            ),
            "docs/archive/handoffs/2026-08-18-qwen38.md": (
                "archive",
                "/archive/handoffs/2026-08-18-qwen38",
            ),
            "docs/archive/handoffs/2026-08-20-production.md": (
                "archive",
                "/archive/handoffs/2026-08-20-production",
            ),
            "docs/archive/qwen35/design/q4km-sm70.md": (
                "archive",
                "/archive/qwen35/design/q4km-sm70",
            ),
            "docs/archive/qwen36/performance/fastllm.md": (
                "archive",
                "/archive/qwen36/performance/fastllm",
            ),
            "docs/archive/qwen36/performance/vllm-and-multi-engine.md": (
                "archive",
                "/archive/qwen36/performance/vllm-and-multi-engine",
            ),
            "docs/concepts/tool-calling/constraint-architecture.md": (
                "docs",
                "/docs/concepts/tool-calling/constraint-architecture",
            ),
            "docs/operations/troubleshooting/silent-hang.md": (
                "docs",
                "/docs/operations/troubleshooting/silent-hang",
            ),
            "docs/operations/troubleshooting/tool-call-copy-fidelity.md": (
                "docs",
                "/docs/operations/troubleshooting/tool-call-copy-fidelity",
            ),
            "docs/performance/qwen38/sweep-matrix.md": (
                "docs",
                "/docs/performance/qwen38/sweep-matrix",
            ),
            "docs/performance/qwen38/toolcall-benchmark-2026-08-22.md": (
                "docs",
                "/docs/performance/qwen38/toolcall-benchmark-2026-08-22",
            ),
            "docs/reference/models/qwen38/sampling-and-template.md": (
                "docs",
                "/docs/reference/models/qwen38/sampling-and-template",
            ),
            "docs/research/performance/prefill-exact-port.md": (
                "docs",
                "/docs/research/performance/prefill-exact-port",
            ),
            "docs/research/performance/short-prefill-plan.md": (
                "docs",
                "/docs/research/performance/short-prefill-plan",
            ),
            "docs/research/storage/optane-l3.md": (
                "docs",
                "/docs/research/storage/optane-l3",
            ),
        })
        for path_value in published_docs:
            with self.subTest(path=path_value):
                self.assertTrue(Path(path_value).is_file())


    def test_production_mapping_contains_the_complete_explicit_docs_rules(self):
        mapping = LayoutMapping.from_file(PRODUCTION_MAPPING)
        expected = {
            "docs/EXPERIENCE.md": "docs/archive/chronicles/v100-fastllm-experience.md",
            "docs/qwen38-sampling.md": "docs/reference/models/qwen38/sampling-and-template.md",
            "docs/analysis/optane-l3-feasibility.md": "docs/research/storage/optane-l3.md",
            "docs/analysis/prefill-exact-port-investigation.md": "docs/research/performance/prefill-exact-port.md",
            "docs/analysis/shortprefill-optimization-plan.md": "docs/research/performance/short-prefill-plan.md",
            "docs/analysis/silent-hang-diagnosis.md": "docs/operations/troubleshooting/silent-hang.md",
            "docs/analysis/upstream-cherry-pick-assessment.md": "docs/archive/assessments/2026-08-22-upstream-cherry-pick.md",
            "docs/benchmarks/fastllm-qwen36-legacy.md": "docs/archive/qwen36/performance/fastllm.md",
            "docs/benchmarks/vllm-performance.md": "docs/archive/qwen36/performance/vllm-and-multi-engine.md",
            "docs/legacy/qwen35-q4km-sm70-design.md": "docs/archive/qwen35/design/q4km-sm70.md",
            "docs/ops/handoff-fastllm.md": "docs/archive/handoffs/2026-07-28-fastllm.md",
            "docs/ops/handoff-deepseek-v4.md": "docs/archive/handoffs/2026-08-09-deepseek-v4.md",
            "docs/ops/handoff-qwen38.md": "docs/archive/handoffs/2026-08-18-qwen38.md",
            "docs/toolcall/constraint-architecture.md": "docs/concepts/tool-calling/constraint-architecture.md",
            "docs/toolcall/copy-fidelity.md": "docs/operations/troubleshooting/tool-call-copy-fidelity.md",
            "handoff/HANDOFF-2026-08-20.md": "docs/archive/handoffs/2026-08-20-production.md",
            "results/q38-sweep/MATRIX.md": "docs/performance/qwen38/sweep-matrix.md",
            "results/q38-sweep/raw/toolcall_bench_2026-08-22/SUMMARY.md": "docs/performance/qwen38/toolcall-benchmark-2026-08-22.md",
            "thinking_proxy.py": "src/gateway/thinking_proxy.py",
        }
        self.assertEqual(dict(mapping.exact_moves), expected)
        self.assertEqual(dict(mapping.pattern_moves), {
            "docs/superpowers/specs/**": "docs/archive/design-decisions/specs/",
            "docs/superpowers/plans/**": "docs/archive/design-decisions/plans/",
        })


class ContentManifestTests(unittest.TestCase):
    manifest_path = Path("site/src/data/content-manifest.json")

    def test_final_content_manifest_covers_final_git_index_once(self):
        tracked = set(git_tracked_paths(Path.cwd()))
        manifest = load_content_manifest(self.manifest_path)
        paths = [item["source_path"] for item in manifest]
        self.assertEqual(set(paths), tracked)
        self.assertEqual(len(set(paths)), len(paths))

    def test_baseline_destinations_join_final_manifest_once(self):
        ledger_path = next(
            Path("operations/migrations").glob("*-repository-layout.json")
        )
        ledger = load_ledger(ledger_path)
        manifest = load_content_manifest(self.manifest_path)
        claim_owners = {
            str(baseline_path): str(item["source_path"])
            for item in manifest
            for baseline_path in item["baseline_paths"]
        }
        claim_counts = Counter(
            str(baseline_path)
            for item in manifest
            for baseline_path in item["baseline_paths"]
        )
        self.assertEqual(set(claim_counts), {row.new_path for row in ledger})
        self.assertTrue(all(count == 1 for count in claim_counts.values()))
        for row in ledger:
            owner = claim_owners[row.new_path]
            if owner != row.new_path:
                self.assertEqual(row.disposition, "proven-generated-regenerate")
                self.assertEqual(
                    (row.new_path, owner),
                    (
                        "site/src/data/repo-index.json",
                        self.manifest_path.as_posix(),
                    ),
                )

    def test_manifest_metadata_matches_every_git_index_blob(self):
        manifest = load_content_manifest(self.manifest_path)
        self_item_path = self.manifest_path.as_posix()
        for item in manifest:
            source_path = str(item["source_path"])
            with self.subTest(source_path=source_path):
                if source_path == self_item_path:
                    self.assertEqual(
                        item["metadata_scope"], "self-reference-omitted"
                    )
                    self.assertIsNone(item["git_blob_oid"])
                    self.assertIsNone(item["byte_length"])
                    self.assertIsNone(item["sha256"])
                    continue

                index_line = subprocess.run(
                    [
                        "git",
                        "ls-files",
                        "-s",
                        "--",
                        source_path,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.rstrip("\n")
                git_mode, blob_oid, stage_and_path = index_line.split(" ", 2)
                stage, indexed_path = stage_and_path.split("\t", 1)
                blob = subprocess.run(
                    ["git", "cat-file", "blob", blob_oid],
                    check=True,
                    stdout=subprocess.PIPE,
                ).stdout
                self.assertEqual(stage, "0")
                self.assertEqual(indexed_path, source_path)
                self.assertEqual(item["metadata_scope"], "git-index-blob")
                self.assertEqual(item["git_mode"], git_mode)
                self.assertEqual(item["git_blob_oid"], blob_oid)
                self.assertEqual(item["byte_length"], len(blob))
                self.assertEqual(
                    item["sha256"], hashlib.sha256(blob).hexdigest()
                )

    def test_manifest_routes_are_unique_and_partitioned(self):
        manifest = load_content_manifest(self.manifest_path)
        route_owners: dict[str, str] = {}
        for item in manifest:
            source_path = str(item["source_path"])
            repository_route = str(item["repository_route"])
            self.assertEqual(
                repository_route, f"/repository/{source_path}"
            )
            routes = [repository_route]
            canonical_route = str(item["canonical_route"])
            if canonical_route != repository_route:
                routes.append(canonical_route)
            for route in routes:
                with self.subTest(route=route):
                    self.assertNotIn(route, route_owners)
                    route_owners[route] = source_path

        canonical_routes = {
            str(item["canonical_route"])
            for item in manifest
            if item["canonical_route"] != item["repository_route"]
        }
        self.assertTrue(
            all(
                route == "/docs"
                or route.startswith(("/docs/", "/evidence/", "/archive/"))
                for route in canonical_routes
            )
        )


class RouteManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            PRODUCTION_ROUTE_LEDGER.read_text(encoding="utf-8")
        )
        cls.routes = [
            LegacyRoute(**row) for row in cls.payload["rows"]
        ]
        cls.by_old_url = {route.old_url: route for route in cls.routes}

    def test_baseline_route_capture_counts_every_page(self):
        self.assertEqual(len(self.routes), 424)
        self.assertEqual(len(self.by_old_url), 424)
        self.assertIn(
            "/v100-perfs/docs/getting-started/",
            {route.trailing_slash_url for route in self.routes},
        )

    def test_committed_route_ledger_is_the_exact_immutable_capture(self):
        self.assertEqual(self.payload["schema_version"], 1)
        self.assertEqual(
            self.payload["base_commit"],
            "67ef398f9da504e020db1349956e2990f2876171",
        )
        self.assertEqual(self.payload["base_path"], "/v100-perfs")
        self.assertEqual(
            self.payload["source_layout_ledger"],
            "operations/migrations/"
            "67ef398f9da504e020db1349956e2990f2876171-repository-layout.json",
        )
        self.assertEqual(self.payload["baseline_route_definitions"], [
            "site/astro.config.mjs",
            "site/src/pages/docs/[...slug].astro",
            "site/src/data/repo-index.json",
        ])
        self.assertEqual(self.payload["curated_route_count"], 9)
        self.assertEqual(self.payload["read_route_count"], 413)
        self.assertEqual(self.payload["row_count"], 424)
        self.assertEqual(
            self.payload["rows"],
            [asdict(route) for route in self.routes],
        )
        self.assertEqual(len(self.by_old_url), len(self.routes))
        self.assertEqual(
            len({route.encoded_url for route in self.routes}),
            len(self.routes),
        )
        self.assertEqual(
            len({route.trailing_slash_url for route in self.routes}),
            len(self.routes),
        )

    def test_route_forms_and_canonical_targets_are_exact(self):
        self.assertEqual(
            self.by_old_url["/v100-perfs/docs/experience"],
            LegacyRoute(
                old_url="/v100-perfs/docs/experience",
                encoded_url="/v100-perfs/docs/experience",
                trailing_slash_url="/v100-perfs/docs/experience/",
                canonical_target=(
                    "/v100-perfs/archive/chronicles/"
                    "v100-fastllm-experience/"
                ),
            ),
        )
        self.assertEqual(
            self.by_old_url[
                "/v100-perfs/read/site/src/pages/docs/[...slug].astro"
            ],
            LegacyRoute(
                old_url=(
                    "/v100-perfs/read/site/src/pages/docs/[...slug].astro"
                ),
                encoded_url=(
                    "/v100-perfs/read/site/src/pages/docs/"
                    "%5B...slug%5D.astro"
                ),
                trailing_slash_url=(
                    "/v100-perfs/read/site/src/pages/docs/"
                    "%5B...slug%5D.astro/"
                ),
                canonical_target=(
                    "/v100-perfs/repository/site/src/pages/docs/"
                    "%5B...slug%5D.astro/"
                ),
            ),
        )
        self.assertEqual(
            self.by_old_url[
                "/v100-perfs/read/results/q38-sweep/MATRIX.md"
            ].canonical_target,
            "/v100-perfs/docs/performance/qwen38/sweep-matrix/",
        )
        self.assertEqual(
            self.by_old_url[
                "/v100-perfs/read/benchmarks/agent_bench.py"
            ].canonical_target,
            "/v100-perfs/evidence/benchmarks/agent_bench.py/",
        )

    def test_generator_eligible_routes_missing_from_old_index_are_frozen(self):
        route = self.by_old_url[
            "/v100-perfs/read/docs/superpowers/plans/"
            "2026-08-22-full-repository-docs-migration.md"
        ]
        self.assertEqual(
            route.canonical_target,
            "/v100-perfs/archive/design-decisions/plans/"
            "2026-08-22-full-repository-docs-migration/",
        )

    def test_alias_graph_rejects_alias_to_alias_cycle(self):
        with self.assertRaisesRegex(ValueError, "alias cycle"):
            validate_alias_targets({
                "/old-a": "/old-b",
                "/old-b": "/old-a",
            })

    def test_alias_graph_rejects_acyclic_alias_to_alias_target(self):
        with self.assertRaisesRegex(ValueError, "target is another alias"):
            validate_alias_targets({
                "/old-a": "/old-b",
                "/old-b": "/canonical",
            })


if __name__ == "__main__":
    unittest.main()
