import gc
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unittest
import warnings
from dataclasses import asdict
from pathlib import Path

from tools.repository_layout.layout_manifest import (
    LayoutError,
    LayoutMapping,
    build_baseline_ledger,
    verify_final_tree,
)
from tools.repository_layout.route_manifest import (
    LegacyRoute,
    capture_legacy_routes,
    validate_alias_targets,
)


PRODUCTION_MAPPING = Path("tools/repository_layout/layout-map.json")
PRODUCTION_ROUTE_LEDGER = Path(
    "operations/migrations/"
    "67ef398f9da504e020db1349956e2990f2876171-legacy-routes.json"
)


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


class RouteManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes = capture_legacy_routes(Path.cwd(), Path("site/dist"))
        cls.by_old_url = {route.old_url: route for route in cls.routes}

    def test_baseline_route_capture_counts_every_page(self):
        self.assertEqual(len(self.routes), 424)
        self.assertEqual(len(self.by_old_url), 424)
        self.assertIn(
            "/v100-perfs/docs/getting-started/",
            {route.trailing_slash_url for route in self.routes},
        )

    def test_committed_route_ledger_is_the_exact_immutable_capture(self):
        payload = json.loads(PRODUCTION_ROUTE_LEDGER.read_text(encoding="utf-8"))
        self.assertEqual(payload["base_path"], "/v100-perfs")
        self.assertEqual(payload["curated_route_count"], 9)
        self.assertEqual(payload["read_route_count"], 413)
        self.assertEqual(payload["row_count"], 424)
        self.assertEqual(payload["rows"], [asdict(route) for route in self.routes])

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
