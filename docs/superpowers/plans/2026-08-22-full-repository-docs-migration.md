# Full Repository Documentation Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Physically reorganize the tracked repository into the approved domain layout, preserve every baseline file through an immutable ledger, and replace the current hard-coded Astro façade with a manifest-driven documentation site.

**Architecture:** Work starts in an isolated Git worktree. An immutable baseline ledger and route ledger are captured before mutation; prefix and explicit mappings are then applied with `git mv`. A separate final content manifest drives canonical docs, evidence/repository pages, legacy aliases, navigation, sitemap, and `llms.txt`.

**Tech Stack:** Git, Python 3.14, Astro 5, React 19, TypeScript, Node.js, `marked`, GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-08-22-full-repository-docs-and-thinking-controls-design.md`

## Global Constraints

- Use an isolated worktree created through `superpowers:using-git-worktrees`; do not mutate the active production workspace while implementing.
- `fastllm_adapter.py` and its production dependency closure must be byte-identical before and after implementation.
- Do not edit or sync sibling `/1CatVLLM/thinking_proxy.py`; do not restart proxy/backend or send inference requests.
- Every baseline tracked file must have one immutable ledger row and one unique destination.
- Only §6 spec allowlist paths may be rewritten or regenerated; every other baseline path must preserve bytes and Git mode.
- Never deduplicate equal files. Logs, binaries, patches, profiles, snapshots, JSON, TSV and captured outputs are retained evidence.
- Baseline migration ledger and final site content manifest are separate artifacts.
- All old baseline routes are frozen before mutation and receive target-correct, cycle-free static aliases.
- Commit byte-identical moves separately from narrative/site rewrites.

---

### Task 1: Create the migration ledger and verifier

**Files:**
- Create: `tools/repository_layout/__init__.py`
- Create: `tools/repository_layout/layout_manifest.py`
- Create: `tools/repository_layout/layout-map.json`
- Create: `tests/test_repository_layout.py`
- Create at runtime: `operations/migrations/<base-commit>-repository-layout.json`

**Interfaces:**
- Produces: `LedgerRow(old_path, new_path, git_mode, blob_oid, byte_length, sha256, disposition)`.
- Produces: `build_baseline_ledger(repo: Path, mapping: LayoutMapping) -> list[LedgerRow]`.
- Produces: `verify_final_tree(repo: Path, ledger: Sequence[LedgerRow], rewrite_allowlist: set[str], regenerate_allowlist: set[str]) -> VerificationReport`.
- Consumes later: final content manifest paths from Task 4.

- [ ] **Step 1: Write failing mapping and conservation tests**

```python
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.repository_layout.layout_manifest import (
    LayoutMapping,
    build_baseline_ledger,
    verify_final_tree,
)


class LayoutManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        for rel, data in {
            "benchmarks/a.log": b"a",
            "results/r.json": b"{}",
            "fastllm_adapter.py": b"pinned",
        }.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_mapping_is_total_and_has_no_destination_collisions(self):
        mapping = LayoutMapping.from_file(
            Path("tools/repository_layout/layout-map.json"))
        rows = build_baseline_ledger(self.repo, mapping)
        self.assertEqual({r.new_path for r in rows}, {
            "evidence/benchmarks/a.log",
            "evidence/results/r.json",
            "fastllm_adapter.py",
        })

    def test_verifier_rejects_rewritten_byte_identical_blob(self):
        mapping = LayoutMapping.from_file(
            Path("tools/repository_layout/layout-map.json"))
        rows = build_baseline_ledger(self.repo, mapping)
        for row in rows:
            source = self.repo / row.old_path
            target = self.repo / row.new_path
            if source != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        target = self.repo / rows[0].new_path
        target.write_bytes(b"changed")
        report = verify_final_tree(self.repo, rows, set(), set())
        self.assertEqual(report.lost_blobs, 0)
        self.assertEqual(report.unexpected_rewrites, 1)
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python -m unittest tests.test_repository_layout -v
```

Expected: import failure because `tools.repository_layout.layout_manifest` does not exist.

- [ ] **Step 3: Implement exact mapping and immutable row generation**

`layout-map.json` must encode these ordered prefix mappings:

```json
{
  "identity_pins": [
    ".github/**", ".gitignore", "README.md", "llms.txt",
    "fastllm_adapter.py", "site/**", "tests/**"
  ],
  "prefix_moves": {
    "benchmarks/": "evidence/benchmarks/",
    "results/": "evidence/results/",
    "handoff/": "operations/handoff/",
    "frp/": "operations/network/frp/",
    "patches/": "operations/patches/",
    "deployed/": "operations/snapshots/deployed/",
    "chat_templates/": "config/chat-templates/",
    "scripts/": "tools/scripts/",
    "video-gen/": "experiments/video-generation/"
  },
  "exact_moves": {
    "thinking_proxy.py": "src/gateway/thinking_proxy.py"
  }
}
```

Add all explicit docs mappings from spec §4.2, including:

```text
docs/superpowers/specs/** -> docs/archive/design-decisions/specs/**
docs/superpowers/plans/** -> docs/archive/design-decisions/plans/**
```

Implementation requirements:

```python
@dataclass(frozen=True)
class LedgerRow:
    old_path: str
    new_path: str
    git_mode: str
    blob_oid: str
    byte_length: int
    sha256: str
    disposition: str
```

Use `git ls-files -s -z` as the only baseline inventory. Reject unmapped paths, absolute/parent destinations, duplicate/case-fold/NFC collisions, and destination overlap. Do not read ignored/untracked files.

- [ ] **Step 4: Encode the pre-approved rewrite/regenerate allowlists**

Copy the exact allowlists from spec §6 into `layout-map.json`. The loader must reject an allowlisted path not present in the baseline and must default all non-allowlisted rows to byte-identical dispositions.

- [ ] **Step 5: Run tests and confirm GREEN**

Run:

```bash
python -m unittest tests.test_repository_layout -v
```

Expected: all mapping, collision, hash, mode and allowlist tests pass.

- [ ] **Step 6: Generate the immutable baseline ledger**

Run in the isolated worktree before any `git mv`:

```bash
python -m tools.repository_layout.layout_manifest capture-layout --repo .
```

Expected output includes:

```text
unmapped=0 collisions=0
ledger=operations/migrations/<base-commit>-repository-layout.json
```

Record the worktree's current `fastllm_adapter.py` SHA-256 in the ledger metadata.

- [ ] **Step 7: Commit verifier and ledger**

```bash
git add tools/repository_layout tests/test_repository_layout.py operations/migrations
git commit -m "Add lossless repository migration ledger"
```

---

### Task 2: Freeze the legacy route ledger

**Files:**
- Create: `tools/repository_layout/route_manifest.py`
- Modify: `tests/test_repository_layout.py`
- Create at runtime: `operations/migrations/<base-commit>-legacy-routes.json`

**Interfaces:**
- Produces: `capture_legacy_routes(repo: Path, dist: Path) -> list[LegacyRoute]`.
- Produces: `LegacyRoute(old_url, encoded_url, trailing_slash_url, canonical_target)`.
- Consumed by: static alias generation in Task 5.

- [ ] **Step 1: Add failing tests for exact route capture and alias-cycle detection**

```python
from tools.repository_layout.route_manifest import (
    capture_legacy_routes,
    validate_alias_targets,
)


class RouteManifestTests(unittest.TestCase):
    def test_baseline_route_capture_counts_every_page(self):
        routes = capture_legacy_routes(Path.cwd(), Path("site/dist"))
        self.assertEqual(len(routes), 421)
        self.assertIn(
            "/v100-perfs/docs/getting-started/",
            {r.trailing_slash_url for r in routes},
        )

    def test_alias_graph_rejects_alias_to_alias_cycle(self):
        with self.assertRaisesRegex(ValueError, "alias cycle"):
            validate_alias_targets({
                "/old-a": "/old-b",
                "/old-b": "/old-a",
            })
```
- [ ] **Step 2: Run route tests and confirm RED**

Run:

```bash
npm --prefix site run build
python -m unittest tests.test_repository_layout -v
```

Expected: `capture_legacy_routes` missing.

- [ ] **Step 3: Implement baseline build/route capture**

Run the current site build before mutation. Capture `/`, `/repo`, nine curated `/docs/*`, and 410 `/read/*` routes from baseline definitions and `dist/`. Store base path, raw path, percent-encoded form, trailing-slash form and approved canonical target. Do not infer old routes from the post-migration site.

- [ ] **Step 4: Generate and validate the 421-row route ledger**

```bash
npm --prefix site run build
python -m tools.repository_layout.route_manifest capture --repo . --dist site/dist
```

Expected:

```text
legacy_routes=421 missing_artifacts=0 duplicate_urls=0
```

- [ ] **Step 5: Commit the immutable route ledger**

```bash
git add tools/repository_layout/route_manifest.py tests/test_repository_layout.py operations/migrations
git commit -m "Freeze legacy documentation routes"
```

---

### Task 3: Apply the full physical migration

**Files:**
- Move: every path selected by `layout-map.json`
- Preserve unchanged: `fastllm_adapter.py`, `.github/**`, `.gitignore`, `README.md`, `llms.txt`, `site/**`, `tests/**`
- Verify: immutable baseline ledger from Task 1

**Interfaces:**
- Produces: moved path topology required by the reasoning-control plan and all later docs tasks.
- Barrier: the effort subagent may start only after this task's commit exists.

- [ ] **Step 1: Run a dry-run and save its exact summary**

```bash
python -m tools.repository_layout.layout_manifest apply --repo . --dry-run
```

Expected: every baseline path receives exactly one disposition; no mutation occurs.

- [ ] **Step 2: Apply moves with Git-aware operations**

The implementation command must call `git mv` per ledger row, deepest paths first, and create destination parents. It must not rewrite contents. Representative moves:

```bash
git mv benchmarks evidence/benchmarks
git mv results evidence/results
git mv handoff operations/handoff
git mv frp operations/network/frp
git mv patches operations/patches
git mv deployed operations/snapshots/deployed
git mv chat_templates config/chat-templates
git mv scripts tools/scripts
git mv video-gen experiments/video-generation
git mv thinking_proxy.py src/gateway/thinking_proxy.py
```

Then apply the explicit docs mappings from the ledger.

- [ ] **Step 3: Verify every byte-identical move before editing references**

```bash
python -m tools.repository_layout.layout_manifest verify-moves --repo .
```

Expected:

```text
legacy_rows=<baseline-count> joined=<baseline-count>
lost_blobs=0 unexpected_rewrites=0 mode_changes=0
```

- [ ] **Step 4: Assert the production safety pin**

```bash
sha256sum fastllm_adapter.py
```

Expected: exactly the baseline ledger SHA-256.

- [ ] **Step 5: Commit byte-identical moves only**

```bash
git add -A
git commit -m "Reorganize repository into domain directories"
```

Do not include reference rewrites or generated outputs in this commit.

---

### Task 4: Repair references after the clean cutover

**Files:**
- Modify only paths in the spec §6 approved-rewrite allowlist
- Modify: `tools/repository_layout/layout_manifest.py`
- Test: `tests/test_repository_layout.py`
- Preserve: historical archive bodies, captured evidence payloads, `fastllm_adapter.py`

**Interfaces:**
- Consumes: baseline ledger old→new mapping.
- Consumes: `load_ledger(path: Path) -> list[LedgerRow]` from Task 1.
- Produces: `rewrite_repository_reference(path: str) -> str`.
- Produces: `scan_internal_references(repo: Path, ledger: Sequence[LedgerRow]) -> ReferenceReport`.

- [ ] **Step 1: Add failing reference-resolution tests**

```python
def test_every_tracked_internal_reference_resolves_after_mapping(self):
    ledger_path = next(
        Path("operations/migrations").glob("*-repository-layout.json"))
    report = scan_internal_references(
        Path.cwd(),
        load_ledger(ledger_path),
    )
    self.assertEqual(report.broken_internal_links, [])
    self.assertEqual(report.stale_old_paths, [])
```

Expected RED: old `benchmarks/`, `results/`, `handoff/`, `chat_templates/`, `scripts/` and `video-gen/` references remain.

- [ ] **Step 2: Update executable imports and path-bearing configs first**

Update moved Python/shell/config code to new canonical paths. Do not add old-path wrappers, aliases or symlinks. Preserve operational semantics and executable modes. `fastllm_adapter.py` is excluded.

- [ ] **Step 3: Update current/allowlisted docs and site references**

Only rewrite baseline files listed in spec §6. Archive pages remain byte-identical; their links are resolved at render time from metadata rather than editing archive bodies.

- [ ] **Step 4: Run reference and production-checksum tests**

```bash
python -m unittest tests.test_repository_layout -v
python -m tools.repository_layout.layout_manifest verify-moves --repo .
```

Expected: no stale executable references; production safety-pin checksum unchanged.

- [ ] **Step 5: Commit reference cutover**

```bash
git add -A
git commit -m "Update references for repository layout cutover"
```

---

### Task 5: Build the final content manifest and canonical routes

**Files:**
- Rewrite: `site/scripts/generate-repo-index.mjs`
- Create: `site/scripts/validate-content-manifest.mjs`
- Replace generated: `site/src/data/repo-index.json` with `site/src/data/content-manifest.json`
- Rewrite: `site/src/pages/docs/[...slug].astro`
- Rewrite: `site/src/pages/read/[...path].astro`
- Rewrite: `site/src/pages/repo.astro`
- Create: `site/src/pages/evidence/[...path].astro`
- Create: `site/src/pages/repository/[...path].astro`
- Create: `site/src/pages/archive/[...slug].astro`
- Test: `tests/test_repository_layout.py`

**Interfaces:**
- Produces `ContentItem` with fields from spec §5.1.
- Produces canonical and repository/evidence routes from one manifest.
- Consumes immutable baseline ledger only for legacy joins, never as the final inventory.
- Consumes `git_tracked_paths(repo: Path) -> list[str]` and `load_ledger(path)` from Task 1.
- Adds test helper `load_content_manifest(path: Path) -> list[dict[str, object]]`.

- [ ] **Step 1: Add failing final-coverage tests**

```python
from collections import Counter


def test_final_content_manifest_covers_final_git_index_once(self):
    tracked = set(git_tracked_paths(Path.cwd()))
    manifest = load_content_manifest(
        Path("site/src/data/content-manifest.json"))
    paths = [item["source_path"] for item in manifest]
    self.assertEqual(set(paths), tracked)
    self.assertEqual(len(set(paths)), len(paths))

def test_baseline_destinations_join_final_manifest_once(self):
    ledger_path = next(
        Path("operations/migrations").glob("*-repository-layout.json"))
    ledger = load_ledger(ledger_path)
    manifest = load_content_manifest(
        Path("site/src/data/content-manifest.json"))
    path_counts = Counter(item["source_path"] for item in manifest)
    self.assertTrue(all(path_counts[row.new_path] == 1 for row in ledger))
```

- [ ] **Step 2: Run tests and confirm RED**

Expected: `content-manifest.json` is missing and old generator omits tracked logs/binaries.

- [ ] **Step 3: Implement Git-index-backed generation**

Use `git ls-files -s -z`; never recurse through the working tree. Classify text, logs, extensionless evidence, binary metadata and generated artifacts. Reject untracked inputs. Include `source_path`, route, title, summary, status, authority, model generation, navigation, aliases, Git mode, length and SHA-256.

- [ ] **Step 4: Implement safe route rendering**

- Text/Markdown: sanitized rendered page.
- Source/config/log: escaped code view.
- Binary: metadata/hash/mode/size plus GitHub blob download link.
- Historical pages: render archive banner in layout; never mutate source body.
- Relative links: resolve through content manifest.

- [ ] **Step 5: Validate final coverage and route uniqueness**

```bash
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
python -m unittest tests.test_repository_layout -v
```

Expected: final tracked coverage 100%, baseline join 100%, no route collisions.

- [ ] **Step 6: Commit manifest and routing layer**

```bash
git add site tests/test_repository_layout.py
git commit -m "Drive documentation routes from tracked content manifest"
```

---

### Task 6: Rewrite canonical narrative and navigation

**Files:**
- Rewrite: `README.md`
- Create: canonical docs listed in spec §5.2
- Rewrite/move: `docs/reference/models/qwen38/sampling-and-template.md`
- Rewrite: `site/src/pages/index.astro`
- Rewrite: `site/src/layouts/BaseLayout.astro`
- Rewrite: `site/src/components/ProtoDashboard.tsx`
- Rewrite: `site/src/components/RepoExplorer.tsx`
- Rewrite: `site/src/components/StatusCard.tsx`
- Rewrite: `site/src/styles/global.css`
- Rewrite: `site/README.md`

**Interfaces:**
- Consumes: deployment snapshot and final content manifest.
- Produces: one canonical current-production page and normal docs navigation.

- [ ] **Step 1: Capture the read-only deployment snapshot**

Use only `/health`, `/props`, admin state, allowlisted process environment and local checksums. Write:

```text
docs/_data/deployment-snapshots/<captured-at>-qwen38.json
```

Include per-field provenance and capture time. Do not send a generation request.

- [ ] **Step 2: Write canonical pages from the approved narrative contract**

Create overview, architecture, reasoning-control, OpenWebUI, client-compatibility, current-state, proxy runbook, API reference and performance-methodology pages. Historical reports are linked as archive/evidence, never summarized as current truth.

- [ ] **Step 3: Replace hard-coded navigation with manifest navigation**

Main navigation must be exactly:

```text
Overview, Getting Started, Concepts, Guides, Operations,
Reference, Performance, Research, Archive, Repository
```

Use `import.meta.env.BASE_URL`; no hard-coded `/v100-perfs` literals in components.

- [ ] **Step 4: Rewrite README as a repository entrance**

README describes scope, canonical docs entry, directory map, current-state source and verification commands. It must not duplicate the production snapshot or EXPERIENCE chronology.

- [ ] **Step 5: Run content and authority tests**

```bash
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
```

Expected: exactly one `status=current, authority=canonical` production page; historical docs never appear in primary nav.

- [ ] **Step 6: Commit narrative cutover**

```bash
git add README.md docs site
git commit -m "Replace historical façade with canonical documentation narrative"
```

---

### Task 7: Generate legacy aliases, sitemap and llms.txt

**Files:**
- Create: `site/scripts/generate-legacy-aliases.mjs`
- Rewrite: `site/scripts/gen-llms-txt.mjs`
- Regenerate: `llms.txt`
- Rewrite: `.github/workflows/deploy-docs.yml`
- Test: `tests/test_repository_layout.py`

**Interfaces:**
- Consumes immutable route ledger plus final canonical mapping.
- Produces static aliases with canonical/meta/JS/visible targets and unique absolute `llms.txt` URLs.
- Produces `verify_alias_build(route_ledger_path: Path, dist: Path) -> AliasReport`.

- [ ] **Step 1: Add failing alias-target tests**

```python
def test_every_legacy_route_builds_exact_noncyclic_alias(self):
    route_ledger_path = next(
        Path("operations/migrations").glob("*-legacy-routes.json"))
    report = verify_alias_build(route_ledger_path, Path("site/dist"))
    self.assertEqual(report.missing, [])
    self.assertEqual(report.wrong_targets, [])
    self.assertEqual(report.cycles, [])
```

- [ ] **Step 2: Implement alias generation from the frozen route ledger**

Each alias page must include the same expected target in `<link rel="canonical">`, meta refresh, JS fallback and visible anchor. Reject alias-to-alias targets and cycles.

- [ ] **Step 3: Rewrite llms.txt generation**

Generate one absolute HTTPS URL per canonical item. Do not perform three overlapping directory walks. Pre-merge validation maps each URL to a local `dist/` artifact.

- [ ] **Step 4: Broaden workflow triggers**

Ensure every final content-manifest source change triggers the docs workflow. The boring safe policy is all tracked repository content except explicitly ignored runtime/model paths, which are not tracked anyway.

- [ ] **Step 5: Run local alias/link checks**

```bash
npm --prefix site run build
python -m unittest tests.test_repository_layout -v
node site/scripts/validate-content-manifest.mjs
```

Expected: 421/421 legacy route aliases valid; internal links zero broken; `llms.txt` unique and locally resolvable.

- [ ] **Step 6: Commit URL compatibility layer**

```bash
git add site .github/workflows/deploy-docs.yml llms.txt tests/test_repository_layout.py
git commit -m "Preserve legacy documentation URLs"
```

---

### Task 8: Verify, merge and prune the worktree

**Files:**
- Verify all changed files
- No production files outside the isolated worktree may change before merge

**Interfaces:**
- Consumes all prior commits plus the reasoning-control plan commit.
- Produces merged `main`, post-merge verification evidence and no stale worktree.

- [ ] **Step 1: Run full worktree verification**

```bash
python -m unittest tests.test_repository_layout -v
python -m unittest tests.test_thinking_proxy_fastllm -v
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
npm --prefix site run build
```

Run browser checks against the built/local site for desktop and mobile navigation, canonical docs, evidence, repository viewer and representative legacy aliases.

- [ ] **Step 2: Verify immutable production dependency checksums**

Compare `fastllm_adapter.py` and the dependency closure recorded in the baseline ledger. Expected: no changed bytes or modes.

- [ ] **Step 3: Merge implementation commits into current main**

Use the repository's normal fast-forward or clean merge path. Do not deploy/sync sibling proxy and do not restart production.

- [ ] **Step 4: Re-run verification in merged main**

Repeat ledger/join tests, focused resolver tests, manifest generation, Astro build and browser smoke from the main worktree. Merge verification must produce the same counts and targets as the isolated worktree.

- [ ] **Step 5: Remove and prune the implementation worktree**

```bash
git worktree remove <implementation-worktree-path>
git worktree prune
git worktree list --porcelain
```

Expected: the implementation worktree path is absent and no stale administrative entry remains.

- [ ] **Step 6: Commit any deterministic generated outputs produced by the main-worktree verification**

Only `llms.txt` and `site/src/data/content-manifest.json` may change. If they differ from the implementation commit, treat that as a reproducibility failure; do not paper over it with a new commit.
