# Task 4 report

## Status

DONE_WITH_CONCERNS

Implementation commit: `84ec512070371adb2ffdbc4fa90aff8b406d2abf`

Focused expectation-fix commit: `aa18ce5fc1f2cebc6894f190bff3860436e457ad`

Review fix round 1: `2942b3f97d71ef5aee1bca52d85e9d6255bc149a`

Final pre-review amendment: `48531b7e5eceece70f98c367425e6a2a71974ae4`

Subjects:

- `Update references for repository layout cutover`
- `Fix reference scanner classification expectation`
- `Resolve curated documentation sources after migration`
- `Approve adapter test template path amendment`
- `Match explicit amendment validation error`
- `Refresh manifest after amendment test update` — final SHA `c7bb672894c4d65acc7d7ca1e6b5de63de8fe06f`

Combined summary: 41 tracked files changed in the implementation commit (410 insertions, 93 deletions), followed by a one-line test expectation correction, a three-file curated-source/scanner review fix, and a six-file explicit amendment plus deterministic generated-output update.

## Implemented

- Added `rewrite_repository_reference(path: str) -> str`, backed by the canonical layout mapping and preserving `./`, query, and fragment suffixes.
- Added `scan_internal_references(repo, ledger) -> ReferenceReport` with:
  - stale old-path findings for executable/current-canonical approved files;
  - broken internal Markdown-target findings;
  - validation that preserved old paths still have a unique ledger destination present in the final tree;
  - explicit preserved-reference classifications for archive, supporting research, evidence, captured artifacts/evidence, and generated output.
- Added focused rewrite/scanner tests, including active-versus-historical classification, missing canonical-target rejection, and the requested production-ledger zero-stale/zero-broken assertion.
- Repaired approved executable/config/current-document references after the physical cutover:
  - moved script root calculations and sibling calls;
  - gateway source/import paths and pinned root `fastllm_adapter.py` import location;
  - canonical chat-template paths;
  - benchmark/config output and evidence paths;
  - proxy test imports;
  - workflow, README, site generator, and approved current-document references.
- Added no old-path wrapper, alias, shim, or symlink.
- Did not edit `fastllm_adapter.py`, captured archive/evidence bodies, patches, logs, binaries, or snapshots.

## Files and counts

### Approved reference rewrites: 39 files

- `.github/workflows/deploy-docs.yml`
- `README.md`
- `docs/performance/qwen38/toolcall-benchmark-2026-08-22.md`
- `evidence/benchmarks/fastllm/agentic_load_bench.py`
- `evidence/benchmarks/fastllm/configs/thinkingcap_native_profiles.json`
- `evidence/benchmarks/fastllm/hermes_toolcall_bench.py`
- `experiments/video-generation/README.md`
- `operations/network/frp/frpc.toml`
- `operations/handoff/start_prod.sh`
- `operations/handoff/scripts/copy_fidelity_probe.sh`
- `operations/handoff/scripts/gpu_verify_window.sh`
- `operations/handoff/scripts/graft_mtp.sh`
- `operations/handoff/scripts/imatrix_session.sh`
- `operations/handoff/scripts/quantize_with_imatrix.sh`
- `operations/handoff/scripts/raise_concurrency.sh`
- `operations/handoff/scripts/run_copy_fidelity.sh`
- `operations/handoff/scripts/run_vision_fidelity.sh`
- `operations/handoff/scripts/sm70-audit/build_bench.sh`
- `operations/handoff/scripts/sweep_driver.sh`
- `operations/handoff/scripts/sweep_launch.sh`
- `operations/handoff/scripts/sweep_one.sh`
- `operations/handoff/scripts/upload_to_hf.sh`
- `operations/handoff/scripts/verify_prefix_cache.py`
- `site/scripts/gen-llms-txt.mjs`
- `src/gateway/thinking_proxy.py`
- `tests/proxy_stream_error_invariants.py`
- `tests/test_thinking_proxy_fastllm.py`
- `tests/test_thinking_proxy_lifecycle.py`
- `tools/scripts/launch_proxy_tmux.sh`
- `tools/scripts/run_proxy_q6_plain.sh`
- `tools/scripts/run_proxy_q6_plain_turbo4.sh`
- `tools/scripts/run_proxy_q6_plain_turbo4_lowwm.sh`
- `tools/scripts/start_llama_turboquant_4x140k.sh`
- `tools/scripts/start_proxy.sh`
- `tools/scripts/sweep_driver.sh`
- `tools/scripts/sweep_driver2.sh`
- `tools/scripts/sweep_launch.sh`
- `tools/scripts/sweep_one.sh`
- `tools/scripts/sweep_one2.sh`

### Task-owned scanner/API files: 2 files

- `tools/repository_layout/layout_manifest.py`
- `tests/test_repository_layout.py`

## Preserved unresolved references by classification

Ledger-driven inspection classified 636 old-path occurrences as preserved references with existing unique canonical ledger destinations. They do not count as stale executable/current-canonical references:

- captured artifacts: 380
- archive documents: 133
- generated `llms.txt` awaiting its separately approved regeneration task: 77
- evidence bodies: 41
- captured evidence profile: 4
- supporting research: 1

Parent ruling applied without widening the allowlist:

1. `docs/research/performance/short-prefill-plan.md:48` is supporting research. Its `docs/analysis/prefill-exact-port-investigation.md` citation resolves through the immutable ledger/legacy alias to `docs/research/performance/prefill-exact-port.md`.
2. `operations/handoff/PRODUCTION-PROFILE.env:181,183,187,193` is captured evidence and remains byte-identical. Its old documentation citations resolve through the immutable ledger/legacy aliases.

Neither preserved file was edited.

## Verification and skipped commands

Per the parent instruction that implementer agents leave validation centralized, the following were not run:

- `python -m unittest tests.test_repository_layout -v`
- `python -m tools.repository_layout.layout_manifest verify-moves --repo .`
- builds, linters, formatters, and project-wide tests

Narrow read-only ledger/reference inspections confirmed the exact allowlist boundary, zero remaining exact old-path occurrences in approved destinations (excluding the intentionally ambiguous root basename handled by scanner policy), zero broken Markdown targets in approved current documents, and the preserved-reference counts above.

## Concerns

- The committed scanner/API and tests have not been executed by this implementer, per parent policy; the controller must run the two focused commands above.
- Supporting/captured legacy strings rely on the immutable ledger/legacy-alias renderer contract. If that render-time resolution changes, the preserved-reference policy must be revalidated rather than rewriting those byte-identical files.

## Parent validation follow-up

The parent focused test run exposed one test-only expectation mismatch: the fixture `archive.md` is an identity pin, so the scanner correctly classified its preserved reference as `identity-pinned`, not `other-byte-identical`. The expectation was corrected without weakening or changing scanner behavior. Validation remains delegated to the parent for rerun.

## Review fix round 1

Important review findings were addressed within the approved allowlist:

- Updated `site/src/pages/docs/[...slug].astro` from `./handoff/start_prod.sh` to `./operations/handoff/start_prod.sh`.
- Replaced all seven stale composed `join(repoDocs, ...)` curated sources with their immutable-ledger destinations.
- Taught the scanner to recognize `./`-prefixed repository references.
- Added resolved active-reference reporting and Astro `join(repoDocs, ...)` resolution.
- Added tests that enumerate all eight curated sources (README plus seven migrated documents) and assert that every resolved path exists.

That review round added no Task 5 content-manifest, site-route, narrative, adapter, or evidence work.

## Final pre-review amendment

The parent combined run established that `tests/test_fastllm_adapter.py` still referenced the removed `chat_templates/` root, producing 12 test errors. The baseline omitted that test from the rewrite allowlist, so the correction is audited separately rather than changing the immutable ledger:

- Added `operations/migrations/67ef398f9da504e020db1349956e2990f2876171-approved-rewrite-amendments.json` with exactly one identity-path row for `tests/test_fastllm_adapter.py`, the reason `template fixture follows chat_templates -> config/chat-templates`, and the controller ruling reference.
- Added strict amendment loading: exact schema/fields, filename/base-commit agreement, nonempty rationale/ruling, explicit non-wildcard source, unique source, source/destination equality with the immutable ledger, and rejection of redundant baseline approvals.
- Updated `verify-moves` to accept only explicit amendment objects, report `approved_rewrite_amendments`, and approve no generalized path class.
- Added unit and production tests proving an unamended rewrite fails, the exact amendment passes, wildcard/non-ledger approval fails, and the baseline row remains `same-byte-identical`.
- Updated only the two template fixture constants in `tests/test_fastllm_adapter.py` to `config/chat-templates/...`; `fastllm_adapter.py` remains untouched and no shim was added.
- Deterministically regenerated `llms.txt` and `site/src/data/content-manifest.json` from the staged final index using the prescribed manifest → llms → manifest sequence. Output reported 476 tracked paths and 476 unique absolute HTTPS canonical URLs.

The parent retains responsibility for the full adapter, layout, reasoning, and `verify-moves` rerun.

## Amendment validation expectation follow-up

The parent focused run confirmed the wildcard amendment is deterministically rejected at the earlier explicit-path validation gate. The test expectation now matches `rewrite amendment paths must be explicit` rather than the unreachable later ledger-membership message. Scanner/verifier behavior was not changed. Commit: `ab819c49d1a74c3f06b433bb2a14627efd6fb0cf`.

## Final generated metadata refresh

After the one-line assertion commit, the deterministic generator updated only `site/src/data/content-manifest.json` metadata for `tests/test_repository_layout.py`; `llms.txt` was unchanged. The required generated metadata delta was committed without source changes at `c7bb672894c4d65acc7d7ca1e6b5de63de8fe06f`.

## Final whole-branch authorization correction

Commit `ceec6d8ab88edf8f30644d84847375f4588a073b` replaced the vague Main-only ruling reference with the user's verbatim delegated instruction, `最终的决定权交给你，别漏文件`, and the controller's exact exception boundary: only the baseline `tests/test_fastllm_adapter.py` row may change, and only its two fixture constants from removed `chat_templates/...` paths to `config/chat-templates/...` paths. The amendment remains one explicit identity-path row; it grants no wildcard or general path-class approval. The immutable baseline ledger, adapter, and runtime shims remain unchanged.

The existing schema already carried the complete auditable authorization in `controller_ruling_reference`, so no field, loader abstraction, or validation exception was added. Its exact-field, one-row production expectation, explicit-path, unique-source, ledger-join, and redundant-approval checks remain strict; only the production expectation was updated to the exact authorization text.

After documentation truth commit `7c5ea5e88bc69f7483c469c94570740ae2a010e2` was present, the required manifest → `llms.txt` → manifest sequence first reported 476 tracked paths and 476 unique absolute HTTPS canonical URLs. `llms.txt` remained byte-identical because no route had changed, and commit `581a5e580b8b909cb49e93aa93a1d204270ddf33` recorded that staged-Git metadata refresh.

Commit `591c124` then made this required report a tracked repository artifact. The final post-report manifest → `llms.txt` → manifest sequence consequently reported 477 tracked paths and 477 unique absolute HTTPS canonical URLs, adding only the report's repository URL to `llms.txt` and refreshing its indexed metadata in `site/src/data/content-manifest.json`.
