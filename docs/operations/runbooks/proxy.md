# Proxy runbook

This runbook separates read-only observation from deployment actions. Documentation capture may inspect state; it must not send inference, suspend, resume, switch profiles, mutate admin state, or restart services.

## Components

- Gateway: `src/gateway/thinking_proxy.py`, normally exposing the client-facing service.
- Backend: FastLLM `apiserver`, owned by the gateway in the production profile.
- Adapter: `fastllm_adapter.py`, a separately checksum-pinned production dependency.
- Profiles and launch material: `operations/handoff/`; machine-local runtime profiles remain outside tracked docs.
- Canonical current state: [Current production](../current-state/production.md).

## Read-only inspection

Use loopback or an equivalent trusted administrative path. Do not include credentials in command history, screenshots, or tracked artifacts.

```bash
curl --fail --silent http://127.0.0.1:8000/health
curl --fail --silent http://127.0.0.1:8002/health
curl --fail --silent http://127.0.0.1:8002/props
```

The authenticated admin state is also read-only:

```bash
curl --fail --silent \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  http://127.0.0.1:8002/admin/api/state
```

Check all of the following rather than treating HTTP 200 as complete readiness:

- proxy lifecycle state and `ready`;
- backend `ready`, `accepting`, `suspended`, active, and queued counts;
- model slug, KV dtype, batch, and token-pool properties;
- template startup validity and error;
- active admin profile;
- error counters relevant to the incident being investigated.

A health response does not prove that a cold or lazily loaded model can complete inference. A real inference readiness probe is a workload action, not a documentation snapshot action, and requires explicit operational approval.

## Capture a deployment snapshot

A canonical snapshot may use only:

- proxy and backend `/health`;
- backend `/props`;
- authenticated `GET /admin/api/state`;
- an explicit non-secret allowlist of process environment fields, including the backend command used to identify executable and model asset paths;
- local file byte lengths and SHA-256 checksums;

Record a UTC capture time and provenance for every field. Never infer a missing value from README, archive, handoff, profile prose, or a prior snapshot. Store missing values as unknown. Never record the admin token.

The capture itself must send no generation request and perform no POST request, signal, process restart, or state change.

## Verify the tracked gateway contract locally

From the repository root, use the focused resolver and gateway gates selected by the integration owner. The canonical source scan must target `src/gateway/`, not the immutable deployed proxy snapshot under `operations/snapshots/`.

For documentation authority and route coverage:

```bash
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
```

For the adapter safety pin:

```bash
sha256sum fastllm_adapter.py
```

Compare the result with the immutable migration ledger or approved verification report. Do not import or execute a sibling production proxy merely to inspect it.

## Prepare a deployment

Deployment is a separate change window. Before it begins:

1. ensure the intended gateway commit and adapter checksum passed local verification;
2. identify the exact machine-local profile and its rollback profile;
3. record the pre-deploy process identities and read-only state;
4. confirm the profile path is absolute, because relative paths can resolve under the project directory and silently fail;
5. confirm credentials are supplied from the runtime environment rather than tracked files.

The tracked launcher is invoked with an explicitly selected absolute profile path:

```bash
PROFILE=/absolute/path/to/selected-profile.env
./operations/handoff/start_prod.sh "$PROFILE"
```

This command is mutating. Do not run it during docs generation, static verification, or snapshot capture.

## Post-deploy checks

After an approved deployment, verify in order:

1. new proxy and backend process identities;
2. allowlisted environment values match the selected profile;
3. proxy and backend health plus authenticated admin state;
4. model/template/executable checksums where applicable;
5. focused protocol requests for the deployed change;
6. stable error envelopes and response invariants;
7. no unexpected error-counter movement.

Only then capture a new immutable snapshot and update the current-state page. A Git merge alone does not update production authority.

## Rollback

Use the preselected rollback profile through the same launcher, then repeat the post-deploy checks. Do not hand-edit the running process environment or copy a proxy file into place as an undocumented shortcut.

Preserve the failed deployment's read-only state, logs, commit, executable checksum, and profile identity as evidence. Do not rewrite the prior immutable snapshot; create a new timestamped observation.
