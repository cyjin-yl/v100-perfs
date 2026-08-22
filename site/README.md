# V100 FastLLM documentation site

This Astro site renders the repository's final Git index through one generated content manifest. Canonical docs, evidence, archive material, and repository metadata retain distinct routes and authority.

## Data model

`src/data/content-manifest.json` is generated from staged Git objects, not a recursive working-tree scan. Every final tracked path has one row containing source and repository routes, title, summary, status, authority, model generation, navigation metadata, Git mode, size, hash, and render kind.

The same manifest drives:

- primary navigation and active-section state;
- breadcrumbs and related canonical links;
- canonical docs, evidence, archive, and repository routes;
- the homepage canonical reading paths;
- repository metadata and safe download links.

The homepage's production summary is read from the immutable deployment snapshot used by `docs/operations/current-state/production.md`. It is static and timestamped. The site does not poll a local admin service, read browser credentials, send inference, or mutate production state.

## Primary navigation

The visible navigation is exactly:

```text
Overview
Getting Started
Concepts
Guides
Operations
Reference
Performance
Research
Archive
Repository
```

Current canonical section targets come from manifest metadata. Research and Archive enter through the canonical overview's labeled gateways rather than promoting supporting or historical files into current navigation. All deployed links are prefixed with `import.meta.env.BASE_URL`; components contain no hard-coded repository base path.

## Routes

```text
/docs/<canonical-doc-route>
/evidence/<manifest-route>
/archive/<canonical-archive-route>
/repository/<tracked-path>
/repo
```

Text is escaped or sanitized before display. Binary and non-regular objects render as metadata with Git mode, size, hash, and a safe GitHub blob link. Historical notices come from layout metadata; preserved archive bodies are not rewritten.

`npm run build` executes the deterministic prebuild sequence content manifest → `llms.txt` → content manifest, runs Astro with the configured base path, then runs the alias postbuild. The postbuild reads the immutable 424-route ledger, writes 423 static redirect documents plus canonical passthroughs, and uses one base-path-safe canonical target for each document's canonical link, meta refresh, `location.replace`, and visible link. It rewrites the sitemap to canonical URLs only.

## Local commands

From the repository root:

```bash
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
```

From `site/`:

```bash
npm install
npm run dev
npm run build
```

`predev` runs the same deterministic generation sequence before Astro's development server. A local build verifies generated artifacts and routes; deployed HTTPS status remains a post-deploy check.

## Rendering boundary

- Astro owns routing, layouts, manifest projection, and static content rendering.
- React is reserved for components that need client state. The canonical homepage dashboard and snapshot card render statically and do not hydrate.
- Global tokens and responsive styles live in `src/styles/global.css`; there is no Tailwind runtime.
- Keyboard focus is visible, navigation is labeled, the page has a skip link, and reduced-motion preferences disable transitions.
