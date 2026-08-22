import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(site, "..");
const dist = resolve(site, "dist");
const migrations = resolve(repo, "operations", "migrations");
const routeLedgers = readdirSync(migrations)
  .filter((name) => name.endsWith("-legacy-routes.json"))
  .sort();

if (routeLedgers.length !== 1) {
  throw new Error(`expected exactly one immutable route ledger, found ${routeLedgers.length}`);
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function encodedRoute(route) {
  return route.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}

function normalizedRoute(route) {
  const decoded = decodeURIComponent(route);
  return decoded === "/" ? decoded : decoded.replace(/\/+$/, "");
}

function targetUrl(basePath, canonicalRoute) {
  const route = encodedRoute(canonicalRoute);
  return `${basePath}${route}${route.endsWith("/") ? "" : "/"}`;
}

function routeBelowBase(route, basePath) {
  const decoded = decodeURIComponent(route);
  if (decoded === basePath || decoded === `${basePath}/`) return "";
  const prefix = `${basePath}/`;
  if (!decoded.startsWith(prefix)) {
    throw new Error(`route is outside ${basePath}: ${route}`);
  }
  const relative = decoded.slice(prefix.length).replace(/\/+$/, "");
  if (relative.split("/").some((segment) => segment === "." || segment === "..")) {
    throw new Error(`unsafe legacy route: ${route}`);
  }
  return relative;
}

function artifactPath(route, basePath) {
  const relative = routeBelowBase(route, basePath);
  return relative ? resolve(dist, relative, "index.html") : resolve(dist, "index.html");
}

function htmlEscape(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function aliasHtml(target) {
  const escaped = htmlEscape(target);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Documentation moved</title>
<link rel="canonical" href="${escaped}">
<meta http-equiv="refresh" content="0;url=${escaped}">
<script>location.replace(${JSON.stringify(target)})</script>
</head>
<body><p>This documentation moved to <a href="${escaped}">${escaped}</a>.</p></body>
</html>
`;
}

function validateAliasGraph(aliases) {
  const sources = new Set(aliases.keys());
  const visiting = new Set();
  const visited = new Set();

  function visit(source, chain) {
    if (visiting.has(source)) {
      throw new Error(`alias cycle: ${[...chain, source].join(" -> ")}`);
    }
    if (visited.has(source)) return;
    visiting.add(source);
    const target = aliases.get(source);
    if (sources.has(target)) visit(target, [...chain, source]);
    visiting.delete(source);
    visited.add(source);
  }

  for (const source of sources) visit(source, []);
  for (const [source, target] of aliases) {
    if (sources.has(target)) {
      throw new Error(`alias target is another alias: ${source} -> ${target}`);
    }
  }
}

const routeLedger = readJson(resolve(migrations, routeLedgers[0]));
if (routeLedger.schema_version !== 1 || routeLedger.row_count !== 424) {
  throw new Error("legacy route ledger is not the immutable 424-row schema v1 capture");
}
if (!Array.isArray(routeLedger.rows) || routeLedger.rows.length !== routeLedger.row_count) {
  throw new Error("legacy route ledger row count does not match its metadata");
}

const basePath = routeLedger.base_path;
const layoutLedger = readJson(resolve(repo, routeLedger.source_layout_ledger));
const manifest = readJson(resolve(site, "src", "data", "content-manifest.json"));
if (!Array.isArray(manifest)) throw new Error("content manifest must be an array");

const layoutByOldPath = new Map();
for (const row of layoutLedger.rows) {
  if (layoutByOldPath.has(row.old_path)) throw new Error(`duplicate baseline path: ${row.old_path}`);
  layoutByOldPath.set(row.old_path, row.new_path);
}

const manifestByBaselinePath = new Map();
for (const item of manifest) {
  for (const baselinePath of item.baseline_paths) {
    if (manifestByBaselinePath.has(baselinePath)) {
      throw new Error(`baseline path has multiple final owners: ${baselinePath}`);
    }
    manifestByBaselinePath.set(baselinePath, item);
  }
}

function finalTargetForReadRoute(oldUrl) {
  const prefix = `${basePath}/read/`;
  const oldPath = decodeURIComponent(oldUrl).slice(prefix.length);
  const finalBaselinePath = layoutByOldPath.get(oldPath);
  const item = manifestByBaselinePath.get(finalBaselinePath);
  if (!finalBaselinePath || !item) {
    throw new Error(`legacy read route has no final manifest owner: ${oldUrl}`);
  }
  return targetUrl(basePath, item.canonical_route);
}

const aliases = new Map();
let canonicalPassthroughCount = 0;
for (const row of routeLedger.rows) {
  if (
    normalizedRoute(row.old_url) !== normalizedRoute(row.encoded_url)
    || normalizedRoute(row.trailing_slash_url) !== normalizedRoute(row.encoded_url)
  ) {
    throw new Error(`legacy route forms disagree: ${row.old_url}`);
  }

  const finalTarget = row.old_url.startsWith(`${basePath}/read/`)
    ? finalTargetForReadRoute(row.old_url)
    : row.canonical_target;
  if (normalizedRoute(finalTarget) !== normalizedRoute(row.canonical_target)) {
    throw new Error(`frozen and final canonical targets disagree for ${row.old_url}`);
  }

  const source = normalizedRoute(row.encoded_url);
  const target = normalizedRoute(finalTarget);
  if (source === target) {
    canonicalPassthroughCount += 1;
    continue;
  }
  if (aliases.has(source)) throw new Error(`duplicate normalized alias: ${row.old_url}`);
  aliases.set(source, targetUrl(basePath, target.slice(basePath.length) || "/"));
}

const removedRepoIndexUrl = `${basePath}/read/site/src/data/repo-index.json`;
const removedRepoIndexOwner = manifestByBaselinePath.get("site/src/data/repo-index.json");
if (
  !removedRepoIndexOwner
  || removedRepoIndexOwner.source_path !== "site/src/data/content-manifest.json"
) {
  throw new Error("removed repo-index does not map to the generated content manifest");
}
aliases.set(
  normalizedRoute(removedRepoIndexUrl),
  targetUrl(basePath, removedRepoIndexOwner.canonical_route),
);

const normalizedAliases = new Map(
  [...aliases].map(([source, target]) => [normalizedRoute(source), normalizedRoute(target)]),
);
validateAliasGraph(normalizedAliases);

for (const target of aliases.values()) {
  const artifact = artifactPath(target, basePath);
  try {
    readFileSync(artifact);
  } catch {
    throw new Error(`canonical target has no local artifact: ${target}`);
  }
}

for (const [source, target] of aliases) {
  const output = artifactPath(source, basePath);
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, aliasHtml(target));
}

const canonicalUrls = new Set([
  `https://cyjin-yl.github.io${basePath}/`,
  `https://cyjin-yl.github.io${basePath}/repo/`,
]);
for (const item of manifest) {
  canonicalUrls.add(`https://cyjin-yl.github.io${targetUrl(basePath, item.canonical_route)}`);
}
const sortedCanonicalUrls = [...canonicalUrls].sort((left, right) =>
  left < right ? -1 : left > right ? 1 : 0
);
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${sortedCanonicalUrls
  .map((url) => `<url><loc>${htmlEscape(url)}</loc></url>`)
  .join("")}</urlset>\n`;
writeFileSync(resolve(dist, "sitemap-0.xml"), sitemap);
copyFileSync(resolve(repo, "llms.txt"), resolve(dist, "llms.txt"));

console.log(
  `legacy aliases: ${routeLedger.rows.length}/${routeLedger.row_count} ledger routes; `
  + `${aliases.size} redirect documents; ${canonicalPassthroughCount} canonical passthroughs; `
  + `${sortedCanonicalUrls.length} sitemap URLs`,
);
