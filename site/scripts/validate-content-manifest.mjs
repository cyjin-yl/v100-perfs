import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(site, "..");
const manifestPath = "site/src/data/content-manifest.json";
const absoluteManifestPath = resolve(repo, manifestPath);
const requiredFields = new Set([
  "id",
  "source_path",
  "canonical_route",
  "repository_route",
  "route_kind",
  "title",
  "summary",
  "status",
  "authority",
  "model_generation",
  "updated_at",
  "nav_section",
  "nav_order",
  "aliases",
  "baseline_paths",
  "git_mode",
  "git_blob_oid",
  "byte_length",
  "sha256",
  "metadata_scope",
  "render_kind",
  "media_type",
  "extension",
  "category",
  "generated_from",
  "download_url",
]);
const statuses = new Set(["current", "supporting", "historical", "generated"]);
const authorities = new Set(["canonical", "reference", "evidence", "archive"]);
const routeKinds = new Set(["docs", "evidence", "repository", "archive"]);
const renderKinds = new Set(["markdown", "text", "metadata"]);

function fail(message) {
  throw new Error(`content manifest validation failed: ${message}`);
}

function git(args, options = {}) {
  const completed = spawnSync("git", ["-C", repo, ...args], {
    encoding: null,
    maxBuffer: 1024 * 1024 * 1024,
    ...options,
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    const detail = completed.stderr?.toString("utf8").trim();
    fail(`git ${args.join(" ")} failed${detail ? `: ${detail}` : ""}`);
  }
  return completed.stdout;
}

function readIndex() {
  const entries = [];
  for (const record of git(["ls-files", "-s", "-z"]).toString("utf8").split("\0")) {
    if (!record) continue;
    const separator = record.indexOf("\t");
    if (separator < 0) fail(`malformed Git-index record: ${record}`);
    const [gitMode, gitBlobOid, stage, extra] = record.slice(0, separator).split(" ");
    const sourcePath = record.slice(separator + 1);
    if (extra || !gitMode || !gitBlobOid || stage !== "0") {
      fail(`unmerged or malformed Git-index entry: ${sourcePath}`);
    }
    entries.push({ sourcePath, gitMode, gitBlobOid });
  }
  return entries.sort((left, right) =>
    left.sourcePath < right.sourcePath ? -1 : left.sourcePath > right.sourcePath ? 1 : 0
  );
}

function readIndexObjects(entries) {
  const input = Buffer.from(entries.map((entry) => `${entry.gitBlobOid}\n`).join(""));
  const output = git(["cat-file", "--batch"], { input });
  const objects = [];
  let cursor = 0;
  for (const entry of entries) {
    const headerEnd = output.indexOf(0x0a, cursor);
    if (headerEnd < 0) fail(`missing Git object header for ${entry.sourcePath}`);
    const header = output.subarray(cursor, headerEnd).toString("ascii");
    const [objectId, objectType, sizeText, extra] = header.split(" ");
    const size = Number.parseInt(sizeText, 10);
    if (
      extra ||
      objectId !== entry.gitBlobOid ||
      !/^(?:blob|commit)$/.test(objectType) ||
      !Number.isSafeInteger(size) ||
      size < 0
    ) {
      fail(`unexpected Git object header for ${entry.sourcePath}: ${header}`);
    }
    const bodyStart = headerEnd + 1;
    const bodyEnd = bodyStart + size;
    if (bodyEnd >= output.length || output[bodyEnd] !== 0x0a) {
      fail(`truncated Git object for ${entry.sourcePath}`);
    }
    objects.push(output.subarray(bodyStart, bodyEnd));
    cursor = bodyEnd + 1;
  }
  if (cursor !== output.length) fail("unexpected trailing data from git cat-file");
  return objects;
}

function safePath(sourcePath) {
  if (typeof sourcePath !== "string" || !sourcePath || sourcePath.startsWith("/")) return false;
  if (sourcePath.includes("\\")) return false;
  return sourcePath.split("/").every(
    (component) => component && component !== "." && component !== "..",
  );
}

function safeRoute(route) {
  if (typeof route !== "string" || !route.startsWith("/") || route === "/") return false;
  if (/[?#\\]/.test(route) || route.includes("//")) return false;
  return route.slice(1).split("/").every(
    (component) => component && component !== "." && component !== "..",
  );
}

function addUnique(seen, value, label) {
  if (seen.has(value)) fail(`duplicate ${label}: ${value}`);
  seen.add(value);
}

function validateScalarMetadata(item, index) {
  for (const field of requiredFields) {
    if (!Object.hasOwn(item, field)) fail(`item ${index} is missing ${field}`);
  }
  if (typeof item.id !== "string" || item.id !== `repo:${item.source_path}`) {
    fail(`invalid id for ${item.source_path}`);
  }
  for (const field of ["title", "summary", "category", "download_url"]) {
    if (typeof item[field] !== "string" || !item[field]) {
      fail(`invalid ${field} for ${item.source_path}`);
    }
  }
  if (!statuses.has(item.status)) fail(`invalid status for ${item.source_path}`);
  if (!authorities.has(item.authority)) fail(`invalid authority for ${item.source_path}`);
  if (!routeKinds.has(item.route_kind)) fail(`invalid route_kind for ${item.source_path}`);
  if (!renderKinds.has(item.render_kind)) fail(`invalid render_kind for ${item.source_path}`);
  if (!Array.isArray(item.aliases) || item.aliases.some((alias) => typeof alias !== "string")) {
    fail(`invalid aliases for ${item.source_path}`);
  }
  if (item.aliases.length !== 0) {
    fail(`Task 5 manifest must not contain legacy aliases: ${item.source_path}`);
  }
  if (
    !Array.isArray(item.baseline_paths) ||
    item.baseline_paths.some((baselinePath) => !safePath(baselinePath))
  ) {
    fail(`invalid baseline_paths for ${item.source_path}`);
  }
  for (const field of ["model_generation", "updated_at", "nav_section", "generated_from"]) {
    if (item[field] !== null && typeof item[field] !== "string") {
      fail(`invalid nullable string ${field} for ${item.source_path}`);
    }
  }
  if (item.nav_order !== null && !Number.isSafeInteger(item.nav_order)) {
    fail(`invalid nav_order for ${item.source_path}`);
  }
  if (item.status === "current" && item.authority !== "canonical") {
    fail(`only canonical content may be current: ${item.source_path}`);
  }
  if ((item.nav_section !== null || item.nav_order !== null) &&
      !(item.status === "current" && item.authority === "canonical")) {
    fail(`only current canonical content may be navigable: ${item.source_path}`);
  }
}

let manifest;
try {
  manifest = JSON.parse(readFileSync(absoluteManifestPath, "utf8"));
} catch (error) {
  fail(`cannot read ${manifestPath}: ${error.message}`);
}
if (!Array.isArray(manifest)) fail("top level must be an array");

const entries = readIndex();
const objects = readIndexObjects(entries);
if (manifest.length !== entries.length) {
  fail(`coverage count is ${manifest.length}, expected ${entries.length}`);
}
const paths = new Set();
const caseFoldedPaths = new Set();
const normalizedPaths = new Set();
const actualRoutes = new Map();
const baselineClaims = new Map();
const routeCounts = { docs: 0, evidence: 0, repository: 0, archive: 0 };
for (let index = 0; index < manifest.length; index += 1) {
  const item = manifest[index];
  const entry = entries[index];
  if (!item || typeof item !== "object" || Array.isArray(item)) fail(`item ${index} is not an object`);
  validateScalarMetadata(item, index);
  if (!safePath(item.source_path)) fail(`unsafe source_path: ${item.source_path}`);
  if (item.source_path !== entry.sourcePath) {
    fail(`manifest is not sorted Git-index coverage at item ${index}: ${item.source_path}`);
  }
  addUnique(paths, item.source_path, "source_path");
  addUnique(caseFoldedPaths, item.source_path.toLocaleLowerCase("en-US"), "case-folded path");
  addUnique(normalizedPaths, item.source_path.normalize("NFC"), "NFC path");
  if (!safeRoute(item.repository_route) ||
      item.repository_route !== `/repository/${item.source_path}`) {
    fail(`invalid repository route for ${item.source_path}`);
  }
  if (!safeRoute(item.canonical_route)) fail(`unsafe canonical route for ${item.source_path}`);
  const priorRepositoryOwner = actualRoutes.get(item.repository_route);
  if (priorRepositoryOwner) {
    fail(`route collision ${item.repository_route}: ${priorRepositoryOwner} and ${item.source_path}`);
  }
  actualRoutes.set(item.repository_route, item.source_path);
  if (item.canonical_route !== item.repository_route) {
    const expectedPrefix = item.route_kind === "docs"
      ? "/docs"
      : item.route_kind === "evidence"
        ? "/evidence"
        : item.route_kind === "archive"
          ? "/archive"
          : null;
    if (!expectedPrefix || !(
      item.canonical_route === expectedPrefix || item.canonical_route.startsWith(`${expectedPrefix}/`)
    )) {
      fail(`canonical route does not match route_kind for ${item.source_path}`);
    }
    const priorCanonicalOwner = actualRoutes.get(item.canonical_route);
    if (priorCanonicalOwner) {
      fail(`route collision ${item.canonical_route}: ${priorCanonicalOwner} and ${item.source_path}`);
    }
    actualRoutes.set(item.canonical_route, item.source_path);
  } else if (item.route_kind !== "repository") {
    fail(`non-repository item lacks a canonical route: ${item.source_path}`);
  }
  routeCounts[item.route_kind] += 1;

  if (item.source_path === manifestPath) {
    if (
      item.metadata_scope !== "self-reference-omitted" ||
      item.git_blob_oid !== null ||
      item.byte_length !== null ||
      item.sha256 !== null
    ) {
      fail("content-manifest self item must use the approved null metadata exception");
    }
  } else {
    const bytes = objects[index];
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    if (
      item.metadata_scope !== "git-index-blob" ||
      item.git_mode !== entry.gitMode ||
      item.git_blob_oid !== entry.gitBlobOid ||
      item.byte_length !== bytes.length ||
      item.sha256 !== sha256
    ) {
      fail(`Git-index metadata mismatch for ${item.source_path}`);
    }
  }
  const expectedExtension = extname(item.source_path).toLocaleLowerCase("en-US");
  if (item.extension !== expectedExtension) fail(`extension mismatch for ${item.source_path}`);
  const expectedDownloadUrl = `https://github.com/cyjin-yl/v100-perfs/blob/main/${
    item.source_path.split("/").map(encodeURIComponent).join("/")
  }?raw=1`;
  if (item.download_url !== expectedDownloadUrl) {
    fail(`download URL mismatch for ${item.source_path}`);
  }
  for (const baselinePath of item.baseline_paths) {
    const priorOwner = baselineClaims.get(baselinePath);
    if (priorOwner) {
      fail(`duplicate baseline claim ${baselinePath}: ${priorOwner} and ${item.source_path}`);
    }
    baselineClaims.set(baselinePath, item.source_path);
  }
}
if (!paths.has(manifestPath)) fail("manifest does not cover itself");

const migrationsDirectory = resolve(repo, "operations", "migrations");
const ledgers = readdirSync(migrationsDirectory)
  .filter((name) => name.endsWith("-repository-layout.json"))
  .sort();
if (ledgers.length !== 1) fail(`expected one immutable baseline ledger, found ${ledgers.length}`);
const ledger = JSON.parse(readFileSync(resolve(migrationsDirectory, ledgers[0]), "utf8"));
if (!ledger || !Array.isArray(ledger.rows) || ledger.row_count !== ledger.rows.length) {
  fail("invalid immutable baseline ledger");
}
const ledgerRows = new Map(ledger.rows.map((row) => [row.new_path, row]));
if (ledgerRows.size !== ledger.rows.length) fail("duplicate immutable baseline destinations");
if (
  baselineClaims.size !== ledger.rows.length ||
  [...baselineClaims.keys()].some((baselinePath) => !ledgerRows.has(baselinePath))
) {
  fail(`baseline join is ${baselineClaims.size}/${ledger.rows.length}, expected one claim per destination`);
}
for (const row of ledger.rows) {
  const owner = baselineClaims.get(row.new_path);
  if (!owner) fail(`missing baseline claim: ${row.new_path}`);
  if (owner !== row.new_path) {
    if (row.disposition !== "proven-generated-regenerate") {
      fail(`non-generated baseline row claimed by ${owner}: ${row.new_path}`);
    }
    if (
      row.new_path !== "site/src/data/repo-index.json" ||
      owner !== manifestPath
    ) {
      fail(`unapproved generated baseline replacement: ${row.new_path} -> ${owner}`);
    }
  }
}

console.log(
  `content manifest valid: tracked=${manifest.length}; baseline_join=${baselineClaims.size}/${ledger.rows.length}; ` +
  `routes docs=${routeCounts.docs}, evidence=${routeCounts.evidence}, ` +
  `repository=${routeCounts.repository}, archive=${routeCounts.archive}; ` +
  `total_static_routes=${actualRoutes.size}`,
);
