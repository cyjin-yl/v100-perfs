import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, posix, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(site, "..");
const outputPath = posix.join("site", "src", "data", "content-manifest.json");
const absoluteOutputPath = resolve(repo, outputPath);
const githubBlobRoot = "https://github.com/cyjin-yl/v100-perfs/blob/main";
const canonicalCurrentPaths = new Set([
  "docs/index.md",
  "docs/getting-started/overview.md",
  "docs/concepts/architecture/system-overview.md",
  "docs/concepts/reasoning/control-model.md",
  "docs/guides/openwebui/reasoning-controls.md",
  "docs/guides/client-compatibility/reasoning-fields.md",
  "docs/operations/current-state/production.md",
  "docs/operations/runbooks/proxy.md",
  "docs/reference/api/reasoning-controls.md",
  "docs/reference/models/qwen38/sampling-and-template.md",
  "docs/performance/methodology/index.md",
]);
const generatedSources = new Map([
  ["llms.txt", "site/scripts/gen-llms-txt.mjs"],
  [outputPath, "site/scripts/generate-repo-index.mjs"],
  ["site/package-lock.json", "site/package.json"],
]);
const navOrder = new Map([
  ["overview", 0],
  ["getting-started", 10],
  ["concepts", 20],
  ["guides", 30],
  ["operations", 40],
  ["reference", 50],
  ["performance", 60],
  ["research", 70],
  ["archive", 80],
]);

function git(args, options = {}) {
  const completed = spawnSync("git", ["-C", repo, ...args], {
    encoding: null,
    maxBuffer: 1024 * 1024 * 1024,
    ...options,
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    const detail = completed.stderr?.toString("utf8").trim();
    throw new Error(`git ${args.join(" ")} failed${detail ? `: ${detail}` : ""}`);
  }
  return completed.stdout;
}

function validateSourcePath(sourcePath) {
  const components = sourcePath.split("/");
  if (
    !sourcePath ||
    sourcePath.startsWith("/") ||
    sourcePath.includes("\\") ||
    components.some((component) => !component || component === "." || component === "..")
  ) {
    throw new Error(`unsafe Git-index path: ${JSON.stringify(sourcePath)}`);
  }
}

function readIndex() {
  const records = git(["ls-files", "-s", "-z"]).toString("utf8").split("\0");
  const entries = [];
  const exactPaths = new Set();
  const caseFoldedPaths = new Map();
  const normalizedPaths = new Map();
  for (const record of records) {
    if (!record) continue;
    const separator = record.indexOf("\t");
    if (separator < 0) throw new Error(`malformed Git-index record: ${record}`);
    const metadata = record.slice(0, separator).split(" ");
    const sourcePath = record.slice(separator + 1);
    if (metadata.length !== 3) throw new Error(`malformed Git-index metadata: ${record}`);
    const [gitMode, gitBlobOid, stage] = metadata;
    if (stage !== "0") throw new Error(`unmerged Git-index entry: ${sourcePath}`);
    if (!/^(?:100644|100755|120000|160000)$/.test(gitMode)) {
      throw new Error(`unsupported Git mode ${gitMode} for ${sourcePath}`);
    }
    if (!/^[0-9a-f]{40,64}$/.test(gitBlobOid)) {
      throw new Error(`invalid Git object id for ${sourcePath}`);
    }
    validateSourcePath(sourcePath);
    if (exactPaths.has(sourcePath)) throw new Error(`duplicate Git-index path: ${sourcePath}`);
    exactPaths.add(sourcePath);
    const folded = sourcePath.toLocaleLowerCase("en-US");
    const priorFolded = caseFoldedPaths.get(folded);
    if (priorFolded && priorFolded !== sourcePath) {
      throw new Error(`case-fold collision: ${priorFolded} and ${sourcePath}`);
    }
    caseFoldedPaths.set(folded, sourcePath);
    const normalized = sourcePath.normalize("NFC");
    const priorNormalized = normalizedPaths.get(normalized);
    if (priorNormalized && priorNormalized !== sourcePath) {
      throw new Error(`Unicode normalization collision: ${priorNormalized} and ${sourcePath}`);
    }
    normalizedPaths.set(normalized, sourcePath);
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
    if (headerEnd < 0) throw new Error(`missing Git object header for ${entry.sourcePath}`);
    const header = output.subarray(cursor, headerEnd).toString("ascii");
    const [objectId, objectType, sizeText, extra] = header.split(" ");
    if (extra || objectId !== entry.gitBlobOid || !/^(?:blob|commit)$/.test(objectType)) {
      throw new Error(`unexpected Git object header for ${entry.sourcePath}: ${header}`);
    }
    const size = Number.parseInt(sizeText, 10);
    if (!Number.isSafeInteger(size) || size < 0) {
      throw new Error(`invalid Git object size for ${entry.sourcePath}: ${sizeText}`);
    }
    const bodyStart = headerEnd + 1;
    const bodyEnd = bodyStart + size;
    if (bodyEnd >= output.length || output[bodyEnd] !== 0x0a) {
      throw new Error(`truncated Git object for ${entry.sourcePath}`);
    }
    objects.push({ objectType, bytes: output.subarray(bodyStart, bodyEnd) });
    cursor = bodyEnd + 1;
  }
  if (cursor !== output.length) throw new Error("unexpected trailing data from git cat-file");
  return objects;
}

function decodeText(bytes) {
  if (bytes.includes(0)) return null;
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}

function cleanInlineText(value) {
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/[`*_~[\]()>#+-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function titleFor(sourcePath, text) {
  if (sourcePath === outputPath) return "Content manifest";
  if (text) {
    const heading = text.split(/\r?\n/).find((line) => /^#\s+\S/.test(line));
    if (heading) return cleanInlineText(heading.replace(/^#\s+/, "")) || basename(sourcePath);
  }
  return basename(sourcePath);
}

function summaryFor(sourcePath, text, renderKind) {
  if (sourcePath === outputPath) {
    return "Deterministic inventory of every path in the final Git index.";
  }
  if (text && renderKind === "markdown") {
    const candidate = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) =>
        line &&
        !line.startsWith("#") &&
        !line.startsWith("```") &&
        line !== "---"
      );
    if (candidate) return cleanInlineText(candidate).slice(0, 240);
  }
  const kind = renderKind === "metadata" ? "binary or non-regular" : "tracked text";
  return `${kind} content from ${sourcePath}.`;
}

function modelGenerationFor(sourcePath) {
  const normalized = sourcePath.toLocaleLowerCase("en-US").replaceAll(/[^a-z0-9]+/g, "");
  if (normalized.includes("qwen38")) return "qwen3.8";
  if (normalized.includes("qwen36")) return "qwen3.6";
  if (normalized.includes("qwen35")) return "qwen3.5";
  return null;
}

function datedAtFor(sourcePath) {
  const match = sourcePath.match(/(?:^|\/)(20\d{2}-\d{2}-\d{2})(?:[^0-9]|$)/);
  return match ? `${match[1]}T00:00:00Z` : null;
}

function withoutMarkdownExtension(sourcePath) {
  return sourcePath.toLocaleLowerCase("en-US").endsWith(".md")
    ? sourcePath.slice(0, -3)
    : sourcePath;
}

function contentRoute(prefix, sourcePath) {
  let routePath = withoutMarkdownExtension(sourcePath);
  if (routePath === "index") routePath = "";
  if (routePath.endsWith("/index")) routePath = routePath.slice(0, -6);
  return routePath ? `${prefix}/${routePath}` : prefix;
}

function routeMetadata(sourcePath) {
  const repositoryRoute = `/repository/${sourcePath}`;
  if (sourcePath === "docs/index.md") {
    return { canonicalRoute: "/docs", repositoryRoute, routeKind: "docs" };
  }
  if (sourcePath.startsWith("docs/archive/")) {
    return {
      canonicalRoute: contentRoute("/archive", sourcePath.slice("docs/archive/".length)),
      repositoryRoute,
      routeKind: "archive",
    };
  }
  if (sourcePath.startsWith("docs/")) {
    return {
      canonicalRoute: contentRoute("/docs", sourcePath.slice("docs/".length)),
      repositoryRoute,
      routeKind: "docs",
    };
  }
  if (sourcePath.startsWith("evidence/")) {
    return {
      canonicalRoute: contentRoute("/evidence", sourcePath.slice("evidence/".length)),
      repositoryRoute,
      routeKind: "evidence",
    };
  }
  return { canonicalRoute: repositoryRoute, repositoryRoute, routeKind: "repository" };
}

function authorityMetadata(sourcePath) {
  if (sourcePath.startsWith("docs/archive/")) {
    return { status: "historical", authority: "archive" };
  }
  if (sourcePath.startsWith("evidence/")) {
    return { status: "supporting", authority: "evidence" };
  }
  if (canonicalCurrentPaths.has(sourcePath)) {
    return { status: "current", authority: "canonical" };
  }
  if (generatedSources.has(sourcePath)) {
    return { status: "generated", authority: "reference" };
  }
  return { status: "supporting", authority: "reference" };
}

function navMetadata(sourcePath, status, authority) {
  if (status !== "current" || authority !== "canonical") {
    return { navSection: null, navOrder: null };
  }
  if (sourcePath === "docs/index.md") {
    return { navSection: "overview", navOrder: navOrder.get("overview") };
  }
  const navSection = sourcePath.split("/")[1] ?? "overview";
  return { navSection, navOrder: navOrder.get(navSection) ?? null };
}

function encodeGitHubPath(sourcePath) {
  return sourcePath.split("/").map(encodeURIComponent).join("/");
}

function loadBaselineJoins() {
  const migrationsDirectory = resolve(repo, "operations", "migrations");
  const ledgers = readdirSync(migrationsDirectory)
    .filter((name) => name.endsWith("-repository-layout.json"))
    .sort();
  if (ledgers.length !== 1) {
    throw new Error(`expected one immutable baseline ledger, found ${ledgers.length}`);
  }
  const ledger = JSON.parse(readFileSync(resolve(migrationsDirectory, ledgers[0]), "utf8"));
  if (!ledger || !Array.isArray(ledger.rows) || ledger.row_count !== ledger.rows.length) {
    throw new Error("invalid immutable baseline ledger");
  }
  const joins = new Map();
  for (const row of ledger.rows) {
    if (!row || typeof row.new_path !== "string") {
      throw new Error("invalid immutable baseline ledger row");
    }
    const replacesGeneratedIndex = row.new_path === "site/src/data/repo-index.json";
    if (replacesGeneratedIndex && row.disposition !== "proven-generated-regenerate") {
      throw new Error("repo-index replacement is not approved for generated regeneration");
    }
    const finalPath = replacesGeneratedIndex ? outputPath : row.new_path;
    const baselinePaths = joins.get(finalPath) ?? [];
    baselinePaths.push(row.new_path);
    joins.set(finalPath, baselinePaths);
  }
  return { joins, rowCount: ledger.rows.length };
}

const baseline = loadBaselineJoins();

const entries = readIndex();
if (!entries.some((entry) => entry.sourcePath === outputPath)) {
  throw new Error(
    `${outputPath} must already be staged so the final Git index includes the generated manifest`
  );
}
const objects = readIndexObjects(entries);
const items = entries.map((entry, index) => {
  const object = objects[index];
  const isRegular = entry.gitMode === "100644" || entry.gitMode === "100755";
  const text = isRegular && object.objectType === "blob" ? decodeText(object.bytes) : null;
  const extension = extname(entry.sourcePath).toLocaleLowerCase("en-US");
  const renderKind = !isRegular || text === null
    ? "metadata"
    : extension === ".md"
      ? "markdown"
      : "text";
  const { canonicalRoute, repositoryRoute, routeKind } = routeMetadata(entry.sourcePath);
  const { status, authority } = authorityMetadata(entry.sourcePath);
  const { navSection, navOrder: itemNavOrder } = navMetadata(
    entry.sourcePath,
    status,
    authority,
  );
  const selfReference = entry.sourcePath === outputPath;
  return {
    id: `repo:${entry.sourcePath}`,
    source_path: entry.sourcePath,
    canonical_route: canonicalRoute,
    repository_route: repositoryRoute,
    route_kind: routeKind,
    title: titleFor(entry.sourcePath, text),
    summary: summaryFor(entry.sourcePath, text, renderKind),
    status,
    authority,
    model_generation: modelGenerationFor(entry.sourcePath),
    updated_at: datedAtFor(entry.sourcePath),
    nav_section: navSection,
    nav_order: itemNavOrder,
    aliases: [],
    baseline_paths: baseline.joins.get(entry.sourcePath) ?? [],
    git_mode: entry.gitMode,
    git_blob_oid: selfReference ? null : entry.gitBlobOid,
    byte_length: selfReference ? null : object.bytes.length,
    sha256: selfReference
      ? null
      : createHash("sha256").update(object.bytes).digest("hex"),
    metadata_scope: selfReference ? "self-reference-omitted" : "git-index-blob",
    render_kind: renderKind,
    media_type: renderKind === "markdown"
      ? "text/markdown"
      : renderKind === "text"
        ? "text/plain"
        : "application/octet-stream",
    extension,
    category: entry.sourcePath.includes("/")
      ? entry.sourcePath.split("/")[0]
      : "root",
    generated_from: generatedSources.get(entry.sourcePath) ?? null,
    download_url: `${githubBlobRoot}/${encodeGitHubPath(entry.sourcePath)}?raw=1`,
  };
});
const joinedBaselinePaths = items.flatMap((item) => item.baseline_paths);
if (
  joinedBaselinePaths.length !== baseline.rowCount ||
  new Set(joinedBaselinePaths).size !== joinedBaselinePaths.length
) {
  throw new Error(
    `baseline join is ${joinedBaselinePaths.length}/${baseline.rowCount}, expected one match per row`
  );
}

mkdirSync(dirname(absoluteOutputPath), { recursive: true });
const temporaryPath = `${absoluteOutputPath}.tmp`;
writeFileSync(temporaryPath, `${JSON.stringify(items, null, 2)}\n`);
renameSync(temporaryPath, absoluteOutputPath);

const counts = items.reduce((result, item) => {
  result[item.route_kind] = (result[item.route_kind] ?? 0) + 1;
  return result;
}, {});
console.log(
  `content-manifest.json: ${items.length} tracked paths; routes ${Object.entries(counts)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([kind, count]) => `${kind}=${count}`)
    .join(", ")}`,
);
