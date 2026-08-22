import { lstatSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(site, "..");
const skipDirs = {
  ".git": true,
  ".astro": true,
  ".cache": true,
  ".mypy_cache": true,
  ".pytest_cache": true,
  ".tox": true,
  "1Cat-vLLM": true,
  "1Cat-vLLM-1.2.0": true,
  "__pycache__": true,
  "build": true,
  "dist": true,
  "exllamav3": true,
  "ik_llama.cpp": true,
  "llama.cpp": true,
  "llama.cpp-turboquant": true,
  "logs": true,
  "models": true,
  "node_modules": true,
  "runtime": true,
  "runtime-artifacts": true,
  "server": true,
};
const allowed = {
  ".astro": true,
  ".c": true,
  ".cc": true,
  ".cmake": true,
  ".cpp": true,
  ".css": true,
  ".csv": true,
  ".cu": true,
  ".cuh": true,
  ".h": true,
  ".hpp": true,
  ".ini": true,
  ".jinja": true,
  ".js": true,
  ".json": true,
  ".jsx": true,
  ".md": true,
  ".mjs": true,
  ".patch": true,
  ".py": true,
  ".sh": true,
  ".sse": true,
  ".toml": true,
  ".ts": true,
  ".tsv": true,
  ".tsx": true,
  ".txt": true,
  ".yaml": true,
  ".yml": true,
};
const maxBytes = 1024 * 1024;

function titleFor(path, text) {
  const heading = text.split(/\r?\n/).find((line) => /^#\s+/.test(line));
  return heading ? heading.replace(/^#\s+/, "").trim() : basename(path);
}
function categoryFor(path) {
  const parts = path.split("/");
  if (parts.length === 1) return "root";
  if (parts[0] === "docs" && parts.length > 2) return parts.slice(0, 2).join("/");
  return parts[0];
}

function shouldSkipDirectory(name) {
  return Object.hasOwn(skipDirs, name) || name.startsWith(".venv");
}

function isSensitiveFile(name) {
  const lower = name.toLowerCase();
  return lower.startsWith(".env") ||
    /(?:^|[._-])(?:credentials?|secrets?)(?:[._-]|$)/.test(lower) ||
    /\.(?:key|p12|pem|pfx)$/.test(lower);
}

function isInsideRepo(path) {
  return path.startsWith(`${repo}${sep}`);
}
function walk(dir, out = []) {
  for (const name of readdirSync(dir).sort()) {
    const path = join(dir, name);
    let stat;
    try {
      stat = lstatSync(path);
    } catch {
      continue;
    }

    if (stat.isSymbolicLink()) continue;
    if (stat.isDirectory()) {
      if (!shouldSkipDirectory(name)) walk(path, out);
      continue;
    }
    if (!stat.isFile() || isSensitiveFile(name) || stat.size > maxBytes) continue;

    const absolute = resolve(path);
    const ext = extname(name).toLowerCase();
    if (!isInsideRepo(absolute) || !Object.hasOwn(allowed, ext)) continue;
    const repoPath = relative(repo, absolute).split(sep).join("/");
    if (repoPath === "site/src/data/repo-index.json") continue;

    const buffer = readFileSync(absolute);
    if (buffer.includes(0)) continue;
    const text = buffer.toString("utf8");

    out.push({
      path: repoPath,
      title: titleFor(repoPath, text),
      category: categoryFor(repoPath),
      ext,
      bytes: stat.size,
    });
  }
  return out;
}
const items = walk(repo).sort((a, b) => a.path.localeCompare(b.path));
const data = join(site, "src", "data");
mkdirSync(data, { recursive: true });
writeFileSync(join(data, "repo-index.json"), JSON.stringify(items, null, 2) + "\n");
console.log(`repo-index.json: ${items.length} readable files`);
