import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(site, "..");
const manifest = JSON.parse(
  readFileSync(resolve(site, "src", "data", "content-manifest.json"), "utf8"),
);

if (!Array.isArray(manifest)) throw new Error("content manifest must be an array");

const origin = "https://cyjin-yl.github.io";
const basePath = "/v100-perfs";

function encodedRoute(route) {
  if (typeof route !== "string" || !route.startsWith("/")) {
    throw new Error(`canonical route must be absolute: ${route}`);
  }
  return route.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}

function canonicalUrl(route) {
  const encoded = encodedRoute(route);
  return `${origin}${basePath}${encoded}${encoded.endsWith("/") ? "" : "/"}`;
}

const urls = manifest
  .map((item) => canonicalUrl(item.canonical_route))
  .sort((left, right) => left < right ? -1 : left > right ? 1 : 0);
const uniqueUrls = new Set(urls);
if (uniqueUrls.size !== urls.length) {
  throw new Error(`content manifest has ${urls.length - uniqueUrls.size} duplicate canonical URLs`);
}
if (urls.some((url) => !url.startsWith("https://"))) {
  throw new Error("llms.txt URLs must all use absolute HTTPS URLs");
}

const llms = [
  "# V100-32G FastLLM documentation",
  "",
  "> Canonical documentation, evidence, archive, and repository artifacts.",
  "> Generated from site/src/data/content-manifest.json; each canonical target appears once.",
  "",
  ...urls.map((url) => `- ${url}`),
  "",
].join("\n");

writeFileSync(resolve(repo, "llms.txt"), llms);
console.log(`llms.txt: ${urls.length} unique absolute HTTPS canonical URLs`);
