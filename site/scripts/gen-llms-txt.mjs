// Generate the legacy llms.txt content from tracked Markdown only.
// Task 7 replaces the content model; this generator only makes the current
// artifact deterministic and prevents ignored working-tree files from leaking.
import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(site, "..");

function git(args) {
  const completed = spawnSync("git", ["-C", repo, ...args], {
    encoding: null,
    maxBuffer: 1024 * 1024 * 1024,
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    const detail = completed.stderr?.toString("utf8").trim();
    throw new Error(`git ${args.join(" ")} failed${detail ? `: ${detail}` : ""}`);
  }
  return completed.stdout;
}

const trackedMarkdown = git(["ls-files", "-z"])
  .toString("utf8")
  .split("\0")
  .filter((path) => path.endsWith(".md"))
  .sort((left, right) => left < right ? -1 : left > right ? 1 : 0);

function indexedText(path) {
  return git(["cat-file", "blob", `:${path}`]).toString("utf8");
}

let llms = "# V100-32G FastLLM 推理知识库\n\n";
llms += "> Qwen3.8-27B on Tesla V100-PCIE-32GB (SM70), FastLLM + thinking_proxy.\n";
llms += "> 单卡 PCIe 记录; 双卡 TP2 与傲腾 NVMe L3 为规划中路线。\n\n";

const sections = [
  {
    title: "Docs",
    files: trackedMarkdown.filter((path) => path.startsWith("docs/")),
    note: "叙述性文档: 架构/分析/运维/工具调用",
  },
  {
    title: "Results",
    files: trackedMarkdown.filter((path) => path.startsWith("evidence/results/")),
    note: "基准结果数据(MATRIX 汇总与 raw json)",
  },
  {
    title: "Repo Root",
    files: trackedMarkdown.filter((path) => !path.startsWith("site/")),
    note: "README 与顶层文档",
  },
];

for (const section of sections) {
  if (!section.files.length) continue;
  llms += `## ${section.title}\n\n${section.note}\n\n`;
  for (const path of section.files) {
    const head = indexedText(path)
      .split("\n")
      .slice(0, 3)
      .join(" ")
      .replace(/[#*>`]/g, "")
      .trim();
    const description = head ? `: ${head.slice(0, 100)}` : "";
    llms += `- [${path}](${path})${description}\n`;
  }
  llms += "\n";
}
llms += "## Optional\n\n- [Full EXPERIENCE chronicle](docs/archive/chronicles/v100-fastllm-experience.md): 完整故障排查编年史(最长)\n";

writeFileSync(resolve(repo, "llms.txt"), llms);
console.log("llms.txt written,", llms.split("\n").length, "lines");
