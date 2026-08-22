// 从仓库文档树生成 LLM 友好的 llms.txt (https://llmstxt.org 规范)
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = join(site, "..");

function walk(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (![".git", "node_modules", "runtime"].includes(name)) walk(p, out);
    } else if (name.endsWith(".md")) {
      out.push(p);
    }
  }
  return out;
}

let llms = "# V100-32G FastLLM 推理知识库\n\n";
llms += "> Qwen3.8-27B on Tesla V100-PCIE-32GB (SM70), FastLLM + thinking_proxy.\n";
llms += "> 单卡 PCIe 记录; 双卡 TP2 与傲腾 NVMe L3 为规划中路线。\n\n";

const sections = [
  { title: "Docs", dir: join(repo, "docs"), note: "叙述性文档: 架构/分析/运维/工具调用" },
  { title: "Results", dir: join(repo, "results"), note: "基准结果数据(MATRIX 汇总与 raw json)" },
  { title: "Repo Root", dir: repo, note: "README 与顶层文档" },
];

for (const sec of sections) {
  const files = walk(sec.dir).filter((p) => !p.includes("/site/") && !p.endsWith("SUMMARY.md") || p.endsWith("SUMMARY.md"));
  const uniq = [...new Set(files)];
  if (!uniq.length) continue;
  llms += `## ${sec.title}\n\n${sec.note}\n\n`;
  for (const p of uniq) {
    const rel = p.slice(repo.length + 1);
    let desc = "";
    try {
      const head = readFileSync(p, "utf8").split("\n").slice(0, 3).join(" ").replace(/[#*>`]/g, "").trim();
      if (head) desc = ": " + head.slice(0, 100);
    } catch {}
    llms += `- [${rel}](${rel})${desc}\n`;
  }
  llms += "\n";
}
llms += "## Optional\n\n- [Full EXPERIENCE chronicle](docs/EXPERIENCE.md): 完整故障排查编年史(最长)\n";

writeFileSync(join(repo, "llms.txt"), llms);
console.log("llms.txt written,", llms.split("\n").length, "lines");
