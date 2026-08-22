import { useMemo, useState } from "react";
import * as React from "react";
import { createReactAdapter } from "@proto.ui/adapter-react";
import { shadcnButton } from "@proto.ui/prototypes-shadcn/button";

const ProtoButton = createReactAdapter({ React })(shadcnButton);
type Item = { path: string; title: string; category: string; ext: string; bytes: number };

/** Full repository reader: every indexed text/document file is reachable here. */
export default function RepoExplorer({ items }: { items: Item[] }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const categories = useMemo(
    () => Array.from(new Set(items.map((item) => item.category))).sort((a, b) => a.localeCompare(b)),
    [items],
  );
  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    const compactQuery = normalizedQuery.replace(/[\s._-]+/g, "");
    return items.filter((item) => {
      const matchesCategory = category === "all" || item.category === category;
      const searchable = `${item.path} ${item.title} ${item.category} ${item.ext}`.toLocaleLowerCase();
      const matchesQuery = !normalizedQuery ||
        searchable.includes(normalizedQuery) ||
        (compactQuery.length > 0 && searchable.replace(/[\s._-]+/g, "").includes(compactQuery));
      return matchesCategory && matchesQuery;
    });
  }, [items, query, category]);
  const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");

  return <div>
    <div style={{display:"flex", gap:8, flexWrap:"wrap", alignItems:"center", margin:"16px 0"}}>
      <input
        aria-label="搜索仓库文件"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="按名称、路径、类别或扩展名搜索…"
        style={{flex:"1 1 280px", background:"hsl(var(--card))", color:"hsl(var(--foreground))", border:"1px solid hsl(var(--border))", borderRadius:"var(--radius)", padding:"8px 10px"}}
      />
      <select
        aria-label="按仓库目录筛选"
        value={category}
        onChange={(event) => setCategory(event.target.value)}
        style={{background:"hsl(var(--card))", color:"hsl(var(--foreground))", border:"1px solid hsl(var(--border))", borderRadius:"var(--radius)", padding:"8px 10px"}}
      >
        <option value="all">全部目录类别</option>
        {categories.map((itemCategory) => (
          <option key={itemCategory} value={itemCategory}>{itemCategory}</option>
        ))}
      </select>
      {(query || category !== "all") && (
        <ProtoButton
          size="sm"
          variant="outline"
          onClick={() => {
            setQuery("");
            setCategory("all");
          }}
        >
          清除筛选
        </ProtoButton>
      )}
    </div>
    <div style={{color:"hsl(var(--muted-foreground))", fontSize:12, marginBottom:8}}>
      显示 {filtered.length} / {items.length} 个可读文件 · 类别直接来自仓库目录
    </div>
    <div style={{display:"grid", gap:6}}>
      {filtered.map((item) => (
        <a
          key={item.path}
          href={`${base}read/${item.path.split("/").map(encodeURIComponent).join("/")}`}
          style={{display:"block", border:"1px solid hsl(var(--border))", borderRadius:"var(--radius)", padding:"9px 11px", background:"hsl(var(--card))"}}
        >
          <strong>{item.title}</strong><br />
          <code style={{fontSize:12, color:"hsl(var(--muted-foreground))"}}>{item.path}</code>
          <span style={{display:"block", marginTop:3, fontSize:11, color:"hsl(var(--muted-foreground))"}}>
            {item.category} · {item.ext.slice(1).toUpperCase()} · {(item.bytes / 1024).toFixed(1)} KiB
          </span>
        </a>
      ))}
      {filtered.length === 0 && (
        <p style={{color:"hsl(var(--muted-foreground))"}}>没有匹配的仓库文件。</p>
      )}
    </div>
  </div>;
}
