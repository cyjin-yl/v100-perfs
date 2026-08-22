import { useMemo, useState } from "react";

type Item = {
  source_path: string;
  title: string;
  summary: string;
  category: string;
  extension: string;
  byte_length: number | null;
  canonical_route: string;
  repository_route: string;
  route_kind: "docs" | "evidence" | "archive" | "repository";
  status: string;
  authority: string;
};

function deployedRoute(route: string) {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  const encoded = route
    .split("/")
    .map((part, index) => index === 0 ? "" : encodeURIComponent(part))
    .join("/");
  return `${base}${encoded}`;
}

/** Searchable reading surface backed by final Git-index manifest rows. */
export default function RepoExplorer({ items }: { items: Item[] }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [authority, setAuthority] = useState("all");
  const categories = useMemo(
    () => Array.from(new Set(items.map((item) => item.category))).sort((a, b) => a.localeCompare(b)),
    [items],
  );
  const authorities = useMemo(
    () => Array.from(new Set(items.map((item) => item.authority))).sort((a, b) => a.localeCompare(b)),
    [items],
  );
  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    const compactQuery = normalizedQuery.replace(/[\s._-]+/g, "");
    return items.filter((item) => {
      const matchesCategory = category === "all" || item.category === category;
      const matchesAuthority = authority === "all" || item.authority === authority;
      const searchable = [
        item.source_path,
        item.title,
        item.summary,
        item.category,
        item.extension,
        item.status,
        item.authority,
      ].join(" ").toLocaleLowerCase();
      const matchesQuery = !normalizedQuery ||
        searchable.includes(normalizedQuery) ||
        (compactQuery.length > 0 && searchable.replace(/[\s._-]+/g, "").includes(compactQuery));
      return matchesCategory && matchesAuthority && matchesQuery;
    });
  }, [items, query, category, authority]);
  const hasFilters = query || category !== "all" || authority !== "all";

  return (
    <section className="repo-explorer" aria-label="Repository manifest search">
      <div className="repo-filters">
        <label>
          <span>Search</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Title, path, summary, status…"
          />
        </label>
        <label>
          <span>Directory</span>
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            <option value="all">All directories</option>
            {categories.map((itemCategory) => (
              <option key={itemCategory} value={itemCategory}>{itemCategory}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Authority</span>
          <select value={authority} onChange={(event) => setAuthority(event.target.value)}>
            <option value="all">All authorities</option>
            {authorities.map((itemAuthority) => (
              <option key={itemAuthority} value={itemAuthority}>{itemAuthority}</option>
            ))}
          </select>
        </label>
        {hasFilters && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setCategory("all");
              setAuthority("all");
            }}
          >
            Clear
          </button>
        )}
      </div>
      <p className="repo-count">
        Showing {filtered.length} of {items.length} final Git-index paths
      </p>
      <div className="repo-results">
        {filtered.map((item) => {
          const readingRoute = item.route_kind === "repository"
            ? item.repository_route
            : item.canonical_route;
          return (
            <a key={item.source_path} href={deployedRoute(readingRoute)}>
              <span className="repo-result-meta">{item.authority} / {item.status}</span>
              <strong>{item.title}</strong>
              <span>{item.summary}</span>
              <code>{item.source_path}</code>
              <small>
                {item.route_kind} · {item.extension.slice(1).toUpperCase() || "NO EXT"} · {
                  item.byte_length === null ? "manifest self-reference" : `${item.byte_length} bytes`
                }
              </small>
            </a>
          );
        })}
        {filtered.length === 0 && <p className="repo-empty">No manifest rows match these filters.</p>}
      </div>
    </section>
  );
}
