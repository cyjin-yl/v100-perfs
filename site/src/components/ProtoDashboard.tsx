type CanonicalItem = {
  title: string;
  summary: string;
  route: string;
  section: string;
};

type ProtoDashboardProps = {
  items: CanonicalItem[];
};

const sections = [
  { id: "getting-started", label: "Start" },
  { id: "concepts", label: "Understand" },
  { id: "guides", label: "Configure" },
  { id: "operations", label: "Operate" },
  { id: "reference", label: "Verify" },
  { id: "performance", label: "Measure" },
] as const;

export default function ProtoDashboard({ items }: ProtoDashboardProps) {
  return (
    <section className="proto-dashboard" aria-labelledby="dashboard-title">
      <div className="dashboard-heading">
        <div>
          <p className="eyebrow">Canonical routes</p>
          <h2 id="dashboard-title">Move through the system by intent.</h2>
        </div>
        <p className="section-copy">
          Titles, summaries, and destinations come from the generated content manifest.
        </p>
      </div>

      <div className="route-board">
        {sections.map((section, index) => {
          const sectionItems = items.filter((item) => item.section === section.id);
          if (sectionItems.length === 0) return null;
          return (
            <section key={section.id} className="route-lane" aria-labelledby={`route-${section.id}`}>
              <div className="route-lane-label">
                <span>{String(index + 1).padStart(2, "0")}</span>
                <h3 id={`route-${section.id}`}>{section.label}</h3>
              </div>
              <div className="route-links">
                {sectionItems.map((item) => (
                  <a key={item.route} href={item.route}>
                    <strong>{item.title}</strong>
                    <span>{item.summary}</span>
                  </a>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </section>
  );
}
