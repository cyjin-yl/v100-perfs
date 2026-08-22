type SnapshotField = {
  label: string;
  value: string;
  source: string;
};

type StatusCardProps = {
  capturedAt: string;
  productionHref: string;
  fields: SnapshotField[];
};

export default function StatusCard({
  capturedAt,
  productionHref,
  fields,
}: StatusCardProps) {
  const captureLabel = new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: "UTC",
  }).format(new Date(capturedAt));

  return (
    <section className="status-card" aria-labelledby="snapshot-status-title">
      <div className="status-card-heading">
        <div>
          <p className="eyebrow">Immutable deployment snapshot</p>
          <h2 id="snapshot-status-title">Observed once. Provenanced field by field.</h2>
        </div>
        <p className="capture-time">
          Captured <time dateTime={capturedAt}>{captureLabel} UTC</time>
        </p>
      </div>
      <dl className="status-fields">
        {fields.map((field) => (
          <div key={field.label}>
            <dt>{field.label}</dt>
            <dd>{field.value}</dd>
            <dd className="field-source">{field.source}</dd>
          </div>
        ))}
      </dl>
      <div className="status-boundary">
        <p>
          This is not a live poll. The capture sent no generation request, changed no state,
          and did not deploy the tracked reasoning-control update.
        </p>
        <a className="text-link" href={productionHref}>
          Inspect the production authority <span aria-hidden="true">→</span>
        </a>
      </div>
    </section>
  );
}
