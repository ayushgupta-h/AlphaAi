/** Skeleton shimmer block */
export function Skeleton({ className = "" }) {
  return (
    <div
      className={`shimmer rounded-lg ${className}`}
      style={{ minHeight: "1.2rem" }}
    />
  );
}

/** Inline error banner */
export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div
      className="rounded-xl border px-4 py-3 text-sm fade-in"
      style={{
        borderColor: "rgba(252,129,74,0.3)",
        background: "rgba(252,129,74,0.08)",
        color: "#fc8150",
      }}
    >
      ⚠ {message}
    </div>
  );
}

/** Small "LIVE" badge */
export function LiveBadge() {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold"
      style={{
        background: "rgba(72,187,120,0.15)",
        color: "#68d391",
        border: "1px solid rgba(72,187,120,0.25)",
      }}
    >
      <span className="relative flex h-2 w-2">
        <span
          className="absolute inline-flex h-full w-full rounded-full opacity-75"
          style={{ background: "#48bb78", animation: "pulse 1.5s ease-out infinite" }}
        />
        <span
          className="relative inline-flex h-2 w-2 rounded-full"
          style={{ background: "#68d391" }}
        />
      </span>
      LIVE
    </span>
  );
}

/** Metric card */
export function MetricCard({ label, value, sub, accent = "#63b3ed" }) {
  return (
    <div
      className="glass-card rounded-2xl p-5 flex flex-col gap-1 transition-all duration-300 hover:-translate-y-1 fade-in"
      style={{ "--hover-shadow": `0 8px 30px rgba(99,179,237,0.12)` }}
    >
      <span
        className="text-xs font-semibold uppercase tracking-widest"
        style={{ color: "var(--text-secondary)" }}
      >
        {label}
      </span>
      <span
        className="text-2xl font-bold font-mono"
        style={{ color: accent }}
      >
        {value}
      </span>
      {sub && (
        <span className="text-xs" style={{ color: "var(--text-muted)" }}>
          {sub}
        </span>
      )}
    </div>
  );
}

/** Section heading */
export function SectionTitle({ children }) {
  return (
    <h2
      className="text-base font-semibold mb-4"
      style={{
        color: "var(--text-primary)",
        borderBottom: "1px solid var(--border)",
        paddingBottom: "0.5rem",
      }}
    >
      {children}
    </h2>
  );
}
