import { useFetch } from "../hooks";
import { api } from "../api";
import { formatPrice, formatPct } from "../utils";
import { Skeleton, ErrorBanner, MetricCard, SectionTitle } from "./ui";
import { Target, Ruler, BarChart2, AlertCircle } from "lucide-react";

function QualityBadge({ quality }) {
  const styles = {
    Excellent: { bg: "rgba(72,187,120,0.15)", color: "#68d391", border: "rgba(72,187,120,0.3)" },
    Good: { bg: "rgba(99,179,237,0.15)", color: "#63b3ed", border: "rgba(99,179,237,0.3)" },
    Fair: { bg: "rgba(237,137,54,0.15)", color: "#ed8936", border: "rgba(237,137,54,0.3)" },
    Poor: { bg: "rgba(252,129,74,0.15)", color: "#fc8150", border: "rgba(252,129,74,0.3)" },
  };
  const s = styles[quality] ?? styles.Fair;
  return (
    <span
      className="text-xs font-semibold px-2.5 py-0.5 rounded-full"
      style={{ background: s.bg, color: s.color, border: `1px solid ${s.border}` }}
    >
      {quality}
    </span>
  );
}

export function MetricsPanel() {
  const { data, loading, error } = useFetch(api.metrics);

  return (
    <div>
      <SectionTitle>📊 Backtest Evaluation Metrics</SectionTitle>

      {error && <ErrorBanner message={error} />}

      {loading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : data ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <MetricCard
              label="Coverage"
              value={data.coverage.toFixed(4)}
              sub={`${formatPct(data.coverage)} of actuals inside interval`}
              accent="#63b3ed"
            />
            <MetricCard
              label="Average Width"
              value={formatPrice(data.average_width)}
              sub="Mean interval width"
              accent="#7c3aed"
            />
            <MetricCard
              label="Winkler Score"
              value={formatPrice(data.winkler_score)}
              sub="Lower = better quality"
              accent="#ec4899"
            />
            <MetricCard
              label="Violations"
              value={`${data.violations}`}
              sub={`of ${data.total_predictions} predictions`}
              accent={data.violations > 50 ? "#fc8150" : "#68d391"}
            />
          </div>

          {/* Coverage quality row */}
          <div
            className="glass-card rounded-xl px-5 py-4 flex items-center justify-between fade-in"
          >
            <div className="flex items-center gap-3">
              <Target size={16} style={{ color: "var(--accent)" }} />
              <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Coverage quality · 720-bar walk-forward backtest
              </span>
            </div>
            <QualityBadge quality={data.coverage_quality} />
          </div>
        </>
      ) : null}
    </div>
  );
}
