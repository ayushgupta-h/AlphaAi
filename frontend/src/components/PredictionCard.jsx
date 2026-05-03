import { useState } from "react";
import { api } from "../api";
import { usePoll } from "../hooks";
import { formatPrice, formatPct, clamp } from "../utils";
import { Skeleton, ErrorBanner, LiveBadge } from "./ui";
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Activity,
} from "lucide-react";

/** Animated interval bar showing where current price sits in the interval */
function IntervalBar({ lower, upper, current }) {
  const pct = clamp(((current - lower) / (upper - lower)) * 100, 2, 98);
  return (
    <div className="mt-4 space-y-2">
      <div className="flex justify-between text-xs font-mono" style={{ color: "var(--text-secondary)" }}>
        <span>{formatPrice(lower)}</span>
        <span style={{ color: "var(--accent)" }}>Current</span>
        <span>{formatPrice(upper)}</span>
      </div>
      <div className="relative h-2 rounded-full overflow-visible" style={{ background: "rgba(99,179,237,0.1)" }}>
        {/* Interval fill */}
        <div
          className="absolute inset-y-0 rounded-full"
          style={{
            left: 0,
            right: 0,
            background: "linear-gradient(90deg, rgba(99,179,237,0.1), rgba(99,179,237,0.35), rgba(99,179,237,0.1))",
          }}
        />
        {/* Current price marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2"
          style={{ left: `${pct}%`, transform: `translate(-50%, -50%)` }}
        >
          <div
            className="w-3.5 h-3.5 rounded-full border-2 border-white"
            style={{
              background: "var(--accent)",
              boxShadow: "0 0 10px var(--accent-glow)",
              transition: "left 0.6s ease",
            }}
          />
        </div>
      </div>
      <p className="text-xs text-center" style={{ color: "var(--text-muted)" }}>
        95% prediction interval for next hour
      </p>
    </div>
  );
}

/** Stat row inside the prediction card */
function StatRow({ label, value, mono = true }) {
  return (
    <div className="flex justify-between items-center py-2" style={{ borderBottom: "1px solid var(--border)" }}>
      <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span className={`text-sm font-semibold ${mono ? "font-mono" : ""}`} style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}

export function PredictionCard() {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const { data, loading, error, refresh } = usePoll(api.predict, 60_000);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refresh();
    setIsRefreshing(false);
  };

  const insideInterval =
    data &&
    data.current_price >= data.lower_bound &&
    data.current_price <= data.upper_bound;

  return (
    <div className="glass-card rounded-2xl p-6 fade-in glow-blue flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: "rgba(99,179,237,0.12)", color: "var(--accent)" }}
          >
            <Activity size={18} />
          </div>
          <div>
            <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Next-Hour Prediction
            </h2>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>BTC/USDT · 95% interval</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <LiveBadge />
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="p-1.5 rounded-lg transition-all duration-200 hover:bg-white/5"
            style={{ color: "var(--text-secondary)" }}
            title="Refresh prediction"
          >
            <RefreshCw size={14} className={isRefreshing ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {loading && !data ? (
        <div className="space-y-3">
          <Skeleton className="h-12 w-48" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      ) : data ? (
        <>
          {/* Current price */}
          <div className="flex items-end gap-3">
            <span className="text-4xl font-bold font-mono gradient-text">
              {formatPrice(data.current_price)}
            </span>
            <div
              className="flex items-center gap-1 mb-1 text-sm font-medium"
              style={{ color: insideInterval ? "var(--green)" : "var(--orange)" }}
            >
              {insideInterval ? <TrendingUp size={15} /> : <TrendingDown size={15} />}
              {insideInterval ? "In interval" : "Outside interval"}
            </div>
          </div>

          {/* Interval bounds */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Lower Bound", value: formatPrice(data.lower_bound), color: "var(--green)" },
              { label: "Upper Bound", value: formatPrice(data.upper_bound), color: "var(--red)" },
            ].map(({ label, value, color }) => (
              <div
                key={label}
                className="rounded-xl p-3 text-center"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)" }}
              >
                <p className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</p>
                <p className="text-base font-bold font-mono" style={{ color }}>{value}</p>
              </div>
            ))}
          </div>

          {/* Width badge */}
          <div
            className="text-center rounded-lg py-2 text-sm font-mono"
            style={{ background: "rgba(99,179,237,0.06)", color: "var(--accent)" }}
          >
            Interval width: <strong>{formatPrice(data.interval_width)}</strong>
          </div>

          {/* Interval bar */}
          <IntervalBar
            lower={data.lower_bound}
            upper={data.upper_bound}
            current={data.current_price}
          />

          {/* Expandable details */}
          <button
            onClick={() => setShowDetails((v) => !v)}
            className="flex items-center gap-1 text-xs transition-colors"
            style={{ color: "var(--text-secondary)" }}
          >
            {showDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            {showDetails ? "Hide" : "Show"} model parameters
          </button>

          {showDetails && (
            <div className="fade-in rounded-xl overflow-hidden" style={{ border: "1px solid var(--border)" }}>
              <StatRow label="Annualised Volatility" value={`${(data.volatility * 100).toFixed(3)}%`} />
              <StatRow label="Annualised Drift" value={`${(data.drift * 100).toFixed(3)}%`} />
              <StatRow label="Confidence Level" value={formatPct(data.confidence_level, 0)} mono={false} />
              <StatRow label="Time Horizon" value={`${data.horizon_hours}h`} mono={false} />
              <StatRow
                label="Last updated"
                value={new Date(data.timestamp).toLocaleTimeString()}
                mono={false}
              />
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
