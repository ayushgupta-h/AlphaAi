import { useMemo } from "react";
import { useFetch } from "../hooks";
import { api } from "../api";
import { formatPrice, formatDateTime } from "../utils";
import { Skeleton, ErrorBanner, SectionTitle } from "./ui";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";

/** Custom tooltip for the chart */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  return (
    <div
      className="rounded-xl px-4 py-3 text-xs shadow-2xl"
      style={{
        background: "rgba(13,23,36,0.95)",
        border: "1px solid var(--border-hover)",
        backdropFilter: "blur(12px)",
      }}
    >
      <p className="font-semibold mb-2" style={{ color: "var(--text-secondary)" }}>
        {label}
      </p>
      <div className="space-y-1">
        <p style={{ color: "#63b3ed" }}>
          Actual: <strong>{formatPrice(d.actual_price)}</strong>
        </p>
        <p style={{ color: "rgba(99,179,237,0.5)" }}>
          Lower: {formatPrice(d.lower_bound)}
        </p>
        <p style={{ color: "rgba(99,179,237,0.5)" }}>
          Upper: {formatPrice(d.upper_bound)}
        </p>
        <p
          className="font-semibold"
          style={{ color: d.covered ? "#68d391" : "#fc8150" }}
        >
          {d.covered ? "✓ Inside interval" : "✗ Violation"}
        </p>
      </div>
    </div>
  );
}

export function BacktestChart() {
  const { data: raw, loading, error } = useFetch(api.backtest);

  // Downsample to every Nth point for performance, keeping violations
  const chartData = useMemo(() => {
    if (!raw) return [];
    // Show every 4th point, always include violations
    return raw
      .filter((r, i) => i % 4 === 0 || !r.covered)
      .map((r) => ({
        ...r,
        date: formatDateTime(r.timestamp),
        // Recharts area band: provide [lower, upper] as range
        band: [r.lower_bound, r.upper_bound],
      }));
  }, [raw]);

  const violations = useMemo(() => raw?.filter((r) => !r.covered) ?? [], [raw]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <SectionTitle>📈 Walk-Forward Backtest — 720 Hourly Predictions</SectionTitle>
      </div>

      {error && <ErrorBanner message={error} />}

      {loading ? (
        <Skeleton className="h-72 w-full" />
      ) : chartData.length > 0 ? (
        <>
          {/* Violation count banner */}
          <div
            className="mb-4 flex items-center gap-3 rounded-xl px-4 py-2.5 text-xs fade-in"
            style={{
              background: "rgba(99,179,237,0.05)",
              border: "1px solid var(--border)",
            }}
          >
            <span style={{ color: "var(--text-secondary)" }}>
              <span style={{ color: "#68d391" }}>●</span> Blue band = covered intervals &nbsp;
              <span style={{ color: "#fc8150" }}>●</span> Orange dots = {violations.length} violations
            </span>
            <span className="ml-auto font-mono" style={{ color: "var(--accent)" }}>
              {raw?.length} predictions total
            </span>
          </div>

          <div
            className="glass-card rounded-2xl p-4 fade-in"
            style={{ height: 340 }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{ top: 10, right: 10, bottom: 0, left: 10 }}
              >
                <defs>
                  {/* Blue gradient fill for interval band */}
                  <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#63b3ed" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#63b3ed" stopOpacity={0.05} />
                  </linearGradient>
                  {/* Blue line gradient */}
                  <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#63b3ed" />
                    <stop offset="100%" stopColor="#7c3aed" />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148,163,184,0.06)"
                />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  interval={Math.floor(chartData.length / 8)}
                />
                <YAxis
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                  width={48}
                />
                <Tooltip content={<CustomTooltip />} />

                {/* Upper bound area (top of band) */}
                <Area
                  type="monotone"
                  dataKey="upper_bound"
                  stroke="rgba(99,179,237,0.3)"
                  strokeWidth={1}
                  fill="url(#bandGrad)"
                  dot={false}
                  activeDot={false}
                  legendType="none"
                />

                {/* Lower bound area (bottom of band — fills over the gradient to cut off) */}
                <Area
                  type="monotone"
                  dataKey="lower_bound"
                  stroke="rgba(99,179,237,0.3)"
                  strokeWidth={1}
                  fill="var(--bg-elevated)"
                  dot={false}
                  activeDot={false}
                  legendType="none"
                />

                {/* Actual price line */}
                <Area
                  type="monotone"
                  dataKey="actual_price"
                  stroke="url(#lineGrad)"
                  strokeWidth={1.5}
                  fill="none"
                  dot={(props) => {
                    const d = props.payload;
                    if (d.covered) return null;
                    return (
                      <circle
                        key={props.index}
                        cx={props.cx}
                        cy={props.cy}
                        r={3}
                        fill="#fc8150"
                        stroke="none"
                      />
                    );
                  }}
                  activeDot={{ r: 4, fill: "#63b3ed", stroke: "#fff", strokeWidth: 1.5 }}
                  name="Actual Price"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : null}
    </div>
  );
}
