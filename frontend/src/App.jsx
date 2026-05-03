import { Header } from "./components/Header";
import { PredictionCard } from "./components/PredictionCard";
import { BacktestChart } from "./components/BacktestChart";
import { MetricsPanel } from "./components/MetricsPanel";
import "./index.css";

/** Hero section */
function Hero() {
  return (
    <div className="text-center py-16 px-4">
      <div
        className="inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-semibold mb-6"
        style={{
          background: "rgba(99,179,237,0.08)",
          border: "1px solid rgba(99,179,237,0.2)",
          color: "var(--accent)",
        }}
      >
        <span className="relative flex h-1.5 w-1.5">
          <span
            className="absolute inline-flex h-full w-full rounded-full opacity-75"
            style={{ background: "var(--green)", animation: "pulse 1.5s ease-out infinite" }}
          />
          <span
            className="relative inline-flex h-1.5 w-1.5 rounded-full"
            style={{ background: "#68d391" }}
          />
        </span>
        AlphaI × Polaris Challenge · Live Inference
      </div>

      <h1 className="text-4xl md:text-5xl font-extrabold mb-4 leading-tight">
        <span className="gradient-text">Bitcoin Probabilistic</span>
        <br />
        <span style={{ color: "var(--text-primary)" }}>Forecasting System</span>
      </h1>

      <p className="max-w-xl mx-auto text-base mb-8" style={{ color: "var(--text-secondary)" }}>
        Next-hour BTC/USDT 95% prediction intervals using{" "}
        <span style={{ color: "var(--accent)" }}>Geometric Brownian Motion</span> with{" "}
        <span style={{ color: "#a78bfa" }}>EWMA volatility</span> and{" "}
        <span style={{ color: "#ec4899" }}>Student-t shocks</span>.
        Walk-forward validated on 720 hourly bars.
      </p>

      {/* Pill badges */}
      <div className="flex flex-wrap justify-center gap-2 text-xs">
        {[
          { label: "GBM Simulation", color: "#63b3ed" },
          { label: "EWMA Volatility", color: "#a78bfa" },
          { label: "Student-t Shocks", color: "#ec4899" },
          { label: "10,000 MC Paths", color: "#68d391" },
          { label: "95% CI", color: "#ed8936" },
        ].map(({ label, color }) => (
          <span
            key={label}
            className="px-3 py-1 rounded-full font-medium"
            style={{
              background: `${color}14`,
              color,
              border: `1px solid ${color}33`,
            }}
          >
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Divider */
function Divider() {
  return (
    <div
      className="my-6"
      style={{ height: 1, background: "var(--border)" }}
    />
  );
}

export default function App() {
  return (
    <div className="bg-animated min-h-screen">
      <Header />

      <main className="max-w-7xl mx-auto px-4 pb-20">
        {/* Hero */}
        <Hero />

        {/* Live Prediction */}
        <section id="predict" className="mb-12">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Prediction card takes 1/3 width on desktop */}
            <div className="lg:col-span-1">
              <PredictionCard />
            </div>

            {/* How it works */}
            <div className="lg:col-span-2 glass-card rounded-2xl p-6 fade-in">
              <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                ⚙️ How It Works
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  {
                    step: "1",
                    title: "Fetch Live Data",
                    desc: "Retrieves the last 60 hourly BTCUSDT closes from Binance in real time.",
                    color: "#63b3ed",
                  },
                  {
                    step: "2",
                    title: "EWMA Volatility",
                    desc: "Computes exponentially weighted moving average volatility (λ=0.94) that adapts to recent market conditions.",
                    color: "#a78bfa",
                  },
                  {
                    step: "3",
                    title: "GBM + Student-t",
                    desc: "Runs 10,000 Monte Carlo GBM paths with Student-t shocks (df=5) to capture fat-tailed crypto returns.",
                    color: "#ec4899",
                  },
                  {
                    step: "4",
                    title: "95% Interval",
                    desc: "Extracts the 2.5th–97.5th percentile from simulated terminal prices as the prediction interval.",
                    color: "#68d391",
                  },
                ].map(({ step, title, desc, color }) => (
                  <div
                    key={step}
                    className="rounded-xl p-4 transition-all duration-200 hover:-translate-y-0.5"
                    style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)" }}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                        style={{ background: `${color}20`, color }}
                      >
                        {step}
                      </span>
                      <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                        {title}
                      </span>
                    </div>
                    <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                      {desc}
                    </p>
                  </div>
                ))}
              </div>

              {/* API CTA */}
              <div
                className="mt-5 rounded-xl p-4 flex items-center justify-between"
                style={{ background: "rgba(99,179,237,0.06)", border: "1px solid rgba(99,179,237,0.15)" }}
              >
                <div>
                  <p className="text-xs font-semibold" style={{ color: "var(--accent)" }}>
                    REST API available
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                    <code className="font-mono">GET /api/predict</code> · <code className="font-mono">GET /api/backtest</code> · <code className="font-mono">GET /api/metrics</code>
                  </p>
                </div>
                <a
                  href="/api/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs font-semibold px-4 py-2 rounded-lg transition-all duration-200 hover:opacity-80"
                  style={{
                    background: "rgba(99,179,237,0.15)",
                    color: "var(--accent)",
                    border: "1px solid rgba(99,179,237,0.3)",
                  }}
                >
                  View Docs →
                </a>
              </div>
            </div>
          </div>
        </section>

        <Divider />

        {/* Metrics */}
        <section id="metrics" className="mb-12">
          <MetricsPanel />
        </section>

        <Divider />

        {/* Backtest chart */}
        <section id="backtest" className="mb-12">
          <BacktestChart />
        </section>

        {/* Footer */}
        <footer className="text-center text-xs py-8" style={{ color: "var(--text-muted)" }}>
          AlphAI Bitcoin Probabilistic Forecasting · AlphaI × Polaris Challenge ·{" "}
          <a href="/api/docs" className="hover:text-white transition-colors">
            API Documentation
          </a>
        </footer>
      </main>
    </div>
  );
}
