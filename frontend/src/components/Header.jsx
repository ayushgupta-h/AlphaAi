import { Bitcoin, ExternalLink } from "lucide-react";

export function Header() {
  return (
    <header
      className="w-full border-b px-6 py-4"
      style={{
        borderColor: "var(--border)",
        background: "rgba(6,11,20,0.8)",
        backdropFilter: "blur(16px)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #63b3ed, #7c3aed)" }}
          >
            <Bitcoin size={16} className="text-white" />
          </div>
          <div>
            <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
              AlphAI
            </span>
            <span
              className="ml-2 text-xs font-medium px-1.5 py-0.5 rounded"
              style={{
                background: "rgba(99,179,237,0.12)",
                color: "var(--accent)",
                border: "1px solid rgba(99,179,237,0.2)",
              }}
            >
              BTC Forecasting
            </span>
          </div>
        </div>

        {/* Nav */}
        <nav className="hidden md:flex items-center gap-6 text-sm" style={{ color: "var(--text-secondary)" }}>
          <a href="#predict" className="hover:text-white transition-colors">Live Prediction</a>
          <a href="#backtest" className="hover:text-white transition-colors">Backtest</a>
          <a href="#metrics" className="hover:text-white transition-colors">Metrics</a>
          <a
            href="/api/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 hover:text-white transition-colors"
          >
            API Docs <ExternalLink size={12} />
          </a>
        </nav>

        {/* Model badge */}
        <div
          className="text-xs rounded-full px-3 py-1 font-medium"
          style={{
            background: "rgba(124,58,237,0.12)",
            color: "#a78bfa",
            border: "1px solid rgba(124,58,237,0.2)",
          }}
        >
          GBM + EWMA · Student-t
        </div>
      </div>
    </header>
  );
}
