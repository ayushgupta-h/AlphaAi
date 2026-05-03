// Central API client — all fetch calls go through here.
// The Vite dev proxy forwards /api/* to http://localhost:8000.
// In production set VITE_API_BASE to your deployed backend URL.

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => get("/api/health"),
  price: () => get("/api/price"),
  predict: () => get("/api/predict"),
  backtest: () => get("/api/backtest"),
  metrics: () => get("/api/metrics"),
};
