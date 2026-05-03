/** Format a number as USD price, e.g. 43251.50 → "$43,251.50" */
export function formatPrice(n, decimals = 2) {
  if (n == null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(n);
}

/** Format a percentage, e.g. 0.9556 → "95.56%" */
export function formatPct(n, decimals = 2) {
  if (n == null) return "—";
  return `${(n * 100).toFixed(decimals)}%`;
}

/** Format a timestamp ISO string to a readable local time */
export function formatTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Format a timestamp ISO string to a readable local date+time */
export function formatDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

/** Clamp a value between min and max */
export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
