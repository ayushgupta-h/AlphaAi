import { useState, useEffect, useCallback, useRef } from "react";
import { api } from "./api";

/** Poll an API fn every `intervalMs` ms. Returns { data, loading, error, refresh }. */
export function usePoll(apiFn, intervalMs = 30_000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const fnRef = useRef(apiFn);
  fnRef.current = apiFn;

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const result = await fnRef.current();
      setData(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return { data, loading, error, refresh };
}

/** One-shot fetch. Returns { data, loading, error }. */
export function useFetch(apiFn) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    apiFn()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error };
}
