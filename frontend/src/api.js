const BASE = "";   // proxied to http://localhost:8000 via package.json proxy

async function get(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  health:          ()                    => get("/health"),
  summary:         ()                    => get("/summary"),
  stores:          ()                    => get("/stores"),
  items:           (store, cat, viab)    => {
    const p = new URLSearchParams({ store_id: store });
    if (cat)  p.set("category",   cat);
    if (viab) p.set("viability",  viab);
    return get(`/items?${p}`);
  },
  forecast:        (store, item, model, months)  => {
    const p = new URLSearchParams({ model: model || "both" });  // "both" now includes ensemble
    if (months) p.set("months", months);
    return get(`/forecast/${store}/${encodeURIComponent(item)}?${p}`);
  },
  pvi:             (store, item)         => get(`/pvi/${store}/${encodeURIComponent(item)}`),
  pviList:         (store, viab, cat, limit, offset) => {
    const p = new URLSearchParams();
    if (store)  p.set("store_id",  store);
    if (viab)   p.set("viability", viab);
    if (cat)    p.set("category",  cat);
    if (limit)  p.set("limit",     limit);
    if (offset) p.set("offset",    offset);
    return get(`/pvi?${p}`);
  },
  recommendation:  (store, item)         => get(`/recommendation/${store}/${encodeURIComponent(item)}`),
  recommendations: (filters)             => {
    const p = new URLSearchParams();
    Object.entries(filters || {}).forEach(([k, v]) => { if (v != null) p.set(k, v); });
    return get(`/recommendations?${p}`);
  },
  evalSummary:     ()                    => get("/eval"),
  evalItem:        (store, item)         => get(`/eval/${store}/${encodeURIComponent(item)}`),
};