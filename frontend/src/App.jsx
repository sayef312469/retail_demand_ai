import { useState, useEffect, useCallback } from "react";
import { api } from "./api";
import StatCard             from "./components/StatCard";
import Badge                from "./components/Badge";
import ForecastChart        from "./components/ForecastChart";
import PVIGauge             from "./components/PVIGauge";
import SubScoreBar          from "./components/SubScoreBar";
import EvalTable            from "./components/EvalTable";
import RecommendationsTable from "./components/RecommendationsTable";
import PlotsGallery         from "./components/PlotsGallery";
import UploadForecast       from "./components/UploadForecast";

const TABS = ["Dashboard", "Forecast", "PVI", "Recommendations", "Charts", "Upload", "Evaluation"];

export default function App() {
  const [tab,         setTab]         = useState("Dashboard");
  const [stores,      setStores]      = useState([]);
  const [items,       setItems]       = useState([]);
  const [store,       setStore]       = useState("");
  const [item,        setItem]        = useState("");
  const [summary,     setSummary]     = useState(null);
  const [forecast,    setForecast]    = useState(null);
  const [pvi,         setPvi]         = useState(null);
  const [rec,         setRec]         = useState(null);
  const [recList,     setRecList]     = useState([]);
  const [evalData,    setEvalData]    = useState(null);
  const [loading,     setLoading]     = useState({});
  const [modelFilter, setModelFilter] = useState("both");
  const [decFilter,   setDecFilter]   = useState("");

  const setLoad = (key, val) => setLoading(p => ({ ...p, [key]: val }));

  useEffect(() => {
    api.summary().then(setSummary).catch(() => {});
    api.stores().then(d => setStores(d.stores || [])).catch(() => {});
    api.recommendations({ limit: 200 }).then(d => setRecList(d.items || [])).catch(() => {});
    api.evalSummary().then(d => setEvalData(d.summary || [])).catch(() => {});
  }, []);

  useEffect(() => {
    if (!store) return;
    setItem(""); setItems([]);
    api.items(store).then(d => setItems(d.items || [])).catch(() => {});
  }, [store]);

  const loadItemData = useCallback(async () => {
    if (!store || !item) return;
    setForecast(null); setPvi(null); setRec(null);
    setLoad("forecast", true);
    api.forecast(store, item, modelFilter)
      .then(setForecast).catch(() => {}).finally(() => setLoad("forecast", false));
    api.pvi(store, item).then(setPvi).catch(() => {});
    api.recommendation(store, item).then(setRec).catch(() => {});
  }, [store, item, modelFilter]);

  useEffect(() => { loadItemData(); }, [loadItemData]);

  useEffect(() => {
    api.recommendations({ store_id: store || undefined, decision: decFilter || undefined, limit: 200 })
      .then(d => setRecList(d.items || [])).catch(() => {});
  }, [store, decFilter]);

  const s = styles;

  return (
    <div style={s.root}>
      <header style={s.header}>
        <div style={s.headerInner}>
          <div>
            <h1 style={s.h1}>Retail Demand AI</h1>
            <p style={s.h1Sub}>Product Viability &amp; Stock Recommendation — Group B-5</p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={s.dot(summary ? "#15803d" : "#b91c1c")} />
            <span style={{ fontSize: 12, color: "#78716c" }}>{summary ? "API connected" : "API offline"}</span>
          </div>
        </div>
      </header>

      <div style={s.layout}>
        {/* ── Sidebar ──────────────────────────────────────────────── */}
        <aside style={s.sidebar}>
          <div style={s.sideSection}>
            <label style={s.label}>Store</label>
            <select style={s.select} value={store} onChange={e => setStore(e.target.value)}>
              <option value="">All stores</option>
              {stores.map(st => <option key={st} value={st}>{st}</option>)}
            </select>
          </div>

          <div style={s.sideSection}>
            <label style={s.label}>Item</label>
            <select style={s.select} value={item} onChange={e => setItem(e.target.value)} disabled={!items.length}>
              <option value="">Select item…</option>
              {items.map(it => (
                <option key={it.item_id} value={it.item_id}>
                  {it.item_id} — {it.viability} ({Number(it.PVI).toFixed(0)})
                </option>
              ))}
            </select>
          </div>

          <div style={s.sideSection}>
            <label style={s.label}>Forecast model</label>
            <div style={{ display: "flex", gap: 6 }}>
              {["both", "prophet", "arima"].map(m => (
                <button key={m} onClick={() => setModelFilter(m)} style={s.pill(modelFilter === m)}>{m}</button>
              ))}
            </div>
          </div>

          <nav style={{ marginTop: "auto" }}>
            {TABS.map(t => (
              <button key={t} onClick={() => setTab(t)} style={s.navBtn(tab === t)}>{t}</button>
            ))}
          </nav>
        </aside>

        {/* ── Main ─────────────────────────────────────────────────── */}
        <main style={s.main}>

          {/* DASHBOARD */}
          {tab === "Dashboard" && (
            <div>
              <h2 style={s.h2}>Dashboard</h2>
              {summary ? (
                <>
                  <div style={s.cardGrid}>
                    <StatCard label="Total items"    value={summary.total_items}           sub={`${summary.total_stores} stores`} accent="blue"   />
                    <StatCard label="High viability" value={summary.viability?.High   || 0} sub="PVI ≥ 67"    accent="green"  />
                    <StatCard label="Low viability"  value={summary.viability?.Low    || 0} sub="PVI < 33"    accent="red"    />
                    <StatCard label="Avg PVI"        value={summary.pvi_mean?.toFixed(1)}   sub={`Median ${summary.pvi_median}`} accent="purple" />
                    <StatCard label="Increase"       value={summary.decisions?.Increase||0} sub="stock up"    accent="blue"   />
                    <StatCard label="Decrease"       value={summary.decisions?.Decrease||0} sub="reduce stock" accent="red"   />
                    <StatCard label="Hold"           value={summary.decisions?.Hold    ||0} sub="maintain"    accent="amber"  />
                    <StatCard label="Anomalies"      value={summary.anomaly_count      || 0} sub="items flagged" accent="amber"/>
                  </div>

                  {summary.model_performance && Object.keys(summary.model_performance).length > 0 && (
                    <div style={{ ...s.card, marginTop: 20 }}>
                      <h3 style={s.h3}>Model performance summary</h3>
                      <p style={{ fontSize: 12, color: "#78716c", margin: "4px 0 14px" }}>
                        MAE, RMSE, MAPE, WAPE and R² are the correct metrics for this regression/forecasting problem.
                        Precision/Recall/F1 are classification metrics and do not apply here.
                      </p>
                      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                        {Object.entries(summary.model_performance).map(([m, v]) => (
                          <div key={m} style={{ background: "#f5f5f4", borderRadius: 10, padding: "12px 16px", minWidth: 200 }}>
                            <div style={{ marginBottom: 8 }}><Badge text={m} size="lg" /></div>
                            {[["MAE", v.MAE_mean], ["RMSE", v.RMSE_mean], ["MAPE", v.MAPE_mean != null ? v.MAPE_mean.toFixed(1) + "%" : "—"], ["WAPE", v.WAPE_mean != null ? v.WAPE_mean.toFixed(1) + "%" : "—"], ["R²", v.R2_mean != null ? v.R2_mean.toFixed(3) : "—"]].map(([label, val]) => (
                              <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 3 }}>
                                <span style={{ color: "#78716c" }}>{label}</span>
                                <span style={{ fontWeight: 600 }}>{val != null ? val : "—"}</span>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div style={s.emptyState}>
                  <p>Run the full pipeline first:</p>
                  <pre style={s.code}>{"python src/forecast.py\npython src/pvi.py\npython src/recommend.py\npython src/evaluate.py\npython src/plot_reports.py"}</pre>
                </div>
              )}
            </div>
          )}

          {/* FORECAST */}
          {tab === "Forecast" && (
            <div>
              <h2 style={s.h2}>Demand Forecast</h2>
              {!item ? (
                <div style={s.emptyState}>Select a store and item from the sidebar.</div>
              ) : loading.forecast ? (
                <div style={s.emptyState}>Loading forecast…</div>
              ) : (
                <div style={s.card}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                    <div>
                      <h3 style={s.h3}>{item}</h3>
                      <p style={{ fontSize: 12, color: "#78716c" }}>{store} · 3-month ahead forecast with 95% confidence bands</p>
                    </div>
                    {rec && <Badge text={rec.Decision} size="lg" />}
                  </div>
                  <ForecastChart data={forecast} />
                  {!forecast && <div style={s.emptyState}>No forecast found for this item.</div>}
                  <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
                    {forecast?.prophet && <span style={{ fontSize: 12, color: "#78716c" }}><span style={{ color: "#3b82f6", fontWeight: 700 }}>— </span>Prophet (dashed blue)</span>}
                    {forecast?.arima   && <span style={{ fontSize: 12, color: "#78716c" }}><span style={{ color: "#7c3aed", fontWeight: 700 }}>— </span>ARIMA (dotted purple) · {forecast.arima.model_order}</span>}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* PVI */}
          {tab === "PVI" && (
            <div>
              <h2 style={s.h2}>Product Viability Index</h2>
              {!item ? (
                <div style={s.emptyState}>Select a store and item from the sidebar.</div>
              ) : !pvi ? (
                <div style={s.emptyState}>No PVI data for this item.</div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  <div style={s.card}>
                    <h3 style={s.h3}>PVI score</h3>
                    <PVIGauge pvi={pvi.PVI} viability={pvi.viability} />
                    <div style={{ textAlign: "center", marginTop: 8 }}><Badge text={pvi.viability} size="lg" /></div>
                    <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[["Rank (overall)", `#${pvi.rank_overall}`], ["Rank in category", `#${pvi.rank_in_category}`], ["Anomaly months", pvi.anomaly_count || 0], ["Model agreement", pvi.model_agreement != null ? `${(pvi.model_agreement * 100).toFixed(0)}%` : "N/A"]].map(([l, v]) => (
                        <div key={l} style={{ background: "#f5f5f4", borderRadius: 8, padding: "10px 12px" }}>
                          <div style={{ fontSize: 11, color: "#78716c" }}>{l}</div>
                          <div style={{ fontSize: 16, fontWeight: 700, color: "#1c1917" }}>{v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={s.card}>
                    <h3 style={s.h3}>Sub-score breakdown</h3>
                    <p style={{ fontSize: 12, color: "#78716c", marginBottom: 16 }}>PVI = 0.40×Demand + 0.25×Growth + 0.20×Stability + 0.15×Price</p>
                    <SubScoreBar pviData={pvi} />
                    {pvi.has_anomaly && (
                      <div style={{ marginTop: 16, padding: "10px 14px", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 8 }}>
                        <p style={{ fontSize: 13, color: "#b45309", fontWeight: 600 }}>Anomaly detected</p>
                        <p style={{ fontSize: 12, color: "#78716c", marginTop: 4 }}>{(pvi.anomaly_pct * 100).toFixed(0)}% of historical months flagged via IQR method.</p>
                      </div>
                    )}
                    {rec && (
                      <div style={{ marginTop: 16, padding: "12px 14px", background: "#f5f5f4", borderRadius: 8 }}>
                        <p style={{ fontSize: 12, color: "#44403c", marginBottom: 6, fontWeight: 600 }}>Stock recommendation</p>
                        <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
                          <Badge text={rec.Decision} size="lg" />
                          <span style={{ fontSize: 12, color: "#78716c" }}>Confidence: <strong>{rec.Confidence}</strong></span>
                        </div>
                        <p style={{ fontSize: 11, color: "#78716c", lineHeight: 1.5 }}>{rec.Explanation}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* RECOMMENDATIONS */}
          {tab === "Recommendations" && (
            <div>
              <h2 style={s.h2}>Stock Recommendations</h2>
              <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
                <span style={{ fontSize: 13, color: "#78716c" }}>Filter:</span>
                {["", "Increase", "Hold", "Decrease"].map(d => (
                  <button key={d} onClick={() => setDecFilter(d)} style={s.pill(decFilter === d)}>{d || "All"}</button>
                ))}
                <span style={{ fontSize: 12, color: "#a8a29e", marginLeft: "auto" }}>{recList.length} items</span>
              </div>
              <div style={s.card}><RecommendationsTable items={recList} /></div>
            </div>
          )}

          {/* CHARTS */}
          {tab === "Charts" && (
            <div>
              <h2 style={s.h2}>Visualisation Charts</h2>
              <p style={{ fontSize: 13, color: "#78716c", marginBottom: 20 }}>
                Click any thumbnail to expand. Run <code>python src/plot_reports.py</code> to regenerate after pipeline updates.
              </p>
              <PlotsGallery />
            </div>
          )}

          {/* UPLOAD */}
          {tab === "Upload" && (
            <div>
              <h2 style={s.h2}>Upload &amp; Forecast Your Own Data</h2>
              <UploadForecast />
            </div>
          )}

          {/* EVALUATION */}
          {tab === "Evaluation" && (
            <div>
              <h2 style={s.h2}>Model Evaluation</h2>
              <div style={{ ...s.card, marginBottom: 16 }}>
                <h3 style={s.h3}>Why these metrics?</h3>
                <p style={{ fontSize: 13, color: "#57534e", marginTop: 6, lineHeight: 1.7 }}>
                  Demand forecasting is a <strong>regression problem</strong> — we are predicting a continuous number (sales units), not a category.
                  The correct metrics are MAE, RMSE, MAPE, WAPE, R² and bias. Precision, Recall, F1-score and Confusion Matrix are
                  <strong> classification metrics</strong> — they require a right/wrong label per prediction, which does not exist here.
                  The stock recommendation (Increase/Hold/Decrease) is classification, but it is rule-based without ground-truth labels, so supervised metrics cannot be computed.
                </p>
              </div>
              <div style={s.card}>
                <h3 style={s.h3}>Aggregate metrics — Prophet vs ARIMA</h3>
                <p style={{ fontSize: 12, color: "#78716c", margin: "6px 0 16px" }}>Holdout evaluation on last 3 months of each series.</p>
                <EvalTable summary={evalData} />
              </div>
              {item && (
                <div style={{ ...s.card, marginTop: 16 }}>
                  <h3 style={s.h3}>Per-item metrics — {item}</h3>
                  <ItemEvalPanel store={store} item={item} />
                </div>
              )}
            </div>
          )}

        </main>
      </div>
    </div>
  );
}

function ItemEvalPanel({ store, item }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!store || !item) return;
    api.evalItem(store, item).then(d => setData(d.metrics)).catch(() => setData([]));
  }, [store, item]);
  if (!data) return <div style={{ color: "#78716c", fontSize: 13, padding: 8 }}>Loading…</div>;
  if (!data.length) return <div style={{ color: "#78716c", fontSize: 13, padding: 8 }}>No eval data for this item.</div>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, marginTop: 8 }}>
      <thead>
        <tr style={{ background: "#f5f5f4" }}>
          {["Model", "MAE", "RMSE", "MAPE", "WAPE", "R²", "Bias", "Test obs"].map(h => (
            <th key={h} style={{ padding: "8px 12px", textAlign: "left", fontWeight: 600, fontSize: 12, color: "#44403c", borderBottom: "1px solid #e7e5e4" }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((r, i) => (
          <tr key={i} style={{ borderTop: "1px solid #f5f5f4" }}>
            <td style={{ padding: "8px 12px", fontWeight: 600 }}><Badge text={r.model} /></td>
            <td style={{ padding: "8px 12px" }}>{r.MAE?.toFixed(2)}</td>
            <td style={{ padding: "8px 12px" }}>{r.RMSE?.toFixed(2)}</td>
            <td style={{ padding: "8px 12px" }}>{r.MAPE != null ? `${r.MAPE.toFixed(1)}%` : "—"}</td>
            <td style={{ padding: "8px 12px" }}>{r.WAPE != null ? `${r.WAPE.toFixed(1)}%` : "—"}</td>
            <td style={{ padding: "8px 12px", color: r.R2 > 0.5 ? "#15803d" : r.R2 < 0 ? "#b91c1c" : "#78716c" }}>{r.R2 != null ? r.R2.toFixed(3) : "—"}</td>
            <td style={{ padding: "8px 12px", color: r.bias > 0 ? "#b91c1c" : "#15803d" }}>{r.bias > 0 ? "+" : ""}{r.bias?.toFixed(2)}</td>
            <td style={{ padding: "8px 12px", color: "#78716c" }}>{r.n_test_obs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const styles = {
  root:      { minHeight: "100vh", display: "flex", flexDirection: "column" },
  header:    { background: "#fff", borderBottom: "1px solid #e7e5e4", padding: "0 24px" },
  headerInner: { maxWidth: 1280, margin: "0 auto", padding: "16px 0", display: "flex", justifyContent: "space-between", alignItems: "center" },
  h1:        { fontSize: 20, fontWeight: 700, color: "#1c1917" },
  h1Sub:     { fontSize: 12, color: "#78716c", marginTop: 2 },
  layout:    { display: "flex", flex: 1, maxWidth: 1280, margin: "0 auto", width: "100%", padding: "24px", gap: 24 },
  sidebar:   { width: 220, flexShrink: 0, display: "flex", flexDirection: "column", gap: 20 },
  sideSection: { display: "flex", flexDirection: "column", gap: 6 },
  label:     { fontSize: 11, fontWeight: 600, color: "#78716c", textTransform: "uppercase", letterSpacing: "0.05em" },
  select:    { padding: "8px 10px", borderRadius: 8, border: "1px solid #e7e5e4", fontSize: 13, background: "#fff", cursor: "pointer", width: "100%" },
  main:      { flex: 1, minWidth: 0 },
  card:      { background: "#fff", border: "1px solid #e7e5e4", borderRadius: 12, padding: "20px 24px" },
  cardGrid:  { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12 },
  h2:        { fontSize: 20, fontWeight: 700, color: "#1c1917", marginBottom: 20 },
  h3:        { fontSize: 15, fontWeight: 600, color: "#1c1917" },
  emptyState: { background: "#fff", border: "1px solid #e7e5e4", borderRadius: 12, padding: 40, textAlign: "center", color: "#78716c", fontSize: 14 },
  code:      { background: "#1c1917", color: "#d4d0ca", borderRadius: 8, padding: "10px 14px", fontSize: 12, fontFamily: "monospace", marginTop: 12, textAlign: "left", display: "inline-block" },
  navBtn:    (active) => ({ display: "block", width: "100%", textAlign: "left", padding: "9px 12px", background: active ? "#f5f5f4" : "transparent", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 13, fontWeight: active ? 600 : 400, color: active ? "#1c1917" : "#57534e", marginBottom: 2 }),
  pill:      (active) => ({ padding: "5px 12px", borderRadius: 20, border: `1px solid ${active ? "#1c1917" : "#e7e5e4"}`, background: active ? "#1c1917" : "#fff", color: active ? "#fff" : "#57534e", fontSize: 12, cursor: "pointer", fontWeight: active ? 600 : 400 }),
  dot:       (color) => ({ width: 8, height: 8, borderRadius: "50%", background: color }),
};
