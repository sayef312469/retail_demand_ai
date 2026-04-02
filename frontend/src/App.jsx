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
  const [recSearchQuery, setRecSearchQuery] = useState("");
  const [recPage,     setRecPage]     = useState(0);
  const [recTotal,    setRecTotal]    = useState(0);
  const REC_PAGE_SIZE = 50;

  const setLoad = (key, val) => setLoading(p => ({ ...p, [key]: val }));

  useEffect(() => {
    api.summary().then(setSummary).catch(() => {});
    api.stores().then(d => setStores(d.stores || [])).catch(() => {});
    // Fetch with high limit to get total count dynamically
    api.recommendations({ limit: 10000 }).then(d => {
      setRecTotal(d.total || 0);
      setRecList(d.items || []);
    }).catch(() => {});
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
    api.forecast(store, item, modelFilter, 12)  // Always fetch full 12 months for client-side slicing in chart
      .then(setForecast).catch(() => {}).finally(() => setLoad("forecast", false));
    api.pvi(store, item).then(setPvi).catch(() => {});
    api.recommendation(store, item).then(setRec).catch(() => {});
  }, [store, item, modelFilter]);

  useEffect(() => { loadItemData(); }, [loadItemData]);

  useEffect(() => {
    setRecPage(0);  // Reset to first page when filters change
    api.recommendations({ store_id: store || undefined, decision: decFilter || undefined, limit: 10000 })
      .then(d => {
        setRecTotal(d.total || 0);
        setRecList(d.items || []);
      }).catch(() => {});
  }, [store, decFilter]);

  // Filter recommendations based on search query
  const filteredRecs = recList.filter(item => {
    if (!recSearchQuery.trim()) return true;
    const query = recSearchQuery.toLowerCase();
    return (
      (item.item_id && item.item_id.toLowerCase().includes(query)) ||
      (item.store_id && item.store_id.toLowerCase().includes(query)) ||
      (item.cat_id && item.cat_id.toLowerCase().includes(query))
    );
  });

  const paginatedRecs = filteredRecs.slice(recPage * REC_PAGE_SIZE, (recPage + 1) * REC_PAGE_SIZE);
  const totalPages = Math.ceil(filteredRecs.length / REC_PAGE_SIZE);
  
  const s = styles;

  return (
    <div style={s.root}>
      {/* ── HEADER ───────────────────────────────────────────────────── */}
      <header style={s.header}>
        <div style={s.headerInner}>
          <div>
            <h1 style={s.h1}>Retail Demand AI</h1>
            <p style={s.h1Sub}>Product Viability &amp; Stock Recommendation — Group B-5</p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={s.statusDot(summary ? "#15803d" : "#b91c1c")} />
            <span style={{ fontSize: 12, color: "#78716c", fontWeight: 500 }}>{summary ? "Connected" : "Offline"}</span>
          </div>
        </div>
      </header>

      {/* ── TAB NAVIGATION (Horizontal Top) ──────────────────────────── */}
      <nav style={s.tabNav}>
        <div style={s.tabNavInner}>
          {TABS.map((t, i) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                ...s.tabButton(tab === t),
                borderBottom: tab === t ? "3px solid #3b82f6" : "3px solid transparent"
              }}
            >
              {t}
            </button>
          ))}
        </div>
      </nav>

      {/* ── MAIN LAYOUT ──────────────────────────────────────────────── */}
      <div style={s.layoutWrapper}>
        {/* ── Controls Panel ───────────────────────────────────────────── */}
        <aside style={s.controlsPanel}>
          <div style={s.controlSection}>
            <label style={s.controlLabel}>Store</label>
            <select style={s.controlSelect} value={store} onChange={e => setStore(e.target.value)}>
              <option value="">All stores</option>
              {stores.map(st => <option key={st} value={st}>{st}</option>)}
            </select>
          </div>

          <div style={s.controlSection}>
            <label style={s.controlLabel}>Item</label>
            <select style={s.controlSelect} value={item} onChange={e => setItem(e.target.value)} disabled={!items.length}>
              <option value="">Select item…</option>
              {items.map(it => (
                <option key={it.item_id} value={it.item_id}>
                  {it.item_id} — {it.viability} ({Number(it.PVI).toFixed(0)})
                </option>
              ))}
            </select>
          </div>

          <div style={s.controlSection}>
            <label style={s.controlLabel}>Forecast Model</label>
            <div style={{ display: "flex", gap: 8 }}>
              {["both", "prophet", "arima"].map(m => (
                <button key={m} onClick={() => setModelFilter(m)} style={s.modelPill(modelFilter === m)}>
                  {m === "both" ? "Ensemble" : m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* ── Main Content ─────────────────────────────────────────────── */}
        <main style={s.mainContent}>

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

                  {summary.distribution && (
                    <div style={{ ...s.card, marginTop: 20 }}>
                      <h3 style={s.h3}>Data distribution & coverage</h3>
                      <p style={{ fontSize: 12, color: "#78716c", margin: "4px 0 14px" }}>
                        Training dataset includes {summary.total_items} items across {summary.total_stores} stores and {summary.total_categories} categories. Breakdown by store and category:
                      </p>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                        {/* Items per Store */}
                        <div>
                          <h4 style={{ fontSize: 13, fontWeight: 600, color: "#44403c", marginBottom: 12 }}>Items per store</h4>
                          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            {Object.entries(summary.distribution.items_per_store || {})
                              .sort(([, a], [, b]) => b - a)
                              .map(([store, count]) => (
                                <div key={store} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                  <span style={{ fontSize: 12, color: "#57534e", minWidth: 60 }}>{store}</span>
                                  <div style={{ flex: 1, height: 20, backgroundColor: "#e7e5e4", borderRadius: 4, marginLeft: 8, position: "relative" }}>
                                    <div style={{ 
                                      height: "100%", 
                                      width: `${(count / Math.max(...Object.values(summary.distribution.items_per_store || {}))) * 100}%`,
                                      backgroundColor: "#3b82f6", 
                                      borderRadius: 4,
                                      transition: "width 0.3s"
                                    }} />
                                  </div>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#1c1917", minWidth: 40, textAlign: "right", marginLeft: 8 }}>{count}</span>
                                </div>
                              ))}
                          </div>
                        </div>

                        {/* Items per Category */}
                        <div>
                          <h4 style={{ fontSize: 13, fontWeight: 600, color: "#44403c", marginBottom: 12 }}>Items per category</h4>
                          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            {Object.entries(summary.distribution.items_per_category || {})
                              .sort(([, a], [, b]) => b - a)
                              .map(([cat, count]) => (
                                <div key={cat} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                  <span style={{ fontSize: 12, color: "#57534e", minWidth: 100 }}>{cat}</span>
                                  <div style={{ flex: 1, height: 20, backgroundColor: "#e7e5e4", borderRadius: 4, marginLeft: 8, position: "relative" }}>
                                    <div style={{ 
                                      height: "100%", 
                                      width: `${(count / Math.max(...Object.values(summary.distribution.items_per_category || {}))) * 100}%`,
                                      backgroundColor: "#10b981", 
                                      borderRadius: 4,
                                      transition: "width 0.3s"
                                    }} />
                                  </div>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#1c1917", minWidth: 40, textAlign: "right", marginLeft: 8 }}>{count}</span>
                                </div>
                              ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div style={s.emptyState}>
                  <p>Run the full pipeline first:</p>
                  <pre style={s.code}>{"python src/forecast.py\npython src/pvi.py\npython src/recommend.py\npython src/evaluate.py\npython src/plot_reports.py"}</pre>
                  <p style={{ fontSize: 12, color: "#78716c", marginTop: 16 }}>Tip: Default trains on top 500 items by sales volume. To scale up:</p>
                  <pre style={s.code}>{"# Train on 1000 items (more comprehensive)\npython src/forecast.py --top-items 1000\n\n# Then run PVI, recommendations, etc."}</pre>
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
                      <p style={{ fontSize: 12, color: "#78716c" }}>{store} · 3–12 month forecast. Ensemble (60% ARIMA + 40% Prophet) shown in green. Select horizon using pills above chart.</p>
                    </div>
                    {rec && <Badge text={rec.Decision} size="lg" />}
                  </div>
                  <ForecastChart data={forecast} />
                  {!forecast && <div style={s.emptyState}>No forecast found for this item.</div>}
                  <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
                    {forecast?.ensemble && <span style={{ fontSize: 12, color: "#78716c" }}><span style={{ color: "#10b981", fontWeight: 700 }}>— </span>Ensemble (60% ARIMA + 40% Prophet, dashes green)</span>}
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
              
              {/* Search bar */}
              <div style={{ marginBottom: 16 }}>
                <input
                  type="text"
                  placeholder="Search by item ID, store, or category…"
                  value={recSearchQuery}
                  onChange={e => { setRecSearchQuery(e.target.value); setRecPage(0); }}
                  style={{
                    width: "100%",
                    padding: "10px 12px",
                    fontSize: 13,
                    border: "1px solid #e7e5e4",
                    borderRadius: 6,
                    fontFamily: "inherit",
                    boxSizing: "border-box"
                  }}
                />
              </div>

              {/* Decision filter */}
              <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
                <span style={{ fontSize: 13, color: "#78716c" }}>Filter:</span>
                {["", "Increase", "Hold", "Decrease"].map(d => (
                  <button key={d} onClick={() => { setDecFilter(d); setRecPage(0); }} style={s.pill(decFilter === d)}>{d || "All"}</button>
                ))}
                <span style={{ fontSize: 12, color: "#a8a29e", marginLeft: "auto" }}>
                  {filteredRecs.length} of {recTotal} items
                </span>
              </div>
              
              <div style={s.card}><RecommendationsTable items={paginatedRecs} /></div>
              
              {/* Pagination controls */}
              {totalPages > 1 && (
                <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 12, marginTop: 20 }}>
                  <button
                    onClick={() => setRecPage(Math.max(0, recPage - 1))}
                    disabled={recPage === 0}
                    style={{
                      padding: "8px 12px",
                      fontSize: 12,
                      border: "1px solid #e7e5e4",
                      borderRadius: 4,
                      background: recPage === 0 ? "#f5f5f4" : "#fff",
                      cursor: recPage === 0 ? "not-allowed" : "pointer",
                      color: recPage === 0 ? "#a8a29e" : "#1c1917",
                      fontWeight: 500
                    }}
                  >
                    ← Previous
                  </button>
                  <span style={{ fontSize: 12, color: "#78716c", minWidth: 120, textAlign: "center" }}>
                    Page {recPage + 1} of {totalPages}
                  </span>
                  <button
                    onClick={() => setRecPage(Math.min(totalPages - 1, recPage + 1))}
                    disabled={recPage === totalPages - 1}
                    style={{
                      padding: "8px 12px",
                      fontSize: 12,
                      border: "1px solid #e7e5e4",
                      borderRadius: 4,
                      background: recPage === totalPages - 1 ? "#f5f5f4" : "#fff",
                      cursor: recPage === totalPages - 1 ? "not-allowed" : "pointer",
                      color: recPage === totalPages - 1 ? "#a8a29e" : "#1c1917",
                      fontWeight: 500
                    }}
                  >
                    Next →
                  </button>
                </div>
              )}
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
  // ── Root & Layout ─────────────────────────────────────────────────────
  root:        { minHeight: "100vh", display: "flex", flexDirection: "column", background: "#f9f8f7", fontFamily: "system-ui, sans-serif" },
  header:      { background: "linear-gradient(135deg, #1c1917 0%, #292522 100%)", borderBottom: "1px solid #1c1917", padding: "0 24px", boxShadow: "0 2px 8px rgba(0, 0, 0, 0.08)" },
  headerInner: { maxWidth: 1400, margin: "0 auto", padding: "18px 0", display: "flex", justifyContent: "space-between", alignItems: "center" },
  h1:         { fontSize: 24, fontWeight: 700, color: "#fff", letterSpacing: "-0.5px" },
  h1Sub:      { fontSize: 12, color: "#bfbbb3", marginTop: 4, fontWeight: 400 },
  statusDot:  (color) => ({ width: 8, height: 8, borderRadius: "50%", background: color, boxShadow: `0 0 12px ${color}` }),

  // ── Tab Navigation ───────────────────────────────────────────────────
  tabNav:     { background: "#fff", borderBottom: "1px solid #e7e5e4", boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04)" },
  tabNavInner: { maxWidth: 1400, margin: "0 auto", padding: "0 24px", display: "flex", gap: 32 },
  tabButton:  (active) => ({
    padding: "12px 8px",
    fontSize: 13,
    fontWeight: active ? 600 : 500,
    color: active ? "#3b82f6" : "#78716c",
    background: "transparent",
    border: "none",
    cursor: "pointer",
    transition: "all 0.2s ease",
    position: "relative",
  }),

  // ── Layout Wrapper ───────────────────────────────────────────────────
  layoutWrapper: { display: "flex", flex: 1, maxWidth: 1400, margin: "0 auto", width: "100%", padding: "24px", gap: 24 },

  // ── Controls Panel (Left Side) ────────────────────────────────────────
  controlsPanel: { width: 200, flexShrink: 0, display: "flex", flexDirection: "column", gap: 18 },
  controlSection: { display: "flex", flexDirection: "column", gap: 8 },
  controlLabel: { fontSize: 11, fontWeight: 700, color: "#78716c", textTransform: "uppercase", letterSpacing: "0.08em" },
  controlSelect: { padding: "9px 11px", borderRadius: 8, border: "1px solid #e7e5e4", fontSize: 13, background: "#fff", cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s", boxShadow: "0 1px 2px rgba(0, 0, 0, 0.02)" },
  modelPill: (active) => ({ padding: "6px 11px", borderRadius: 6, border: `1px solid ${active ? "#3b82f6" : "#e7e5e4"}`, background: active ? "#eff6ff" : "#fff", color: active ? "#1e40af" : "#78716c", fontSize: 12, cursor: "pointer", fontWeight: active ? 600 : 500, transition: "all 0.2s" }),

  // ── Main Content ──────────────────────────────────────────────────────
  mainContent: { flex: 1, minWidth: 0, borderRadius: 12, background: "#fff", border: "1px solid #e7e5e4", padding: "28px 32px", boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)" },

  // ── Cards & Typography ────────────────────────────────────────────────
  card:       { background: "#fff", border: "1px solid #e7e5e4", borderRadius: 10, padding: "20px 24px", marginTop: 16, boxShadow: "0 1px 2px rgba(0, 0, 0, 0.03)" },
  cardGrid:   { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 14 },
  h2:         { fontSize: 22, fontWeight: 700, color: "#1c1917", marginBottom: 22, letterSpacing: "-0.3px" },
  h3:         { fontSize: 15, fontWeight: 700, color: "#1c1917", letterSpacing: "-0.2px" },

  // ── Empty States ───────────────────────────────────────────────────────
  emptyState: { background: "#f9f8f7", border: "2px dashed #e7e5e4", borderRadius: 10, padding: 48, textAlign: "center", color: "#78716c", fontSize: 14 },
  code:       { background: "#1c1917", color: "#fef3c7", borderRadius: 6, padding: "10px 12px", fontSize: 12, fontFamily: "monospace", marginTop: 14, textAlign: "left", display: "inline-block", letterSpacing: "0.5px" },

  // ── Pills & Buttons ────────────────────────────────────────────────────
  pill:       (active) => ({ padding: "6px 12px", borderRadius: 6, border: `1px solid ${active ? "#1c1917" : "#e7e5e4"}`, background: active ? "#1c1917" : "#fff", color: active ? "#fff" : "#57534e", fontSize: 12, cursor: "pointer", fontWeight: active ? 600 : 500, transition: "all 0.15s" }),
  dot:        (color) => ({ width: 8, height: 8, borderRadius: "50%", background: color, boxShadow: `0 0 12px ${color}` }),
};
