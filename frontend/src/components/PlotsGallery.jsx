import { useState, useEffect } from "react";

const PLOT_DESCRIPTIONS = {
  "forecast_vs_actual_top5":  "Forecast vs actual sales for the 5 highest-PVI items. Solid line = actuals, dashed = Prophet, dotted = ARIMA. Shaded bands = 95% confidence intervals.",
  "mae_rmse_comparison":      "MAE and RMSE comparison between Prophet and ARIMA. Lower is better. Error bars show standard deviation across all items.",
  "mape_wape_comparison":     "MAPE (scale-independent error %) and WAPE (volume-weighted error %). WAPE is more robust when some items have zero sales.",
  "r2_distribution":          "Box plot of R² scores per model. R²=1 means perfect forecast; R²=0 means no better than predicting the average; R²<0 means the model is actively harmful.",
  "residual_distribution":    "Histogram of forecast bias (predicted − actual) per model. A distribution centred on zero means the model neither over- nor under-predicts systematically.",
  "actual_vs_predicted":      "Scatter plot of actual vs predicted sales. Points on the dashed diagonal = perfect forecast. The fitted regression line shows systematic over/under-prediction.",
  "pvi_distribution":         "PVI score histogram coloured by viability zone (green=High, amber=Medium, red=Low). Boundary lines at 33 and 67.",
  "pvi_subscores_heatmap":    "Heatmap of average PVI sub-scores (demand, growth, stability, price) per product category. Green = high score, red = low score.",
  "decision_breakdown":       "Stock recommendation breakdown per store (stacked bar) and overall (donut chart).",
};

export default function PlotsGallery() {
  const [plots,    setPlots]    = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    fetch("/plots")
      .then(r => r.json())
      .then(d => { setPlots(d.plots || []); setLoading(false); })
      .catch(() => { setError("Could not load plots. Run python src/plot_reports.py first."); setLoading(false); });
  }, []);

  if (loading) return <div style={styles.state}>Loading plots…</div>;
  if (error)   return (
    <div style={styles.errorBox}>
      <p style={{ fontWeight: 600, marginBottom: 6 }}>No plots yet</p>
      <p>{error}</p>
      <pre style={styles.code}>python src/plot_reports.py</pre>
    </div>
  );
  if (!plots.length) return (
    <div style={styles.errorBox}>
      <p style={{ fontWeight: 600, marginBottom: 6 }}>No plots generated yet</p>
      <p>Run the plot generator to create all 9 charts:</p>
      <pre style={styles.code}>python src/plot_reports.py</pre>
    </div>
  );

  const getKey = (filename) => filename.replace(".png", "");

  return (
    <div>
      {/* Thumbnail grid */}
      <div style={styles.grid}>
        {plots.map(plot => (
          <div key={plot.filename} style={styles.thumb(selected?.filename === plot.filename)}
               onClick={() => setSelected(selected?.filename === plot.filename ? null : plot)}>
            <img src={plot.url} alt={plot.title} style={styles.thumbImg}
                 onError={e => { e.target.style.display = "none"; }} />
            <p style={styles.thumbLabel}>{plot.title}</p>
          </div>
        ))}
      </div>

      {/* Expanded view */}
      {selected && (
        <div style={styles.expanded}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600, color: "#1c1917" }}>{selected.title}</h3>
              <p style={{ fontSize: 13, color: "#78716c", marginTop: 4, maxWidth: 640, lineHeight: 1.6 }}>
                {PLOT_DESCRIPTIONS[getKey(selected.filename)] || ""}
              </p>
            </div>
            <button onClick={() => setSelected(null)} style={styles.closeBtn}>✕ Close</button>
          </div>
          <img src={selected.url} alt={selected.title}
               style={{ width: "100%", borderRadius: 8, border: "1px solid #e7e5e4" }} />
          <p style={{ fontSize: 11, color: "#a8a29e", marginTop: 8 }}>
            Saved at: outputs/plots/{selected.filename} — re-run plot_reports.py to refresh.
          </p>
        </div>
      )}
    </div>
  );
}

const styles = {
  grid:      { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12, marginBottom: 20 },
  thumb:     (active) => ({
    background: "#fff", border: `2px solid ${active ? "#1c1917" : "#e7e5e4"}`,
    borderRadius: 10, overflow: "hidden", cursor: "pointer",
    transition: "border-color 0.15s, box-shadow 0.15s",
    boxShadow: active ? "0 2px 12px rgba(0,0,0,0.1)" : "none",
  }),
  thumbImg:  { width: "100%", height: 130, objectFit: "cover", display: "block" },
  thumbLabel:{ fontSize: 11, fontWeight: 500, color: "#44403c", padding: "6px 10px", textAlign: "center" },
  expanded:  { background: "#fff", border: "1px solid #e7e5e4", borderRadius: 12, padding: "20px 24px" },
  closeBtn:  { padding: "6px 14px", border: "1px solid #e7e5e4", borderRadius: 8, background: "#fff", cursor: "pointer", fontSize: 12, color: "#57534e", flexShrink: 0 },
  state:     { padding: 40, textAlign: "center", color: "#78716c", fontSize: 14 },
  errorBox:  { background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 10, padding: "16px 20px", fontSize: 13, color: "#78350f" },
  code:      { background: "#1c1917", color: "#d4d0ca", borderRadius: 8, padding: "10px 14px", fontFamily: "monospace", fontSize: 12, marginTop: 10, display: "inline-block" },
};
