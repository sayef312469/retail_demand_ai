import { useState } from "react";
import ForecastChart from "./ForecastChart";

function _pviColor(score) {
  if (score >= 67) return "#059669";
  if (score >= 33) return "#d97706";
  return "#dc2626";
}

function _decisionColor(decision) {
  if (decision === "Increase") return "#059669";
  if (decision === "Decrease") return "#dc2626";
  return "#2563eb";
}

export default function UploadForecast() {
  const [file,      setFile]      = useState(null);
  const [dateCol,   setDateCol]   = useState("date");
  const [salesCol,  setSalesCol]  = useState("sales");
  const [periods,   setPeriods]   = useState(3);
  const [model,     setModel]     = useState("both");
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [preview,   setPreview]   = useState(null);

  async function handlePreview(f) {
    setFile(f); setResult(null); setError(null); setPreview(null);
    const form = new FormData();
    form.append("file", f);
    try {
      const res  = await fetch("/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setPreview(data);
      // auto-fill column names if detected
      if (data.date_cols?.length)    setDateCol(data.date_cols[0]);
      if (data.numeric_cols?.length) setSalesCol(data.numeric_cols[0]);
    } catch (e) { setError(e.message); }
  }

  async function handleForecast() {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const form = new FormData();
    form.append("file", file);
    const params = new URLSearchParams({ date_col: dateCol, sales_col: salesCol, periods, model });
    try {
      const res  = await fetch(`/upload/forecast?${params}`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Forecast failed");
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  const s = styles;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Explainer */}
      <div style={s.info}>
        <p style={{ fontWeight: 600, marginBottom: 6 }}>What this does</p>
        <p>Upload your own historical sales CSV and get demand forecasts using Prophet and/or ARIMA models for the next 1–12 months. Your file needs at least a date column and a sales/quantity column — everything else is optional.</p>
        <p style={{ marginTop: 6, color: "#78716c", fontSize: 12 }}>
          Example columns: <code>date, sales</code> or <code>week_start_date, Weekly_Sales</code> or <code>month, quantity</code>
        </p>
      </div>

      {/* Upload zone */}
      <div style={s.card}>
        <label style={s.uploadZone}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>+</div>
          <div style={{ fontWeight: 600, color: "#1c1917" }}>
            {file ? file.name : "Click to choose your CSV file"}
          </div>
          <div style={{ fontSize: 12, color: "#78716c", marginTop: 4 }}>
            {file ? `${(file.size/1024).toFixed(1)} KB` : "CSV format only"}
          </div>
          <input type="file" accept=".csv" style={{ display: "none" }}
            onChange={e => e.target.files[0] && handlePreview(e.target.files[0])} />
        </label>

        {preview && (
          <div style={{ marginTop: 16 }}>
            <p style={{ fontSize: 13, color: "#57534e", marginBottom: 10 }}>
              {preview.rows} rows detected. Configure columns then run forecast:
            </p>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
              <div>
                <label style={s.label}>Date column</label>
                <select style={s.select} value={dateCol} onChange={e => setDateCol(e.target.value)}>
                  {preview.columns.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label style={s.label}>Sales column</label>
                <select style={s.select} value={salesCol} onChange={e => setSalesCol(e.target.value)}>
                  {preview.columns.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label style={s.label}>Months ahead</label>
                <select style={s.select} value={periods} onChange={e => setPeriods(Number(e.target.value))}>
                  {[1,2,3,6,9,12].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div>
                <label style={s.label}>Model</label>
                <select style={s.select} value={model} onChange={e => setModel(e.target.value)}>
                  <option value="both">Both (Prophet + ARIMA)</option>
                  <option value="prophet">Prophet only</option>
                  <option value="arima">ARIMA only</option>
                </select>
              </div>
              <button style={s.btn} onClick={handleForecast} disabled={loading}>
                {loading ? "Forecasting…" : "Run forecast →"}
              </button>
            </div>

            {/* Data preview table */}
            <div style={{ marginTop: 14, overflowX: "auto" }}>
              <p style={{ fontSize: 11, color: "#a8a29e", marginBottom: 6 }}>First 5 rows of your file:</p>
              <table style={{ fontSize: 12, borderCollapse: "collapse", minWidth: 400 }}>
                <thead>
                  <tr>{preview.columns.map(c => (
                    <th key={c} style={{ padding: "5px 10px", background: "#f5f5f4", borderBottom: "1px solid #e7e5e4", textAlign: "left", fontWeight: 600, color: "#44403c" }}>{c}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {(preview.preview || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #f5f5f4" }}>
                      {preview.columns.map(c => (
                        <td key={c} style={{ padding: "5px 10px", color: "#57534e" }}>{String(row[c] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {error && <div style={s.error}>{error}</div>}

      {/* Forecast result */}
      {result && (
        <div style={s.card}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>{result.filename} — Forecast & Recommendation</h3>
              <p style={{ fontSize: 12, color: "#78716c" }}>
                {result.rows_used} monthly data points · {result.forecast_periods}-month horizon
                · date col: <code>{result.date_col_used}</code> · sales col: <code>{result.sales_col_used}</code>
              </p>
            </div>
          </div>

          {/* Recommendation Panel */}
          {result.recommendation && !result.recommendation.error && (
            <div style={{ ...s.recPanel, marginBottom: 20 }}>
              <div style={{ display: "flex", gap: 24, alignItems: "flex-start" }}>
                {/* PVI & Decision */}
                <div>
                  <div style={{ fontSize: 11, color: "#78716c", textTransform: "uppercase", fontWeight: 600, marginBottom: 8 }}>PVI Score</div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                    <div style={{ fontSize: 32, fontWeight: 700, color: _pviColor(result.recommendation.pvi_score) }}>
                      {result.recommendation.pvi_score}
                    </div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#57534e" }}>
                      / 100 <span style={{ fontSize: 11, color: "#a8a29e" }}>({result.recommendation.viability})</span>
                    </div>
                  </div>
                </div>

                {/* Decision */}
                <div>
                  <div style={{ fontSize: 11, color: "#78716c", textTransform: "uppercase", fontWeight: 600, marginBottom: 8 }}>Decision</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: _decisionColor(result.recommendation.decision) }}>
                    {result.recommendation.decision}
                  </div>
                  <div style={{ fontSize: 11, color: "#78716c", marginTop: 4 }}>
                    Confidence: <strong>{result.recommendation.confidence}</strong>
                  </div>
                </div>

                {/* Anomaly */}
                <div>
                  <div style={{ fontSize: 11, color: "#78716c", textTransform: "uppercase", fontWeight: 600, marginBottom: 8 }}>Anomaly</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 24, height: 24, borderRadius: 4, background: result.recommendation.has_anomaly ? "#fca5a5" : "#bbf7d0", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: result.recommendation.has_anomaly ? "#7f1d1d" : "#166534" }}>
                        {result.recommendation.has_anomaly ? "✓" : "—"}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#1c1917" }}>
                        {result.recommendation.has_anomaly ? "Detected" : "None"}
                      </div>
                      {result.recommendation.has_anomaly && (
                        <div style={{ fontSize: 11, color: "#78716c" }}>{result.recommendation.anomaly_pct}% of months</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Sub-scores bar chart */}
              {result.recommendation.sub_scores && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid #e7e5e4" }}>
                  <div style={{ display: "flex", gap: 16, fontSize: 12 }}>
                    {Object.entries(result.recommendation.sub_scores).map(([key, val]) => (
                      <div key={key} style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: "#57534e", textTransform: "capitalize" }}>{key}</div>
                          <div style={{ fontSize: 11, color: "#a8a29e" }}>{val.toFixed(2)}</div>
                        </div>
                        <div style={{ width: "100%", height: 6, background: "#e7e5e4", borderRadius: 3, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${val * 100}%`, background: "#3b82f6" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Explanation */}
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid #e7e5e4", fontSize: 12, color: "#57534e", lineHeight: 1.6 }}>
                {result.recommendation.explanation}
              </div>
            </div>
          )}

          {result.recommendation?.error && (
            <div style={{ ...s.error, marginBottom: 16 }}>
              <strong>Recommendation Error:</strong> {result.recommendation.error}
            </div>
          )}

          {/* Prophet forecast display */}
          {result.prophet && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <div style={{ width: 12, height: 12, background: "#3b82f6", borderRadius: 2 }} />
                <h4 style={{ fontSize: 13, fontWeight: 600 }}>Prophet Forecast</h4>
              </div>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {result.prophet.yhat?.map((v, i) => (
                  <div key={i} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "#78716c" }}>Month {i + 1}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#3b82f6" }}>{v.toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ARIMA forecast display */}
          {result.arima && !result.arima.error && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <div style={{ width: 12, height: 12, background: "#7c3aed", borderRadius: 2 }} />
                <h4 style={{ fontSize: 13, fontWeight: 600 }}>ARIMA Forecast {result.arima.model_order && `${result.arima.model_order}`}</h4>
              </div>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {result.arima.yhat?.map((v, i) => (
                  <div key={i} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "#78716c" }}>Month {i + 1}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#7c3aed" }}>{v.toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.arima?.error && (
            <div style={{ marginBottom: 16, ...s.error }}>
              <strong>ARIMA:</strong> {result.arima.error}
            </div>
          )}

          <ForecastChart data={{ 
            actuals: result.actuals, 
            prophet: result.prophet,
            arima: result.arima && !result.arima.error ? result.arima : undefined
          }} />

          <p style={{ fontSize: 11, color: "#a8a29e", marginTop: 10 }}>{result.note}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: { background: "#fff", border: "1px solid #e7e5e4", borderRadius: 12, padding: "20px 24px" },
  recPanel: { background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: "16px" },
  info: { background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 10, padding: "14px 18px", fontSize: 13, color: "#1e3a5f", lineHeight: 1.6 },
  uploadZone: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", border: "2px dashed #e7e5e4", borderRadius: 10, padding: "32px 20px", cursor: "pointer", background: "#fafaf9", transition: "border-color 0.2s" },
  label: { display: "block", fontSize: 11, fontWeight: 600, color: "#78716c", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 },
  select: { padding: "8px 10px", borderRadius: 8, border: "1px solid #e7e5e4", fontSize: 13, background: "#fff", cursor: "pointer" },
  btn: { padding: "9px 20px", borderRadius: 8, border: "none", background: "#1c1917", color: "#fff", fontWeight: 600, fontSize: 13, cursor: "pointer" },
  error: { background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#7f1d1d" },
};
