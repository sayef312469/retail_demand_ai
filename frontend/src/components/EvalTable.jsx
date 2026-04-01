export default function EvalTable({ summary }) {
  if (!summary?.length) {
    return <div style={{ color: "#78716c", fontSize: 14, padding: 24 }}>No evaluation data yet. Run evaluate.py first.</div>;
  }

  const metrics = ["MAE", "RMSE", "MAPE", "bias"];
  const models  = [...new Set(summary.map((r) => r.model))];

  // pivot: metric → { model: { mean, median } }
  const pivot = {};
  metrics.forEach((m) => {
    pivot[m] = {};
    models.forEach((mo) => {
      const row = summary.find((r) => r.model === mo && r.metric === m);
      pivot[m][mo] = row || null;
    });
  });

  const metaColor = { MAE: "#3b82f6", RMSE: "#8b5cf6", MAPE: "#f59e0b", bias: "#10b981" };

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#f5f5f4" }}>
            <th style={th}>Metric</th>
            {models.map((m) => (
              <th key={m} colSpan={2} style={{ ...th, textTransform: "capitalize" }}>{m}</th>
            ))}
          </tr>
          <tr style={{ background: "#fafaf9" }}>
            <th style={th}></th>
            {models.map((m) => (
              <>
                <th key={m + "_mean"}   style={{ ...th, fontWeight: 400, color: "#78716c" }}>Mean</th>
                <th key={m + "_median"} style={{ ...th, fontWeight: 400, color: "#78716c" }}>Median</th>
              </>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric} style={{ borderTop: "1px solid #f5f5f4" }}>
              <td style={{ ...td, fontWeight: 600, color: metaColor[metric] || "#1c1917" }}>{metric}</td>
              {models.map((m) => {
                const d = pivot[metric][m];
                return (
                  <>
                    <td key={m + "_mean"}   style={td}>{d ? d.mean.toLocaleString()   : "—"}</td>
                    <td key={m + "_median"} style={td}>{d ? d.median.toLocaleString() : "—"}</td>
                  </>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: 11, color: "#a8a29e", marginTop: 8 }}>
        Evaluated on last 3 months held out from each series.
        MAPE skips zero-actual months. Lower MAE/RMSE/MAPE and bias near 0 = better.
      </p>
    </div>
  );
}

const th = {
  padding: "10px 14px",
  textAlign: "left",
  fontWeight: 600,
  fontSize: 12,
  color: "#44403c",
  borderBottom: "1px solid #e7e5e4",
};
const td = {
  padding: "10px 14px",
  color: "#1c1917",
};
