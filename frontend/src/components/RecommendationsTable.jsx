import Badge from "./Badge";

const DECISION_ICON = { Increase: "↑", Hold: "→", Decrease: "↓" };
const CONF_COLOR    = { High: "#15803d", Medium: "#b45309", Low: "#b91c1c" };

export default function RecommendationsTable({ items, loading }) {
  if (loading) return <div style={styles.state}>Loading…</div>;
  if (!items?.length) return <div style={styles.state}>No recommendations found.</div>;

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#f5f5f4" }}>
            {["Item", "Store", "Category", "PVI", "Viability", "Decision", "Confidence", "Anomaly", "Explanation"].map((h) => (
              <th key={h} style={th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((row, i) => (
            <tr key={i} style={{ borderTop: "1px solid #f5f5f4", background: i % 2 === 0 ? "#fff" : "#fafaf9" }}>
              <td style={{ ...td, fontWeight: 600, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {row.item_id}
              </td>
              <td style={td}>{row.store_id}</td>
              <td style={td}>{row.cat_id}</td>
              <td style={{ ...td, fontWeight: 700, color: pviColor(row.PVI) }}>
                {Number(row.PVI).toFixed(1)}
              </td>
              <td style={td}><Badge text={row.Viability} /></td>
              <td style={td}>
                <span style={{ fontWeight: 700, marginRight: 4 }}>{DECISION_ICON[row.Decision]}</span>
                <Badge text={row.Decision} />
              </td>
              <td style={{ ...td, fontWeight: 600, color: CONF_COLOR[row.Confidence] }}>
                {row.Confidence}
              </td>
              <td style={{ ...td, textAlign: "center" }}>
                {row.has_anomaly ? (
                  <span title={`${(row.anomaly_pct * 100).toFixed(0)}% of months flagged`}
                        style={{ fontSize: 16 }}>⚠️</span>
                ) : "—"}
              </td>
              <td style={{ ...td, color: "#57534e", maxWidth: 320, fontSize: 11 }}>
                {row.Explanation}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function pviColor(pvi) {
  if (pvi >= 67) return "#15803d";
  if (pvi >= 33) return "#b45309";
  return "#b91c1c";
}

const th = { padding: "10px 12px", textAlign: "left", fontSize: 12, fontWeight: 600, color: "#44403c", borderBottom: "1px solid #e7e5e4", whiteSpace: "nowrap" };
const td = { padding: "10px 12px", color: "#1c1917", verticalAlign: "top" };
const styles = { state: { padding: 32, textAlign: "center", color: "#78716c", fontSize: 14 } };
