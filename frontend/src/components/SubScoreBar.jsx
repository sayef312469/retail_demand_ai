const LABELS = {
  demand_norm:    { label: "Demand",    weight: "40%", color: "#3b82f6" },
  growth_norm:    { label: "Growth",    weight: "25%", color: "#10b981" },
  stability_norm: { label: "Stability", weight: "20%", color: "#f59e0b" },
  price_norm:     { label: "Price",     weight: "15%", color: "#8b5cf6" },
};

export default function SubScoreBar({ pviData }) {
  if (!pviData) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {Object.entries(LABELS).map(([key, meta]) => {
        const val = pviData[key];
        if (val == null) return null;
        const pct = Math.round(val * 100);

        return (
          <div key={key}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 13, color: "#44403c", fontWeight: 500 }}>
                {meta.label}
                <span style={{ fontSize: 11, color: "#a8a29e", marginLeft: 4 }}>({meta.weight})</span>
              </span>
              <span style={{ fontSize: 13, fontWeight: 600, color: "#1c1917" }}>{pct}%</span>
            </div>
            <div style={{ background: "#f5f5f4", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div style={{
                width: `${pct}%`,
                height: "100%",
                background: meta.color,
                borderRadius: 4,
                transition: "width 0.6s ease",
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
