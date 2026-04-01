const STYLES = {
  High:     { bg: "#f0fdf4", color: "#15803d", border: "#bbf7d0" },
  Medium:   { bg: "#fffbeb", color: "#b45309", border: "#fde68a" },
  Low:      { bg: "#fef2f2", color: "#b91c1c", border: "#fecaca" },
  Increase: { bg: "#eff6ff", color: "#1d4ed8", border: "#bfdbfe" },
  Hold:     { bg: "#f5f3ff", color: "#6d28d9", border: "#ddd6fe" },
  Decrease: { bg: "#fef2f2", color: "#b91c1c", border: "#fecaca" },
  prophet:  { bg: "#eff6ff", color: "#1d4ed8", border: "#bfdbfe" },
  arima:    { bg: "#faf5ff", color: "#7e22ce", border: "#e9d5ff" },
};

export default function Badge({ text, size = "sm" }) {
  const s = STYLES[text] || { bg: "#f5f5f4", color: "#57534e", border: "#e7e5e4" };
  const pad = size === "lg" ? "5px 14px" : "2px 10px";
  const fs  = size === "lg" ? 14 : 12;

  return (
    <span style={{
      background:   s.bg,
      color:        s.color,
      border:       `1px solid ${s.border}`,
      borderRadius: 20,
      padding:      pad,
      fontSize:     fs,
      fontWeight:   600,
      whiteSpace:   "nowrap",
    }}>
      {text}
    </span>
  );
}
