export default function StatCard({ label, value, sub, accent }) {
  const colors = {
    blue:   { bg: "#eff6ff", text: "#1d4ed8" },
    green:  { bg: "#f0fdf4", text: "#15803d" },
    amber:  { bg: "#fffbeb", text: "#b45309" },
    red:    { bg: "#fef2f2", text: "#b91c1c" },
    purple: { bg: "#faf5ff", text: "#7e22ce" },
  };
  const c = colors[accent] || colors.blue;

  return (
    <div style={{
      background: "#fff",
      border: "1px solid #e7e5e4",
      borderRadius: 12,
      padding: "18px 20px",
      display: "flex",
      flexDirection: "column",
      gap: 4,
    }}>
      <span style={{ fontSize: 12, color: "#78716c", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </span>
      <span style={{ fontSize: 28, fontWeight: 700, color: "#1c1917" }}>{value}</span>
      {sub && (
        <span style={{
          fontSize: 12,
          background: c.bg,
          color: c.text,
          borderRadius: 6,
          padding: "2px 8px",
          alignSelf: "flex-start",
          fontWeight: 500,
        }}>
          {sub}
        </span>
      )}
    </div>
  );
}
