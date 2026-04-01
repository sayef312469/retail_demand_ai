export default function PVIGauge({ pvi, viability }) {
  const clamped = Math.min(100, Math.max(0, pvi || 0));
  const color =
    viability === "High"   ? "#15803d" :
    viability === "Medium" ? "#b45309" : "#b91c1c";

  // SVG arc gauge
  const R   = 60;
  const cx  = 80;
  const cy  = 80;
  const startAngle = 180;
  const sweep = (clamped / 100) * 180;

  const toRad = (deg) => (deg * Math.PI) / 180;
  const px = (a) => cx + R * Math.cos(toRad(a));
  const py = (a) => cy + R * Math.sin(toRad(a));

  const endAngle = 180 + sweep;
  const largeArc = sweep > 180 ? 1 : 0;

  const bgPath = `M ${px(180)} ${py(180)} A ${R} ${R} 0 1 1 ${px(360)} ${py(360)}`;
  const fgPath = `M ${px(180)} ${py(180)} A ${R} ${R} 0 ${largeArc} 1 ${px(endAngle)} ${py(endAngle)}`;

  return (
    <div style={{ textAlign: "center" }}>
      <svg viewBox="0 0 160 100" width="200" height="125">
        {/* Background arc */}
        <path d={bgPath} fill="none" stroke="#e7e5e4" strokeWidth="12" strokeLinecap="round" />
        {/* Value arc */}
        <path d={fgPath} fill="none" stroke={color} strokeWidth="12" strokeLinecap="round" />
        {/* Value text */}
        <text x={cx} y={cy - 5} textAnchor="middle" fontSize="28" fontWeight="700" fill="#1c1917">
          {clamped.toFixed(1)}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" fontSize="11" fill="#78716c">
          out of 100
        </text>
      </svg>

      {/* Sub-score bars */}
    </div>
  );
}
