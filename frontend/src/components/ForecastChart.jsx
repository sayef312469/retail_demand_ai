import {
  ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

function formatDate(ds) {
  return new Date(ds).toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

export default function ForecastChart({ data }) {
  if (!data) return <div style={styles.empty}>No forecast data available.</div>;

  const { actuals = {}, prophet, arima } = data;

  // Merge all dates into one timeline
  const dateSet = new Set([
    ...(actuals.dates || []),
    ...(prophet?.dates || []),
    ...(arima?.dates   || []),
  ]);
  const allDates = Array.from(dateSet).sort();

  const actualMap  = Object.fromEntries((actuals.dates  || []).map((d, i) => [d, actuals.sales[i]]));
  const prophetMap = Object.fromEntries((prophet?.dates || []).map((d, i) => [d, {
    yhat: prophet.yhat[i],
    lower: prophet.yhat_lower?.[i],
    upper: prophet.yhat_upper?.[i],
  }]));
  const arimaMap   = Object.fromEntries((arima?.dates   || []).map((d, i) => [d, {
    yhat: arima.yhat[i],
    lower: arima.yhat_lower?.[i],
    upper: arima.yhat_upper?.[i],
  }]));

  const chartData = allDates.map((d) => ({
    date:         d,
    label:        formatDate(d),
    actual:       actualMap[d]    ?? null,
    prophet:      prophetMap[d]?.yhat  ?? null,
    arima:        arimaMap[d]?.yhat    ?? null,
    prophetUpper: prophetMap[d]?.upper ?? null,
    prophetLower: prophetMap[d]?.lower ?? null,
    arimaUpper:   arimaMap[d]?.upper   ?? null,
    arimaLower:   arimaMap[d]?.lower   ?? null,
  }));

  // Forecast start = first date that has prophet/arima but no actual
  const forecastStart = chartData.findIndex((r) => r.actual == null && (r.prophet != null || r.arima != null));

  return (
    <div style={{ width: "100%", height: 340 }}>
      <ResponsiveContainer>
        <ComposedChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f4" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: "#78716c" }}
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fontSize: 11, fill: "#78716c" }} width={56} />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e7e5e4" }}
            formatter={(v) => v != null ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />

          {/* Vertical line at forecast start */}
          {forecastStart > 0 && chartData[forecastStart] && (
            <Area dataKey="_divider" fill="none" stroke="none" />
          )}

          {/* Confidence bands */}
          {prophet && (
            <Area
              dataKey="prophetUpper"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.08}
              legendType="none"
              name="Prophet CI"
            />
          )}
          {arima && (
            <Area
              dataKey="arimaUpper"
              stroke="none"
              fill="#7c3aed"
              fillOpacity={0.08}
              legendType="none"
              name="ARIMA CI"
            />
          )}

          {/* Actuals */}
          <Line
            dataKey="actual"
            name="Actual sales"
            stroke="#1c1917"
            strokeWidth={2}
            dot={false}
            connectNulls={false}
          />

          {/* Prophet forecast */}
          {prophet && (
            <Line
              dataKey="prophet"
              name="Prophet forecast"
              stroke="#3b82f6"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
            />
          )}

          {/* ARIMA forecast */}
          {arima && (
            <Line
              dataKey="arima"
              name="ARIMA forecast"
              stroke="#7c3aed"
              strokeWidth={2}
              strokeDasharray="3 3"
              dot={false}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles = {
  empty: { padding: 40, textAlign: "center", color: "#78716c", fontSize: 14 },
};
