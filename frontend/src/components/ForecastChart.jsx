import React, { useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, ReferenceLine,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

function formatDate(ds) {
  return new Date(ds).toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

export default function ForecastChart({ data }) {
  const [selectedMonths, setSelectedMonths] = useState(3);

  if (!data) return <div style={styles.empty}>No forecast data available.</div>;

  const { actuals = {}, prophet, arima, ensemble } = data;

  // Merge all dates into one timeline
  const dateSet = new Set([
    ...(actuals.dates || []),
    ...(prophet?.dates || []),
    ...(arima?.dates   || []),
    ...(ensemble?.dates || []),
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
  const ensembleMap = Object.fromEntries((ensemble?.dates || []).map((d, i) => [d, {
    yhat: ensemble.yhat[i],
    lower: ensemble.yhat_lower?.[i],
    upper: ensemble.yhat_upper?.[i],
  }]));

  const chartData = allDates.map((d, idx) => ({
    date:         d,
    label:        formatDate(d),
    actual:       actualMap[d]    ?? null,
    ensemble:     ensembleMap[d]?.yhat  ?? null,
    prophet:      prophetMap[d]?.yhat  ?? null,
    arima:        arimaMap[d]?.yhat    ?? null,
    ensembleUpper: ensembleMap[d]?.upper ?? null,
    ensembleLower: ensembleMap[d]?.lower ?? null,
    prophetUpper: prophetMap[d]?.upper ?? null,
    prophetLower: prophetMap[d]?.lower ?? null,
    arimaUpper:   arimaMap[d]?.upper   ?? null,
    arimaLower:   arimaMap[d]?.lower   ?? null,
    _sortIndex:   idx,
  }));

  // Forecast start = first date that has prophet/arima but no actual
  const forecastStart = chartData.findIndex((r) => r.actual == null && (r.prophet != null || r.arima != null));
  const forecastStartDate = forecastStart >= 0 ? chartData[forecastStart]?.date : null;

  // Count forecast periods
  const forecastPeriods = chartData.filter(r => r.actual == null && (r.prophet != null || r.arima != null)).length;

  // Calculate summary statistics for the selected horizon
  const prophetForecast = chartData.filter(r => r.prophet != null && r.actual == null).slice(0, selectedMonths);
  const arimaForecast = chartData.filter(r => r.arima != null && r.actual == null).slice(0, selectedMonths);
  
  const prophetAvg = prophetForecast.length > 0 
    ? (prophetForecast.reduce((sum, r) => sum + (r.prophet || 0), 0) / prophetForecast.length).toFixed(2)
    : "—";
  const prophetEndpoint = prophetForecast.length > 0 
    ? prophetForecast[prophetForecast.length - 1].prophet?.toFixed(2)
    : "—";
  const arimaAvg = arimaForecast.length > 0 
    ? (arimaForecast.reduce((sum, r) => sum + (r.arima || 0), 0) / arimaForecast.length).toFixed(2)
    : "—";

  const handleMonthsChange = (months) => {
    setSelectedMonths(months);
  };

  return (
    <div>
      {/* Months Selector Pills */}
      <div style={styles.selectorContainer}>
        <div style={styles.selectorLabel}>Forecast horizon:</div>
        <div style={styles.pillContainer}>
          {[3, 6, 9, 12].map(m => (
            <button
              key={m}
              onClick={() => handleMonthsChange(m)}
              style={{
                ...styles.pill,
                ...(selectedMonths === m ? styles.pillActive : styles.pillInactive),
              }}
            >
              {m}M
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div style={{ width: "100%", height: 340, marginTop: 16 }}>
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

            {/* Reference line at forecast start */}
            {forecastStartDate && (
              <ReferenceLine
                x={forecastStartDate}
                stroke="#d4a574"
                strokeDasharray="5 5"
                label={{ value: "Forecast", position: "right", fill: "#92400e", fontSize: 11 }}
              />
            )}

            {/* Confidence bands */}
            {ensemble && (
              <Area
                dataKey="ensembleUpper"
                stroke="none"
                fill="#10b981"
                fillOpacity={0.12}
                legendType="none"
                name="Ensemble CI"
              />
            )}
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

            {/* Ensemble forecast (60% ARIMA + 40% Prophet) */}
            {ensemble && (
              <Line
                dataKey="ensemble"
                name="Ensemble forecast (60% ARIMA + 40% Prophet)"
                stroke="#10b981"
                strokeWidth={2.5}
                strokeDasharray="4 4"
                dot={false}
              />
            )}

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

      {/* Summary Cards */}
      <div style={styles.summaryContainer}>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Prophet Avg</div>
          <div style={styles.summaryValue}>{prophetAvg}</div>
          <div style={styles.summaryHint}>{selectedMonths} months</div>
        </div>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Prophet M{selectedMonths}</div>
          <div style={styles.summaryValue}>{prophetEndpoint}</div>
          <div style={styles.summaryHint}>endpoint</div>
        </div>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>ARIMA Avg</div>
          <div style={styles.summaryValue}>{arimaAvg}</div>
          <div style={styles.summaryHint}>{selectedMonths} months</div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  empty: { 
    padding: 40, 
    textAlign: "center", 
    color: "#78716c", 
    fontSize: 14 
  },
  selectorContainer: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    paddingBottom: 8,
  },
  selectorLabel: {
    fontSize: 13,
    fontWeight: 600,
    color: "#57534e",
  },
  pillContainer: {
    display: "flex",
    gap: 8,
  },
  pill: {
    padding: "6px 14px",
    fontSize: 12,
    fontWeight: 500,
    border: "1px solid #e7e5e4",
    borderRadius: 20,
    cursor: "pointer",
    transition: "all 0.2s",
    backgroundColor: "transparent",
  },
  pillActive: {
    backgroundColor: "#3b82f6",
    color: "white",
    borderColor: "#3b82f6",
  },
  pillInactive: {
    backgroundColor: "transparent",
    color: "#57534e",
    borderColor: "#e7e5e4",
  },
  summaryContainer: {
    display: "flex",
    gap: 16,
    marginTop: 20,
    paddingTop: 16,
    borderTop: "1px solid #e7e5e4",
  },
  summaryCard: {
    flex: 1,
    padding: 12,
    backgroundColor: "#fafaf8",
    borderRadius: 8,
    border: "1px solid #e7e5e4",
  },
  summaryLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: "#a8a29e",
    textTransform: "uppercase",
    marginBottom: 4,
  },
  summaryValue: {
    fontSize: 18,
    fontWeight: 700,
    color: "#1c1917",
    marginBottom: 2,
  },
  summaryHint: {
    fontSize: 10,
    color: "#78716c",
  },
};
