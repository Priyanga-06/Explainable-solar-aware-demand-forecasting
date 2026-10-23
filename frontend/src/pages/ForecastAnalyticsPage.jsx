import React, { useState } from 'react'
import { useDashboard } from '../hooks/useDashboard'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line
} from 'recharts'

export default function ForecastAnalyticsPage() {
  const { data, loading, error, refresh } = useDashboard()
  const [selectedZone, setSelectedZone] = useState('Residential')

  if (loading) return <LoadingState message="Loading analytics..." />
  if (error?.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} onRetry={refresh} />
  if (!data) return null

  const { shap_explanations, peak_cause_explanation, forecast_timestamps,
          zone_forecasts, peak_severity } = data

  const zoneExplanation = shap_explanations?.[selectedZone] || {}
  const topFactors = zoneExplanation.top_factors || []

  const shapChartData = topFactors.map(f => ({
    feature: f.feature,
    shap: Math.abs(f.shap_value),
    direction: f.direction,
    impact: f.impact,
  }))

  const hourlyData = (forecast_timestamps || []).map((ts, i) => ({
    time: ts.slice(11, 16),
    severity: peak_severity?.net_demand_24h?.[i] ?? 0,
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Forecast & Analytics</div>
          <div className="page-subtitle">SHAP explainability, peak cause analysis, feature importance</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['Residential', 'Commercial', 'Industrial'].map(z => (
            <button
              key={z}
              onClick={() => setSelectedZone(z)}
              className={selectedZone === z ? 'btn-primary' : 'btn-secondary'}
              style={{ fontSize: 12, padding: '5px 12px' }}
            >
              {z}
            </button>
          ))}
        </div>
      </div>

      {/* Peak cause explanation */}
      {peak_cause_explanation?.peak_cause_explanation && (
        <div className="card" style={{ borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Peak Cause Explanation — {selectedZone}</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            {peak_cause_explanation.peak_cause_explanation}
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* SHAP Top Factors */}
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Top Influencing Factors</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 14 }}>
            SHAP values — marginal contribution to the demand forecast
          </div>

          {topFactors.length === 0 ? (
            <div className="text-muted" style={{ fontSize: 12 }}>No SHAP data available. Train models first.</div>
          ) : (
            topFactors.map((f, i) => (
              <div key={i} style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 12 }}>{f.feature}</span>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                      {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                    </span>
                    <span className="badge" style={{
                      fontSize: 9,
                      background: f.direction === 'positive' ? 'rgba(248,81,73,0.15)' : 'rgba(63,185,80,0.15)',
                      color: f.direction === 'positive' ? 'var(--accent-red)' : 'var(--accent-green)',
                    }}>
                      {f.direction === 'positive' ? '▲ Demand' : '▼ Demand'}
                    </span>
                  </div>
                </div>
                <div style={{
                  height: 6, borderRadius: 3,
                  background: 'var(--bg-secondary)',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.min(Math.abs(f.shap_value) / 20 * 100, 100)}%`,
                    background: f.direction === 'positive'
                      ? 'var(--accent-red)' : 'var(--accent-green)',
                    borderRadius: 3,
                  }} />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
                  {f.plain_explanation}
                </div>
              </div>
            ))
          )}

          {zoneExplanation.technical_note && (
            <div style={{ marginTop: 12, fontSize: 10, color: 'var(--text-muted)', borderTop: '1px solid var(--border)', paddingTop: 10 }}>
              ⓘ {zoneExplanation.technical_note}
            </div>
          )}
        </div>

        {/* SHAP bar chart */}
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 14 }}>Feature Importance (|SHAP|)</div>
          {shapChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={shapChartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} width={120} />
                <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }} />
                <Bar dataKey="shap" fill="var(--accent-purple)" name="|SHAP Value|" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-muted" style={{ fontSize: 12, padding: 20 }}>Train models to see SHAP feature importance.</div>
          )}
        </div>
      </div>

      {/* Net demand severity over 24 hours */}
      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 14 }}>Net Grid Demand — 24-Hour Profile</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={hourlyData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval={3} />
            <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }} />
            <Line type="monotone" dataKey="severity" stroke="var(--accent-orange)" dot={false} strokeWidth={2} name="Net Demand" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
