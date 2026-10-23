import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'
import { useModelMetrics } from '../hooks/useDashboard'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'

export default function ModelComparisonPage() {
  const { data, loading, error } = useModelMetrics()

  if (loading) return <LoadingState message="Loading model metrics..." />
  if (error?.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} />
  if (!data) return null

  const rows = data.demand_model_comparison || []
  const solar = data.solar_model

  // Build chart data per zone
  const zones = ['Residential', 'Commercial', 'Industrial']
  const maeData = zones.map(z => {
    const row = rows.reduce((acc, r) => {
      if (r.zone === z) acc[r.model] = r.MAE
      return acc
    }, { zone: z })
    return row
  })
  const r2Data = zones.map(z => {
    const row = rows.reduce((acc, r) => {
      if (r.zone === z) acc[r.model] = r.R2
      return acc
    }, { zone: z })
    return row
  })

  const metricDefs = data.metric_definitions || {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Model Comparison</div>
          <div className="page-subtitle">
            GRU vs XGBoost Baseline — actual evaluation metrics from the held-out test set
          </div>
        </div>
        <span className="badge badge-blue" style={{ fontSize: 11 }}>Actual Test Set Metrics</span>
      </div>

      <div style={{ background: 'rgba(88,166,255,0.06)', border: '1px solid rgba(88,166,255,0.2)', borderRadius: 8, padding: '10px 16px', fontSize: 12, color: 'var(--text-secondary)' }}>
        ⓘ {data.data_source_note}
      </div>

      {/* Comparison table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', fontWeight: 600 }}>
          Demand Forecasting — GRU vs XGBoost Baseline
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>
              {['Zone', 'Model', 'MAE (MW)', 'RMSE (MW)', 'MAPE (%)', 'R²'].map(h => (
                <th key={h} style={{
                  padding: '8px 14px', textAlign: 'left', fontSize: 11, fontWeight: 600,
                  color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em',
                  borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)'
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} style={{ borderBottom: '1px solid rgba(48,54,61,0.5)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 500 }}>{row.zone}</td>
                <td style={{ padding: '10px 14px' }}>
                  <span className="badge" style={{
                    background: row.model === 'GRU' ? 'rgba(88,166,255,0.15)' : 'rgba(219,109,40,0.15)',
                    color: row.model === 'GRU' ? 'var(--accent-blue)' : 'var(--accent-orange)',
                  }}>{row.model}</span>
                </td>
                <td style={{ padding: '10px 14px', color: row.MAE ? 'var(--text-primary)' : 'var(--text-muted)', fontFamily: 'monospace' }}>
                  {row.MAE?.toFixed(3) ?? '—'}
                </td>
                <td style={{ padding: '10px 14px', fontFamily: 'monospace' }}>
                  {row.RMSE?.toFixed(3) ?? '—'}
                </td>
                <td style={{ padding: '10px 14px', fontFamily: 'monospace' }}>
                  {row.MAPE?.toFixed(2) ?? '—'}
                </td>
                <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: row.R2 > 0.8 ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                  {row.R2?.toFixed(4) ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.some(r => !r.MAE) && (
          <div style={{ padding: '12px 16px', fontSize: 12, color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
            ⓘ Dashes indicate models not yet trained. Run <code>python backend/training/train_all.py</code>.
          </div>
        )}
      </div>

      {/* MAE chart */}
      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 14 }}>MAE Comparison by Zone</div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={maeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="zone" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
            <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="GRU"              fill="#58a6ff" name="GRU" />
            <Bar dataKey="XGBoost Baseline" fill="#db6d28" name="XGBoost Baseline" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Solar model */}
      {solar?.metrics && (
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12 }}>Solar Model — XGBoost</div>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            {Object.entries(solar.metrics).map(([k, v]) => (
              v !== null && (
                <div key={k}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>{k}</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--accent-yellow)', fontFamily: 'monospace' }}>
                    {typeof v === 'number' ? v.toFixed(3) : v}
                  </div>
                </div>
              )
            ))}
          </div>
        </div>
      )}

      {/* Metric definitions */}
      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 10 }}>Metric Definitions</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {Object.entries(metricDefs).map(([k, v]) => (
            <div key={k} style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--text-primary)' }}>{k}</strong>: {v}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
