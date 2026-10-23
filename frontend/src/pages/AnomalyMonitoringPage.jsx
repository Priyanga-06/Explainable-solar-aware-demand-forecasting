import React, { useState } from 'react'
import { AlertTriangle, CheckCircle } from 'lucide-react'
import { useDashboard } from '../hooks/useDashboard'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'

export default function AnomalyMonitoringPage() {
  const { data, loading, error, refresh } = useDashboard()
  const [zoneFilter, setZoneFilter] = useState('All')

  if (loading) return <LoadingState message="Loading anomaly data..." />
  if (error?.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} onRetry={refresh} />
  if (!data) return null

  const anomalies = data.anomalies || []
  const filtered = zoneFilter === 'All' ? anomalies : anomalies.filter(a => a.zone === zoneFilter)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Anomaly Monitoring</div>
          <div className="page-subtitle">
            Unusual consumption patterns detected by Isolation Forest — per zone
          </div>
        </div>
        <select value={zoneFilter} onChange={e => setZoneFilter(e.target.value)}>
          {['All', 'Residential', 'Commercial', 'Industrial'].map(z => (
            <option key={z}>{z}</option>
          ))}
        </select>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 600 }}>Detected Anomalies</span>
          <span className="text-muted" style={{ fontSize: 12 }}>{filtered.length} result(s)</span>
        </div>

        {filtered.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 24, color: 'var(--text-muted)' }}>
            <CheckCircle size={18} color="var(--accent-green)" />
            No anomalies detected for {zoneFilter === 'All' ? 'any zone' : zoneFilter}.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                {['Zone', 'Timestamp', 'Observed Load', 'Expected Load', 'Deviation', 'Score', 'Status'].map(h => (
                  <th key={h} style={{
                    padding: '8px 14px', textAlign: 'left', fontSize: 11, fontWeight: 600,
                    color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em',
                    borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)'
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(48,54,61,0.5)' }}>
                  <td style={{ padding: '10px 14px' }}>{a.zone}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-secondary)' }}>{a.timestamp}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--accent-orange)', fontWeight: 600 }}>
                    {a.observed_load_mw} MW
                  </td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-secondary)' }}>
                    {a.expected_load_mw ? `${a.expected_load_mw} MW` : '—'}
                  </td>
                  <td style={{ padding: '10px 14px', color: a.deviation?.startsWith('+') ? 'var(--accent-red)' : 'var(--accent-cyan)' }}>
                    {a.deviation ?? '—'}
                  </td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                    {a.anomaly_score?.toFixed(3)}
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <AlertTriangle size={13} color="var(--severity-critical)" />
                      <span className="badge badge-anomaly">{a.status}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card" style={{ background: 'rgba(248,81,73,0.04)', borderColor: 'rgba(248,81,73,0.15)', fontSize: 12, color: 'var(--text-secondary)' }}>
        <strong style={{ color: 'var(--text-primary)' }}>Note:</strong>{' '}
        Anomalies are labelled as "Possible abnormal consumption / possible non-technical loss".
        They are <em>not</em> automatically classified as electricity theft.
        Ground-truth theft data would be required to make that determination.
        Manual investigation is recommended.
      </div>
    </div>
  )
}
