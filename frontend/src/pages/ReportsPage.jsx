import React, { useState } from 'react'
import { FileText, Download, AlertCircle, FlaskConical } from 'lucide-react'
import api from '../api/client'
import SeverityBadge from '../components/SeverityBadge'

export default function ReportsPage() {
  const [loading, setLoading]             = useState(false)
  const [report, setReport]               = useState(null)
  const [error, setError]                 = useState(null)
  const [includeSimulation, setInclude]   = useState(false)

  const handleGenerate = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.generateReport(includeSimulation)
      setReport(res.data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = () => {
    if (!report) return
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `power-grid-report-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const sec = report?.sections || {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Decision Support Report</div>
          <div className="page-subtitle">
            Generate and download officer reports — all values from actual model outputs
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, cursor: 'pointer', width: 'fit-content' }}>
          <input
            type="checkbox"
            checked={includeSimulation}
            onChange={e => setInclude(e.target.checked)}
            style={{ accentColor: 'var(--accent-purple)', width: 16, height: 16 }}
          />
          <FlaskConical size={14} color="var(--accent-purple)" />
          Include most recent What-If simulation results
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>(clearly labelled as Scenario Simulation)</span>
        </label>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <button
            className="btn-primary"
            onClick={handleGenerate}
            disabled={loading}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {loading
              ? <><span className="loading-spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Generating…</>
              : <><FileText size={14} /> Generate Report</>
            }
          </button>
          {report && (
            <button
              className="btn-secondary"
              onClick={handleDownload}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <Download size={14} /> Download JSON
            </button>
          )}
        </div>

        {error && <div className="error-banner">{error}</div>}
      </div>

      {/* Report content */}
      {report && (
        <>
          {/* Data note */}
          <div className="card" style={{ background: 'rgba(88,166,255,0.06)', borderColor: 'rgba(88,166,255,0.2)' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
              <AlertCircle size={14} color="var(--accent-blue)" />
              <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--accent-blue)' }}>Data Notice</span>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{report.data_note}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              Generated: {new Date(report.generated_at).toLocaleString()}
            </div>
          </div>

          {/* KPI summary */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 12 }}>Current Conditions</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10 }}>
              {Object.entries(sec.kpi_summary || {}).map(([k, v]) => (
                <div key={k} style={{ background: 'var(--bg-secondary)', padding: '10px 14px', borderRadius: 6 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {k.replace(/_/g, ' ')}
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                    {typeof v === 'number' ? v.toFixed(1) : String(v ?? '—')}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Zone forecasts */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 12 }}>Zone-wise Peak Forecast</div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(sec.zone_forecasts || {}).map(([zone, d]) => (
                <div key={zone} style={{ background: 'var(--bg-secondary)', padding: '10px 16px', borderRadius: 6, minWidth: 130 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{zone}</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--accent-blue)' }}>{d.peak_mw?.toFixed(0)} MW</div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>at H+{d.peak_hour ?? '—'}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Peak severity */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 10 }}>Peak Severity</div>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
              <SeverityBadge level={sec.peak_severity?.level} />
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Peak net demand</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)' }}>{sec.peak_severity?.peak_net_mw} MW</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>At</div>
                <div style={{ fontSize: 13 }}>{sec.peak_severity?.peak_time}</div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 400 }}>
                {sec.peak_severity?.description}
              </div>
            </div>
          </div>

          {/* Solar-load match */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Solar-Load Matching</div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <strong style={{ color: sec.solar_load_match?.match_pct >= 50 ? 'var(--accent-green)' : 'var(--accent-yellow)', fontSize: 24, marginRight: 8 }}>
                {sec.solar_load_match?.match_pct}%
              </strong>
              {sec.solar_load_match?.label} — {sec.solar_load_match?.description}
            </div>
          </div>

          {/* SHAP explanation */}
          {sec.shap_explanation?.peak_cause_explanation && (
            <div className="card" style={{ borderLeft: '3px solid var(--accent-purple)' }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>Peak Cause Explanation (SHAP)</div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                {sec.shap_explanation.peak_cause_explanation}
              </div>
            </div>
          )}

          {/* Anomalies */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 10 }}>
              Detected Anomalies ({(sec.anomalies || []).length})
            </div>
            {(sec.anomalies || []).length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No anomalies detected.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {(sec.anomalies || []).slice(0, 8).map((a, i) => (
                  <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '6px 10px', background: 'var(--bg-secondary)', borderRadius: 5 }}>
                    <span style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>[{a.zone}]</span>
                    {' '}{a.timestamp}{' '}·{' '}{a.observed_load_mw} MW{' '}·{' '}
                    <span style={{ color: a.deviation?.startsWith('+') ? 'var(--accent-red)' : 'var(--accent-cyan)' }}>{a.deviation}</span>
                    {' '}· {a.status}
                  </div>
                ))}
                {(sec.anomalies || []).length > 8 && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    +{sec.anomalies.length - 8} more — download JSON for full list
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Recommendations */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 10 }}>
              Recommendations ({(sec.recommendations || []).length})
            </div>
            {(sec.recommendations || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 10, paddingLeft: 10, borderLeft: '2px solid var(--border)' }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                  P{r.priority} [{r.category}] {r.action}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>Why: {r.reason}</div>
              </div>
            ))}
          </div>

          {/* Model comparison */}
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: 10 }}>Model Comparison (Test Set Metrics)</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  {['Zone', 'Model', 'MAE (MW)', 'RMSE (MW)', 'MAPE (%)', 'R²'].map(h => (
                    <th key={h} style={{ padding: '6px 10px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(sec.model_comparison || []).map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(48,54,61,0.4)' }}>
                    <td style={{ padding: '8px 10px' }}>{row.zone}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <span className="badge" style={{ background: row.model === 'GRU' ? 'rgba(88,166,255,0.15)' : 'rgba(219,109,40,0.15)', color: row.model === 'GRU' ? 'var(--accent-blue)' : 'var(--accent-orange)' }}>{row.model}</span>
                    </td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{row.MAE?.toFixed(3) ?? '—'}</td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{row.RMSE?.toFixed(3) ?? '—'}</td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{row.MAPE?.toFixed(2) ?? '—'}</td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', color: row.R2 > 0.8 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{row.R2?.toFixed(4) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Simulation section */}
          {sec.simulation && (
            <div className="card" style={{ border: '1px solid rgba(188,140,255,0.35)' }}>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
                <FlaskConical size={16} color="var(--accent-purple)" />
                <span className="simulated-label">Scenario Simulation</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>— NOT actual or predicted values</span>
              </div>

              {sec.simulation.note && !sec.simulation.baseline_peak_mw ? (
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{sec.simulation.note}</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Baseline Peak Net</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-blue)' }}>{sec.simulation.baseline_peak_mw} MW</div>
                      <SeverityBadge level={sec.simulation.baseline_severity} />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Simulated Peak Net</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-purple)' }}>{sec.simulation.simulated_peak_mw} MW</div>
                      <SeverityBadge level={sec.simulation.simulated_severity} />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Peak Reduction</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-green)' }}>
                        {sec.simulation.peak_reduction_mw} MW ({sec.simulation.peak_reduction_pct}%)
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    <strong>Scenario inputs:</strong>{' '}
                    {Object.entries(sec.simulation.scenario_inputs || {}).map(([k, v]) =>
                      `${k.replace(/_/g, ' ')}: ${v}%`
                    ).join(', ')}
                  </div>
                  {sec.simulation.recommended_scenario && (
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '8px 12px', background: 'rgba(88,166,255,0.07)', borderRadius: 5 }}>
                      <strong>Recommendation:</strong> {sec.simulation.recommended_scenario}
                    </div>
                  )}
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Run at: {sec.simulation.timestamp}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
