import React from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts'
import {
  Zap, Sun, Grid3x3, Activity, TrendingUp, AlertTriangle, ShieldCheck
} from 'lucide-react'
import { useDashboard } from '../hooks/useDashboard'
import KPICard from '../components/KPICard'
import SeverityBadge from '../components/SeverityBadge'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'
import './DashboardPage.css'

export default function DashboardPage() {
  const { data, loading, error, lastUpdated, refresh } = useDashboard()

  if (loading) return <LoadingState message="Loading dashboard data..." />
  if (error && error.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} onRetry={refresh} />
  if (!data) return null

  const { kpi, forecast_timestamps, total_demand_24h, solar_24h, net_demand_24h,
          peak_severity, forecast_confidence, solar_load_match, zone_forecasts } = data

  // Build chart data
  const chartData = (forecast_timestamps || []).map((ts, i) => ({
    time: ts.slice(11, 16),  // HH:MM
    demand: total_demand_24h?.[i] ?? 0,
    solar: solar_24h?.[i] ?? 0,
    net: net_demand_24h?.[i] ?? 0,
    lower: forecast_confidence ? total_demand_24h?.[i] * (1 - 0.075) : undefined,
    upper: forecast_confidence ? total_demand_24h?.[i] * (1 + 0.075) : undefined,
  }))

  const severityLevel = peak_severity?.severity?.level || 'Normal'
  const severityColours = {
    Normal: 'var(--severity-normal)',
    Medium: 'var(--severity-medium)',
    High: 'var(--severity-high)',
    Critical: 'var(--severity-critical)',
  }

  return (
    <div className="dashboard-page">
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Officer Dashboard</div>
          <div className="page-subtitle">
            Zone-wise demand forecasting & grid health monitoring
            {lastUpdated && ` · Updated ${lastUpdated.toLocaleTimeString()}`}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className="badge badge-blue" style={{ fontSize: 11 }}>Synthetic Data</span>
          <button className="btn-secondary" onClick={refresh} style={{ fontSize: 12 }}>
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Row */}
      <div className="kpi-grid">
        <KPICard
          title="Current Load"
          value={kpi?.current_load_mw?.toFixed(0)}
          unit="MW"
          icon={Zap}
          colour="var(--accent-blue)"
          subtitle="Total across all zones"
        />
        <KPICard
          title="Next-Hour Demand"
          value={kpi?.next_hour_demand_mw?.toFixed(0)}
          unit="MW"
          icon={TrendingUp}
          colour="var(--accent-cyan)"
          subtitle="Forecast — next hour"
        />
        <KPICard
          title="24-Hour Peak"
          value={kpi?.peak_24h_mw?.toFixed(0)}
          unit="MW"
          icon={Activity}
          colour={severityColours[severityLevel]}
          badge={severityLevel}
          subtitle={`Expected at ${peak_severity?.peak_timestamp?.slice(11, 16) || '—'}`}
        />
        <KPICard
          title="Solar Generation"
          value={kpi?.solar_generation_mw?.toFixed(0)}
          unit="MW"
          icon={Sun}
          colour="var(--accent-yellow)"
          subtitle="Current forecast"
        />
        <KPICard
          title="Net Grid Demand"
          value={kpi?.net_grid_demand_mw?.toFixed(0)}
          unit="MW"
          icon={Grid3x3}
          colour="var(--accent-orange)"
          subtitle="After solar offset"
        />
        <KPICard
          title="Demand Status"
          value={severityLevel}
          icon={ShieldCheck}
          colour={severityColours[severityLevel]}
          subtitle={peak_severity?.severity?.description?.slice(0, 55) + '…' || '—'}
        />
      </div>

      {/* Main charts */}
      <div className="dashboard-charts">
        {/* 24-Hour Load Forecast */}
        <div className="card chart-card">
          <div className="chart-card__header">
            <span className="chart-card__title">24-Hour Demand & Solar Forecast</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <span className="badge badge-blue" style={{ fontSize: 10 }}>Predicted</span>
              {forecast_confidence && (
                <span
                  title={forecast_confidence.note}
                  className="badge"
                  style={{ fontSize: 10, background: 'rgba(88,166,255,0.08)', color: 'var(--text-muted)', cursor: 'help' }}
                >
                  ±Estimated Range
                </span>
              )}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <defs>
                <linearGradient id="demandGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#58a6ff" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="solarGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#d29922" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#d29922" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="netGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#db6d28" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#db6d28" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="time" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} interval={3} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
              <Tooltip
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', fontSize: 12 }}
                labelStyle={{ color: 'var(--text-primary)' }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
              <Area type="monotone" dataKey="demand" stroke="#58a6ff" fill="url(#demandGrad)" name="Total Demand" dot={false} />
              <Area type="monotone" dataKey="solar"  stroke="#d29922" fill="url(#solarGrad)"  name="Solar Generation" dot={false} />
              <Area type="monotone" dataKey="net"    stroke="#db6d28" fill="url(#netGrad)"    name="Net Grid Demand" dot={false} />
              {peak_severity?.peak_hour_index !== undefined && (
                <ReferenceLine
                  x={chartData[peak_severity.peak_hour_index]?.time}
                  stroke="var(--severity-critical)"
                  strokeDasharray="4 2"
                  label={{ value: 'PEAK', fill: 'var(--severity-critical)', fontSize: 10 }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Zone-wise peak bar */}
        <div className="card chart-card">
          <div className="chart-card__header">
            <span className="chart-card__title">Zone-wise Peak Demand</span>
            <span className="badge badge-blue" style={{ fontSize: 10 }}>Predicted</span>
          </div>
          <div className="zone-bars">
            {Object.entries(zone_forecasts || {}).map(([zone, fc]) => {
              const peak = fc.peak_mw || 0
              const max = 500
              const pct = Math.min((peak / max) * 100, 100)
              const colour = zone === 'Industrial' ? 'var(--accent-orange)'
                           : zone === 'Commercial' ? 'var(--accent-cyan)'
                           : 'var(--accent-blue)'
              return (
                <div key={zone} className="zone-bar-row">
                  <span className="zone-bar-label">{zone}</span>
                  <div className="zone-bar-track">
                    <div className="zone-bar-fill" style={{ width: `${pct}%`, background: colour }} />
                  </div>
                  <span className="zone-bar-value" style={{ color: colour }}>{peak.toFixed(0)} MW</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Solar-Load Match */}
        <div className="card chart-card solar-match-card">
          <div className="chart-card__header">
            <span className="chart-card__title">Solar-Load Matching</span>
          </div>
          <div className="solar-match-display">
            <div
              className="solar-match-score"
              style={{ color: solar_load_match?.match_pct >= 50 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}
            >
              {solar_load_match?.match_pct ?? '—'}%
            </div>
            <div className="solar-match-label">{solar_load_match?.label || '—'}</div>
            <div className="solar-match-desc">{solar_load_match?.description || '—'}</div>
            <div className="divider" />
            <div className="solar-match-formula">
              <span className="text-muted" style={{ fontSize: 11 }}>Formula: </span>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                {solar_load_match?.formula || 'Solar-Load Match % = (Total Daytime Solar / Total Daytime Demand) × 100'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Forecast confidence + Severity */}
      <div className="dashboard-lower">
        {/* Confidence */}
        {forecast_confidence && (
          <div className="card">
            <div className="chart-card__header">
              <span className="chart-card__title">Forecast Confidence</span>
            </div>
            <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', marginTop: 12 }}>
              <div>
                <div className="text-muted" style={{ fontSize: 11 }}>Expected Peak</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--accent-blue)' }}>
                  {forecast_confidence.expected_peak_mw} MW
                </div>
              </div>
              <div>
                <div className="text-muted" style={{ fontSize: 11 }}>Estimated Range</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>
                  {forecast_confidence.estimated_lower_mw} – {forecast_confidence.estimated_upper_mw} MW
                </div>
              </div>
            </div>
            <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
              ⓘ {forecast_confidence.note}
            </div>
          </div>
        )}

        {/* Peak Severity Detail */}
        {peak_severity?.severity && (
          <div className="card" style={{ borderLeft: `3px solid ${severityColours[severityLevel]}` }}>
            <div className="chart-card__header">
              <span className="chart-card__title">Peak Severity</span>
              <SeverityBadge level={severityLevel} />
            </div>
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 26, fontWeight: 700, color: severityColours[severityLevel] }}>
                {peak_severity.peak_net_demand_mw} MW
              </div>
              <div className="text-secondary" style={{ fontSize: 12, marginTop: 4 }}>
                Expected at {peak_severity.peak_timestamp}
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
                {peak_severity.severity.description}
              </div>
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                Threshold: {peak_severity.severity.threshold_range}
                {' '}·{' '}
                <span>Configured in {peak_severity.severity.thresholds_source}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
