import React from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useDashboard } from '../hooks/useDashboard'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'

export default function SolarIntegrationPage() {
  const { data, loading, error, refresh } = useDashboard()

  if (loading) return <LoadingState message="Loading solar forecast..." />
  if (error?.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} onRetry={refresh} />
  if (!data) return null

  const { solar_24h, total_demand_24h, net_demand_24h, forecast_timestamps, solar_load_match } = data

  const chartData = (forecast_timestamps || []).map((ts, i) => ({
    time: ts.slice(11, 16),
    demand: total_demand_24h?.[i] ?? 0,
    solar:  solar_24h?.[i] ?? 0,
    net:    net_demand_24h?.[i] ?? 0,
  }))

  const peakSolar = Math.max(...(solar_24h || [0]))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Solar Integration</div>
          <div className="page-subtitle">XGBoost solar forecast, net grid demand, and solar-load matching</div>
        </div>
        <span className="badge badge-blue" style={{ fontSize: 11 }}>XGBoost Solar Model</span>
      </div>

      {/* KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
        <div className="card" style={{ borderTop: '2px solid var(--accent-yellow)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Peak Solar</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--accent-yellow)', marginTop: 4 }}>
            {peakSolar.toFixed(0)} MW
          </div>
        </div>
        <div className="card" style={{ borderTop: '2px solid var(--accent-green)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Solar-Load Match</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: solar_load_match?.match_pct >= 50 ? 'var(--accent-green)' : 'var(--accent-yellow)', marginTop: 4 }}>
            {solar_load_match?.match_pct ?? '—'}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>{solar_load_match?.label}</div>
        </div>
        <div className="card" style={{ borderTop: '2px solid var(--accent-cyan)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Daytime Solar Total</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--accent-cyan)', marginTop: 4 }}>
            {solar_load_match?.daytime_solar_mw?.toFixed(0) ?? '—'} MW·h
          </div>
        </div>
        <div className="card" style={{ borderTop: '2px solid var(--accent-blue)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Daytime Demand</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--accent-blue)', marginTop: 4 }}>
            {solar_load_match?.daytime_demand_mw?.toFixed(0) ?? '—'} MW·h
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 14, alignItems: 'center' }}>
          <span style={{ fontWeight: 600 }}>Demand vs Solar vs Net Grid Demand</span>
          <span className="badge badge-blue" style={{ fontSize: 10 }}>Predicted</span>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="sG1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#58a6ff" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="sG2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#d29922" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#d29922" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="sG3" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#db6d28" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#db6d28" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="time" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} interval={3} />
            <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Area type="monotone" dataKey="demand" stroke="#58a6ff" fill="url(#sG1)" name="Total Demand" dot={false} />
            <Area type="monotone" dataKey="solar"  stroke="#d29922" fill="url(#sG2)" name="Solar Generation" dot={false} />
            <Area type="monotone" dataKey="net"    stroke="#db6d28" fill="url(#sG3)" name="Net Grid Demand" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Solar-load match details */}
      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 10 }}>Solar-Load Matching Formula</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 2 }}>
          <strong>{solar_load_match?.description}</strong>
          <br />
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {solar_load_match?.formula}
          </span>
        </div>
      </div>
    </div>
  )
}
