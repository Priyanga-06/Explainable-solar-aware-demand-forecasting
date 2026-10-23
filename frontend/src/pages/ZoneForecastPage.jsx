import React, { useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useDashboard } from '../hooks/useDashboard'
import SeverityBadge from '../components/SeverityBadge'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'

const ZONE_COLOURS = {
  Residential: '#58a6ff',
  Commercial:  '#39c5cf',
  Industrial:  '#db6d28',
}

export default function ZoneForecastPage() {
  const { data, loading, error, refresh } = useDashboard()
  const [selectedZone, setSelectedZone] = useState('Residential')

  if (loading) return <LoadingState message="Loading zone forecasts..." />
  if (error?.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} onRetry={refresh} />
  if (!data) return null

  const { zone_forecasts, forecast_timestamps, peak_severity } = data
  const zoneData = zone_forecasts?.[selectedZone]
  const colour = ZONE_COLOURS[selectedZone]

  const chartData = (forecast_timestamps || []).map((ts, i) => ({
    time: ts.slice(11, 16),
    demand: zoneData?.predicted_demand_mw?.[i] ?? 0,
    lower:  zoneData?.confidence_lower_mw?.[i] ?? 0,
    upper:  zoneData?.confidence_upper_mw?.[i] ?? 0,
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Zone Forecast</div>
          <div className="page-subtitle">24-hour zone-wise electricity demand forecast (GRU model)</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['Residential', 'Commercial', 'Industrial'].map(z => (
            <button
              key={z}
              onClick={() => setSelectedZone(z)}
              style={{
                background: selectedZone === z ? ZONE_COLOURS[z] + '22' : 'var(--bg-card)',
                border: `1px solid ${selectedZone === z ? ZONE_COLOURS[z] : 'var(--border)'}`,
                color: selectedZone === z ? ZONE_COLOURS[z] : 'var(--text-secondary)',
                padding: '6px 14px', borderRadius: 6, fontWeight: selectedZone === z ? 600 : 400,
              }}
            >
              {z}
            </button>
          ))}
        </div>
      </div>

      {/* Zone summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        {Object.entries(zone_forecasts || {}).map(([zone, fc]) => (
          <div
            key={zone}
            className="card"
            style={{
              cursor: 'pointer',
              borderTop: `2px solid ${ZONE_COLOURS[zone]}`,
              opacity: selectedZone === zone ? 1 : 0.7,
            }}
            onClick={() => setSelectedZone(zone)}
          >
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{zone}</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: ZONE_COLOURS[zone], marginTop: 4 }}>
              {fc.peak_mw?.toFixed(0)} MW
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
              Peak at H+{fc.peak_hour_offset ?? '—'}
            </div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 14, alignItems: 'center' }}>
          <span style={{ fontWeight: 600 }}>{selectedZone} — 24-Hour Demand Forecast</span>
          <span className="badge badge-blue" style={{ fontSize: 10 }}>Predicted (GRU)</span>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <defs>
              <linearGradient id="zoneGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colour} stopOpacity={0.25} />
                <stop offset="95%" stopColor={colour} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="time" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} interval={3} />
            <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Area type="monotone" dataKey="upper"  stroke="none" fill={colour} fillOpacity={0.08} name="Upper Estimate" dot={false} />
            <Area type="monotone" dataKey="demand" stroke={colour} fill="url(#zoneGrad)" name={`${selectedZone} Demand`} dot={false} strokeWidth={2} />
            <Area type="monotone" dataKey="lower"  stroke="none" fill={colour} fillOpacity={0.04} name="Lower Estimate" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
          ⓘ Shaded band is the Estimated Range (±7.5%). Not a statistically derived confidence interval.
        </div>
      </div>
    </div>
  )
}
