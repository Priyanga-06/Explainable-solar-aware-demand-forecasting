import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, LineChart, Line
} from 'recharts'
import { FlaskConical, Play, Info } from 'lucide-react'
import { useSimulation, useDashboard } from '../hooks/useDashboard'
import SeverityBadge from '../components/SeverityBadge'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'
import './WhatIfSimulatorPage.css'

const DEFAULT_PARAMS = {
  industrial_shift_pct:      10,
  commercial_shift_pct:      5,
  residential_reduction_pct: 3,
  solar_increase_pct:        5,
}

const LIMITS = {
  industrial_shift_pct:      { min: 0, max: 30, label: 'Industrial Load Shift',       unit: '%' },
  commercial_shift_pct:      { min: 0, max: 20, label: 'Commercial Load Shift',        unit: '%' },
  residential_reduction_pct: { min: 0, max: 15, label: 'Residential Demand Reduction', unit: '%' },
  solar_increase_pct:        { min: 0, max: 20, label: 'Solar Utilisation Increase',   unit: '%' },
}

export default function WhatIfSimulatorPage() {
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const { result, loading, error, runSimulation } = useSimulation()
  const { data: dashboardData, loading: dashLoading } = useDashboard()

  const handleSlider = (key, value) => {
    setParams(p => ({ ...p, [key]: Number(value) }))
  }

  const handleRun = () => runSimulation(params)

  if (dashLoading) return <LoadingState message="Loading baseline forecast..." />
  if (!dashboardData && !dashLoading) return <NotTrainedState />

  // Comparison table data for chart
  const comparisonData = result ? [
    {
      metric: 'Peak Demand',
      Baseline: result.baseline.peak_demand_mw,
      Simulated: result.simulated.peak_demand_mw,
    },
    {
      metric: 'Solar',
      Baseline: result.baseline.peak_solar_mw,
      Simulated: result.simulated.peak_solar_mw,
    },
    {
      metric: 'Net Demand',
      Baseline: result.baseline.peak_net_demand_mw,
      Simulated: result.simulated.peak_net_demand_mw,
    },
  ] : []

  // 24-hour comparison chart
  const hourlyData = result
    ? (result.baseline.demand_24h || []).map((_, i) => ({
        hour: `H+${i}`,
        'Baseline Net':  result.baseline.net_24h?.[i] ?? 0,
        'Simulated Net': result.simulated.net_24h?.[i] ?? 0,
      }))
    : []

  return (
    <div className="simulator-page">
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FlaskConical size={22} color="var(--accent-purple)" />
            What-If Peak Mitigation Simulator
          </div>
          <div className="page-subtitle">
            Test hypothetical peak-reduction scenarios before taking operational action.
          </div>
        </div>
        <span className="simulated-label">Scenario Simulation — Not an ML Prediction</span>
      </div>

      {/* Disclaimer */}
      <div className="simulator-disclaimer">
        <Info size={14} style={{ flexShrink: 0, marginTop: 2 }} />
        <p>
          This simulator applies proportional adjustments to the <strong>Baseline Forecast</strong>.
          It does <strong>not</strong> re-run any machine learning model. All simulated results are
          clearly separated from actual forecasts and are labelled <em>"Simulated What-If Scenario"</em>.
        </p>
      </div>

      <div className="simulator-layout">
        {/* Controls */}
        <div className="card simulator-controls">
          <div className="simulator-controls__title">Scenario Parameters</div>
          <div className="divider" />

          {Object.entries(LIMITS).map(([key, meta]) => (
            <div key={key} className="slider-group">
              <div className="slider-group__header">
                <span>{meta.label}</span>
                <span className="slider-group__value">{params[key]}{meta.unit}</span>
              </div>
              <input
                type="range"
                min={meta.min}
                max={meta.max}
                step="1"
                value={params[key]}
                onChange={e => handleSlider(key, e.target.value)}
              />
              <div className="slider-group__range">
                <span>{meta.min}%</span>
                <span>{meta.max}%</span>
              </div>
            </div>
          ))}

          <button
            className="btn-simulate"
            onClick={handleRun}
            disabled={loading}
          >
            {loading ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
                <span className="loading-spinner" style={{ width: 16, height: 16 }} />
                Running Simulation...
              </span>
            ) : (
              <span style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
                <Play size={16} />
                RUN SIMULATION
              </span>
            )}
          </button>

          {error && <div className="error-banner" style={{ marginTop: 12 }}>{error}</div>}
        </div>

        {/* Results */}
        <div className="simulator-results">
          {!result && !loading && (
            <div className="simulator-placeholder">
              <FlaskConical size={48} color="var(--border)" />
              <div style={{ marginTop: 12, color: 'var(--text-muted)' }}>
                Adjust parameters and click <strong>RUN SIMULATION</strong> to see results.
              </div>
            </div>
          )}

          {result && (
            <>
              {/* Comparison Table */}
              <div className="card">
                <div className="sim-result-header">
                  <span className="chart-card__title">Simulation Results</span>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <span className="badge badge-blue">Baseline Forecast</span>
                    <span className="simulated-label">Simulated What-If Scenario</span>
                  </div>
                </div>

                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Baseline Forecast</th>
                      <th>Simulated Scenario</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Peak Demand</td>
                      <td className="text-blue">{result.baseline.peak_demand_mw} MW</td>
                      <td className="text-purple">{result.simulated.peak_demand_mw} MW</td>
                    </tr>
                    <tr>
                      <td>Solar Generation</td>
                      <td className="text-yellow">{result.baseline.peak_solar_mw} MW</td>
                      <td className="text-yellow">{result.simulated.peak_solar_mw} MW</td>
                    </tr>
                    <tr>
                      <td>Net Grid Demand</td>
                      <td className="text-orange">{result.baseline.peak_net_demand_mw} MW</td>
                      <td className="text-orange">{result.simulated.peak_net_demand_mw} MW</td>
                    </tr>
                    <tr>
                      <td>Peak Time</td>
                      <td>{result.baseline.peak_timestamp}</td>
                      <td>{result.simulated.peak_timestamp}</td>
                    </tr>
                    <tr>
                      <td>Severity</td>
                      <td><SeverityBadge level={result.baseline.severity} /></td>
                      <td><SeverityBadge level={result.simulated.severity} /></td>
                    </tr>
                  </tbody>
                </table>

                {/* Impact Summary */}
                <div className="impact-summary">
                  <div className="impact-item">
                    <div className="impact-item__label">Peak Reduction</div>
                    <div className="impact-item__value text-green">
                      {result.impact.peak_reduction_mw} MW
                    </div>
                  </div>
                  <div className="impact-item">
                    <div className="impact-item__label">Peak Reduction %</div>
                    <div className="impact-item__value text-green">
                      {result.impact.peak_reduction_pct}%
                    </div>
                  </div>
                  <div className="impact-item">
                    <div className="impact-item__label">Severity Change</div>
                    <div className="impact-item__value" style={{ fontSize: 13 }}>
                      {result.impact.severity_improvement}
                    </div>
                  </div>
                </div>

                {/* Recommendation */}
                <div className="sim-recommendation">
                  <strong>Recommended:</strong> {result.impact.recommended_scenario}
                </div>
              </div>

              {/* Before/After Chart — Bar */}
              <div className="card">
                <div className="chart-card__header">
                  <span className="chart-card__title">Before vs After — Key Metrics</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <span className="badge badge-blue" style={{ fontSize: 10 }}>Baseline</span>
                    <span className="simulated-label" style={{ fontSize: 10, padding: '2px 8px' }}>Simulated</span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={comparisonData} barCategoryGap="35%">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="metric" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                    <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
                    <Tooltip
                      contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Baseline"  fill="#58a6ff" name="Baseline Forecast" />
                    <Bar dataKey="Simulated" fill="#bc8cff" name="Simulated Scenario" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* 24-hour net demand comparison */}
              <div className="card">
                <div className="chart-card__header">
                  <span className="chart-card__title">24-Hour Net Demand: Baseline vs Simulated</span>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={hourlyData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="hour" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval={3} />
                    <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit=" MW" />
                    <Tooltip
                      contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="Baseline Net"  stroke="#58a6ff" dot={false} strokeWidth={2} name="Baseline Forecast" />
                    <Line type="monotone" dataKey="Simulated Net" stroke="#bc8cff" dot={false} strokeWidth={2} name="Simulated Scenario" strokeDasharray="5 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Disclaimer */}
              <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '0 2px' }}>
                ⚠ {result.disclaimer}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
