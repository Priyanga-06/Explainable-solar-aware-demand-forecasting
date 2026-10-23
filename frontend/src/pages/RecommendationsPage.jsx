import React from 'react'
import { Lightbulb, AlertTriangle, Sun, Wrench, Zap, Shield } from 'lucide-react'
import { useDashboard } from '../hooks/useDashboard'
import { LoadingState, ErrorState, NotTrainedState } from '../components/LoadingState'

const CATEGORY_ICONS = {
  Emergency:        { icon: AlertTriangle, colour: 'var(--severity-critical)' },
  'Demand Response':{ icon: Zap,           colour: 'var(--accent-orange)' },
  'Solar Optimisation': { icon: Sun,       colour: 'var(--accent-yellow)' },
  'Supply Preparation': { icon: Shield,    colour: 'var(--accent-blue)' },
  Monitoring:       { icon: Lightbulb,     colour: 'var(--accent-cyan)' },
  Maintenance:      { icon: Wrench,        colour: 'var(--accent-green)' },
  'Anomaly Response': { icon: AlertTriangle, colour: 'var(--accent-red)' },
}

export default function RecommendationsPage() {
  const { data, loading, error, refresh } = useDashboard()

  if (loading) return <LoadingState message="Loading recommendations..." />
  if (error?.includes('not found')) return <NotTrainedState />
  if (error) return <ErrorState error={error} onRetry={refresh} />
  if (!data) return null

  const recommendations = data.recommendations || []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Recommendations</div>
          <div className="page-subtitle">
            Rule-based adaptive recommendations with explanations — sorted by priority
          </div>
        </div>
        <span className="text-muted" style={{ fontSize: 12 }}>{recommendations.length} active</span>
      </div>

      {recommendations.length === 0 ? (
        <div className="card" style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>
          No recommendations at this time.
        </div>
      ) : (
        recommendations.map((rec, i) => {
          const meta = CATEGORY_ICONS[rec.category] || { icon: Lightbulb, colour: 'var(--accent-blue)' }
          const IconComp = meta.icon
          return (
            <div
              key={i}
              className="card"
              style={{ borderLeft: `3px solid ${meta.colour}` }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: '50%',
                  background: meta.colour + '22',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0,
                }}>
                  <IconComp size={15} color={meta.colour} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }}>
                    <span style={{
                      fontSize: 10, fontWeight: 700, color: 'var(--text-muted)',
                      textTransform: 'uppercase', letterSpacing: '0.05em'
                    }}>
                      Priority {rec.priority}
                    </span>
                    <span className="badge" style={{
                      fontSize: 10,
                      background: meta.colour + '22',
                      color: meta.colour,
                    }}>{rec.category}</span>
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
                    {rec.action}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    <strong style={{ color: 'var(--text-muted)', fontSize: 11 }}>Why: </strong>
                    {rec.reason}
                  </div>
                </div>
              </div>
            </div>
          )
        })
      )}
    </div>
  )
}
