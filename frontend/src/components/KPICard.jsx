import React from 'react'

const SEVERITY_CLASS = {
  Normal: 'text-green',
  Medium: 'text-yellow',
  High: 'text-orange',
  Critical: 'text-red',
}

/**
 * KPICard — Top-level metric card for the dashboard header row
 */
export default function KPICard({ title, value, unit, icon: Icon, colour, badge, subtitle }) {
  return (
    <div className="card kpi-card" style={{ borderTop: `2px solid ${colour || 'var(--border)'}` }}>
      <div className="kpi-card__header">
        <span className="kpi-card__title">{title}</span>
        {Icon && <Icon size={16} color={colour || 'var(--text-muted)'} />}
      </div>
      <div className="kpi-card__value" style={{ color: colour || 'var(--text-primary)' }}>
        {value ?? <span className="text-muted">—</span>}
        {unit && <span className="kpi-card__unit"> {unit}</span>}
      </div>
      {badge && (
        <div style={{ marginTop: 6 }}>
          <span className={`badge badge-${badge.toLowerCase()}`}>{badge}</span>
        </div>
      )}
      {subtitle && <div className="kpi-card__subtitle">{subtitle}</div>}
    </div>
  )
}
