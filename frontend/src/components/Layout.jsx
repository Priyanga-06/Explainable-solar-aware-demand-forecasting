import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, MapPin, Sun, AlertTriangle, TrendingUp,
  Lightbulb, BarChart3, FlaskConical, FileText, Settings, Menu, X
} from 'lucide-react'
import './Layout.css'

const NAV_ITEMS = [
  { to: '/dashboard',        label: 'Dashboard',         icon: LayoutDashboard },
  { to: '/zone-forecast',    label: 'Zone Forecast',      icon: MapPin },
  { to: '/solar',            label: 'Solar Integration',  icon: Sun },
  { to: '/anomalies',        label: 'Anomaly Monitoring', icon: AlertTriangle },
  { to: '/analytics',        label: 'Forecast & Analytics', icon: TrendingUp },
  { to: '/recommendations',  label: 'Recommendations',    icon: Lightbulb },
  { to: '/model-comparison', label: 'Model Comparison',   icon: BarChart3 },
  { to: '/simulator',        label: 'What-If Simulator',  icon: FlaskConical, highlight: true },
  { to: '/reports',          label: 'Reports',            icon: FileText },
  { to: '/settings',         label: 'Settings',           icon: Settings },
]

export default function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className={`layout ${collapsed ? 'layout--collapsed' : ''}`}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar__header">
          <div className="sidebar__brand">
            {!collapsed && (
              <div>
                <div className="sidebar__brand-title">PowerAI</div>
                <div className="sidebar__brand-sub">Decision Support System</div>
              </div>
            )}
            <button
              className="sidebar__toggle"
              onClick={() => setCollapsed(c => !c)}
              aria-label="Toggle sidebar"
            >
              {collapsed ? <Menu size={18} /> : <X size={18} />}
            </button>
          </div>
        </div>

        <nav className="sidebar__nav">
          {NAV_ITEMS.map(({ to, label, icon: Icon, highlight }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `sidebar__link ${isActive ? 'sidebar__link--active' : ''} ${highlight ? 'sidebar__link--highlight' : ''}`
              }
              title={collapsed ? label : undefined}
            >
              <Icon size={16} />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        {!collapsed && (
          <div className="sidebar__footer">
            <div className="sidebar__footer-note">
              Data: Synthetic/Demo
            </div>
            <div className="sidebar__footer-note">
              Not real Tata Power data
            </div>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="main-content">
        <div className="main-content__inner">
          {children}
        </div>
      </main>
    </div>
  )
}
