import React from 'react'
import { Settings } from 'lucide-react'

export default function SettingsPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header">
        <div>
          <div className="page-title">Settings</div>
          <div className="page-subtitle">System configuration and threshold management</div>
        </div>
      </div>

      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 10 }}>Severity Thresholds</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          Configured in: <code style={{ background: 'rgba(88,166,255,0.1)', padding: '1px 6px', borderRadius: 4 }}>config/settings.py → SEVERITY_THRESHOLDS</code>
          <br />Modify that file to change threshold values — no code changes needed elsewhere.
        </div>
        <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[
            { level: 'Normal',   range: '0 – 400 MW',    colour: 'var(--severity-normal)' },
            { level: 'Medium',   range: '400 – 600 MW',  colour: 'var(--severity-medium)' },
            { level: 'High',     range: '600 – 800 MW',  colour: 'var(--severity-high)' },
            { level: 'Critical', range: '> 800 MW',      colour: 'var(--severity-critical)' },
          ].map(t => (
            <div key={t.level} style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <span className="badge" style={{ background: t.colour + '22', color: t.colour, width: 70, justifyContent: 'center' }}>{t.level}</span>
              <span style={{ fontSize: 12, fontFamily: 'monospace' }}>{t.range}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 10 }}>Simulator Limits</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 10 }}>
          Configured in: <code style={{ background: 'rgba(88,166,255,0.1)', padding: '1px 6px', borderRadius: 4 }}>config/settings.py → SIMULATOR_LIMITS</code>
        </div>
        {[
          { key: 'industrial_shift_pct',      label: 'Industrial Load Shift',       max: '30%' },
          { key: 'commercial_shift_pct',      label: 'Commercial Load Shift',        max: '20%' },
          { key: 'residential_reduction_pct', label: 'Residential Demand Reduction', max: '15%' },
          { key: 'solar_increase_pct',        label: 'Solar Utilisation Increase',   max: '20%' },
        ].map(s => (
          <div key={s.key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 8, borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
            <span>{s.label}</span>
            <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>0 – {s.max}</span>
          </div>
        ))}
      </div>

      <div className="card" style={{ background: 'rgba(248,81,73,0.04)', borderColor: 'rgba(248,81,73,0.15)' }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Academic Data Notice</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          This system uses <strong>synthetic/demo data</strong> generated to be structurally
          representative of Indian electricity distribution networks. It is <strong>not</strong> real
          Tata Power operational data. The system is described as:{' '}
          <em>"Designed for Indian power distribution contexts and demonstrated using
          available distribution-network-relevant data."</em>
        </div>
      </div>
    </div>
  )
}
