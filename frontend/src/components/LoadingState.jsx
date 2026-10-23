import React from 'react'

export function LoadingState({ message = 'Loading...' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '32px 0' }}>
      <span className="loading-spinner" />
      <span className="text-secondary">{message}</span>
    </div>
  )
}

export function ErrorState({ error, onRetry }) {
  return (
    <div className="error-banner" style={{ marginTop: 16 }}>
      <strong>Error:</strong> {error}
      {onRetry && (
        <button
          className="btn-secondary"
          onClick={onRetry}
          style={{ marginLeft: 16, padding: '4px 10px', fontSize: 12 }}
        >
          Retry
        </button>
      )}
    </div>
  )
}

export function NotTrainedState() {
  return (
    <div className="error-banner" style={{ background: 'rgba(88,166,255,0.08)', borderColor: 'rgba(88,166,255,0.3)', color: 'var(--accent-blue)' }}>
      <strong>Models not yet trained.</strong>{' '}
      Run <code style={{ background: 'rgba(88,166,255,0.15)', padding: '1px 6px', borderRadius: 4 }}>
        python backend/training/train_all.py
      </code>{' '}
      to train all models before using the dashboard.
    </div>
  )
}
