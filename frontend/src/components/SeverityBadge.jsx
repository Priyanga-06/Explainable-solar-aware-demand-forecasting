import React from 'react'

const BADGE_CLASS = {
  Normal:   'badge-normal',
  Medium:   'badge-medium',
  High:     'badge-high',
  Critical: 'badge-critical',
}

export default function SeverityBadge({ level }) {
  if (!level) return null
  return (
    <span className={`badge ${BADGE_CLASS[level] || 'badge-medium'}`}>
      {level}
    </span>
  )
}
