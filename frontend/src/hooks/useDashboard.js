/**
 * useDashboard — main data hook
 * Fetches the full dashboard payload once and provides it to all pages.
 * Uses polling to refresh every 5 minutes.
 */

import { useState, useEffect, useCallback } from 'react'
import api from '../api/client'

const REFRESH_INTERVAL_MS = 5 * 60 * 1000  // 5 minutes

export function useDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchDashboard = useCallback(async () => {
    try {
      setError(null)
      const result = await api.getDashboard()
      setData(result.data)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  // Force-refresh: invalidate server cache then refetch
  const forceRefresh = useCallback(async () => {
    try {
      await api.invalidateCache()
    } catch (_) { /* ignore cache invalidation errors */ }
    setLoading(true)
    await fetchDashboard()
  }, [fetchDashboard])

  useEffect(() => {
    fetchDashboard()
    const interval = setInterval(fetchDashboard, REFRESH_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [fetchDashboard])

  return { data, loading, error, lastUpdated, refresh: fetchDashboard, forceRefresh }
}

export function useSimulation() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runSimulation = useCallback(async (params) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.runSimulation(params)
      setResult(res.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { result, loading, error, runSimulation }
}

export function useModelMetrics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getModelMetrics()
      .then(res => setData(res.data))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { data, loading, error }
}
