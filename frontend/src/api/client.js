/**
 * API Client
 * Centralised axios configuration for all backend requests.
 * All values come from actual backend outputs — no fake/hardcoded data.
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Response interceptor for consistent error handling
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred'
    return Promise.reject(new Error(message))
  }
)

// ── API endpoints ─────────────────────────────────────────────────────────────

export const api = {
  // Full dashboard payload
  getDashboard: () => apiClient.get('/api/v1/dashboard'),

  // Forecasts
  getDemandForecast: (zone = null) =>
    apiClient.get('/api/v1/forecast/demand', { params: zone ? { zone } : {} }),
  getZoneSummary:   () => apiClient.get('/api/v1/forecast/zones'),
  getSolarForecast: () => apiClient.get('/api/v1/forecast/solar'),
  getNetDemand:     () => apiClient.get('/api/v1/net-demand'),

  // Anomalies
  getAnomalies: (zone = null) =>
    apiClient.get('/api/v1/anomalies', { params: zone ? { zone } : {} }),

  // Explainability
  getSHAPExplanation: (zone) =>
    apiClient.post('/api/v1/explain/shap', { zone }),

  // Severity & recommendations
  getSeverity:        () => apiClient.get('/api/v1/severity'),
  getRecommendations: () => apiClient.get('/api/v1/recommendations'),

  // Model comparison
  getModelMetrics: () => apiClient.get('/api/v1/models/metrics'),

  // What-If simulator
  runSimulation: (params) =>
    apiClient.post('/api/v1/simulation/what-if', params),
  getSimulationHistory: (limit = 20) =>
    apiClient.get('/api/v1/simulation/history', { params: { limit } }),
  getSimulatorLimits: () => apiClient.get('/api/v1/simulation/limits'),

  // Reports
  generateReport: (includeSimulation = false) =>
    apiClient.post('/api/v1/reports/generate', null, {
      params: { include_simulation: includeSimulation },
    }),

  // Cache invalidation (force server to recompute dashboard payload)
  invalidateCache: () => apiClient.post('/api/v1/cache/invalidate'),

  // Health
  health: () => apiClient.get('/health'),
}

export default api
