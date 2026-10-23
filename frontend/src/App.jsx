import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import ZoneForecastPage from './pages/ZoneForecastPage'
import SolarIntegrationPage from './pages/SolarIntegrationPage'
import AnomalyMonitoringPage from './pages/AnomalyMonitoringPage'
import ForecastAnalyticsPage from './pages/ForecastAnalyticsPage'
import RecommendationsPage from './pages/RecommendationsPage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import WhatIfSimulatorPage from './pages/WhatIfSimulatorPage'
import ReportsPage from './pages/ReportsPage'
import SettingsPage from './pages/SettingsPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"              element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard"     element={<DashboardPage />} />
        <Route path="/zone-forecast" element={<ZoneForecastPage />} />
        <Route path="/solar"         element={<SolarIntegrationPage />} />
        <Route path="/anomalies"     element={<AnomalyMonitoringPage />} />
        <Route path="/analytics"     element={<ForecastAnalyticsPage />} />
        <Route path="/recommendations" element={<RecommendationsPage />} />
        <Route path="/model-comparison" element={<ModelComparisonPage />} />
        <Route path="/simulator"     element={<WhatIfSimulatorPage />} />
        <Route path="/reports"       element={<ReportsPage />} />
        <Route path="/settings"      element={<SettingsPage />} />
        <Route path="*"              element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Layout>
  )
}
