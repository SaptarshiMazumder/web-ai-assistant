import { Settings } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function SettingsPage() {
  const { refreshAll, loading } = useDashboardData()
  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <Settings className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Settings</span>
            </span>
          </div>
        </div>
        <div className="page-actions">
          <button className="ghost" onClick={refreshAll} disabled={loading}>
            Refresh
          </button>
        </div>
      </div>
      <div className="page-divider" />

      <div className="empty-panel">Settings controls will be added here.</div>
    </div>
  )
}
