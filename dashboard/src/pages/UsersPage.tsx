import { Users } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function UsersPage() {
  const { refreshAll, loading } = useDashboardData()
  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <Users className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Users</span>
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

      <div className="empty-panel">Add user-level controls here as you expand access policies.</div>
    </div>
  )
}
