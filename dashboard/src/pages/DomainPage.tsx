import { Globe } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function DomainPage() {
  const { refreshAll, loading } = useDashboardData()
  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <Globe className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Domain</span>
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

      <div className="page-body">
        <div className="empty-panel">Domain controls will live here.</div>
      </div>
    </div>
  )
}
