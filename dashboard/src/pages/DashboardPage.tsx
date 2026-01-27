import { LayoutDashboard } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function DashboardPage() {
  const { bots, orgs, activeOrgId, isSuperAdmin, refreshAll, loading } = useDashboardData()
  const botCount = bots.length
  const orgCount = orgs.length

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <LayoutDashboard className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Dashboard</span>
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
        <div className="card-grid">
          <section className="card">
            <div className="card-title">Active org</div>
            <div className="detail-row">
              <span>Organization</span>
              <span>{activeOrgId || 'Not selected'}</span>
            </div>
          </section>
          <section className="card">
            <div className="card-title">Bots</div>
            <div className="detail-row">
              <span>Total bots</span>
              <span>{botCount}</span>
            </div>
          </section>
          <section className="card">
            <div className="card-title">Organizations</div>
            <div className="detail-row">
              <span>Visible orgs</span>
              <span>{isSuperAdmin ? orgCount : orgs.length || 1}</span>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
