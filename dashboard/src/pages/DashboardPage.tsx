import { useDashboardData } from '../hooks/useDashboardData'

export default function DashboardPage() {
  const { bots, orgs, activeOrgId, isSuperAdmin } = useDashboardData()
  const botCount = bots.length
  const orgCount = orgs.length

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">Dashboard</div>
          <div className="page-subtitle">Overview of your workspace.</div>
        </div>
      </div>

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
  )
}
