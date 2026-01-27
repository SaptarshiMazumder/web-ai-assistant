import { UserCircle } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function AccountPage() {
  const { user, logout, refreshAll, loading } = useDashboardData()

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <UserCircle className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Account</span>
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

      <section className="card">
        <div className="card-title">Profile</div>
        <div className="detail-row">
          <span>Email</span>
          <span>{user?.email || 'Not available'}</span>
        </div>
        <div className="row">
          <button className="ghost" onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}>
            Sign out
          </button>
        </div>
      </section>
    </div>
  )
}
