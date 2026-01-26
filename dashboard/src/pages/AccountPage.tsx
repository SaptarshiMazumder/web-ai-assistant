import { useDashboardData } from '../hooks/useDashboardData'

export default function AccountPage() {
  const { user } = useDashboardData()

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">Account</div>
          <div className="page-subtitle">Profile and session details.</div>
        </div>
      </div>

      <section className="card">
        <div className="card-title">Profile</div>
        <div className="detail-row">
          <span>Email</span>
          <span>{user?.email || 'Not available'}</span>
        </div>
      </section>
    </div>
  )
}
