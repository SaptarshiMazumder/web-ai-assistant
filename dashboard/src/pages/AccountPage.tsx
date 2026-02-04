import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'

export default function AccountPage() {
  const { user, logout } = useDashboardData()

  return (
    <div className="page narrow">
      <PageHeader title="Account" />
      <div className="page-body page-body-narrow">
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
    </div>
  )
}
