import PageHeader from '../components/PageHeader'

export default function SettingsPage() {
  return (
    <div className="page">
      <PageHeader title="Settings" />
      <div className="page-body page-body-narrow">
        <section className="card">
          <div className="card-title">Settings</div>
          <p className="muted">Manage your account preferences, notifications, and application settings here.</p>
          <p className="muted" style={{ marginTop: '0.5rem' }}>
            Additional settings controls will be added in future updates.
          </p>
        </section>
      </div>
    </div>
  )
}
