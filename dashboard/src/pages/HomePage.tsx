import PageHeader from '../components/PageHeader'

export default function HomePage() {
  return (
    <div className="page">
      <PageHeader title="Dashboard" />
      <div className="page-body">
        <p className="muted">Select a bot from the sidebar or go to Bots to view and manage your agents.</p>
      </div>
    </div>
  )
}
