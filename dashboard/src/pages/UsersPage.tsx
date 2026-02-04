import PageHeader from '../components/PageHeader'

export default function UsersPage() {
  return (
    <div className="page">
      <PageHeader title="Users" />
      <div className="page-body">
        <div className="empty-panel">Add user-level controls here as you expand access policies.</div>
      </div>
    </div>
  )
}
