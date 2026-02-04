import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'

export default function OrgMembersPage() {
  const {
    activeOrgId,
    orgMembers,
    newMemberEmail,
    setNewMemberEmail,
    newMemberRole,
    setNewMemberRole,
    addOrgMember,
    loading,
  } = useDashboardData()

  if (!activeOrgId) {
    return <div className="empty-panel">Select an organization to manage users.</div>
  }

  return (
    <div className="page">
      <PageHeader title="Users" subtitle="Manage organization members and access." />

      <section className="card">
        <div className="card-title">Add member</div>
        <div className="stack">
          <input value={newMemberEmail} onChange={(event) => setNewMemberEmail(event.target.value)} placeholder="user@company.com" />
          <select value={newMemberRole} onChange={(event) => setNewMemberRole(event.target.value)}>
            <option value="org_admin">org_admin</option>
            <option value="org_member">org_member</option>
          </select>
          <button className="secondary" onClick={addOrgMember} disabled={loading || !newMemberEmail.trim()}>
            Add member
          </button>
        </div>
      </section>

      <section className="card">
        <div className="card-title">Members</div>
        <div className="list">
          {orgMembers.map((member) => (
            <div key={member.user_id} className="list-row">
              <div>
                <div className="list-title">{member.email}</div>
                <div className="muted">{member.role}</div>
              </div>
            </div>
          ))}
          {!orgMembers.length && <div className="empty">No members yet.</div>}
        </div>
      </section>
    </div>
  )
}
