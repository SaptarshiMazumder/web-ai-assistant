import { Users, UserPlus, Mail } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../components/ui'

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
    <AnimatedPage className="page">
      <PageHeader title="Users" subtitle="Manage organization members and access." />

      <SectionHeader
        eyebrow="Team"
        title="Manage members"
        subtitle="Invite new team members and view your current roster."
      />

      <GlassCard>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <UserPlus size={16} style={{ color: 'var(--ui-flow-accent)' }} />
          Add member
        </div>
        <div className="stack">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Mail size={14} style={{ color: 'var(--ui-flow-muted)' }} />
            <span className="muted" style={{ fontSize: '0.85rem' }}>Email address</span>
          </div>
          <input value={newMemberEmail} onChange={(event) => setNewMemberEmail(event.target.value)} placeholder="user@company.com" />
          <select value={newMemberRole} onChange={(event) => setNewMemberRole(event.target.value)}>
            <option value="org_admin">Admin</option>
            <option value="org_member">Member</option>
          </select>
          <UiButton variant="primary" onClick={addOrgMember} disabled={loading || !newMemberEmail.trim()} style={{ alignSelf: 'flex-start' }}>
            Add member
          </UiButton>
        </div>
      </GlassCard>

      <GlassCard>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Users size={16} style={{ color: 'var(--ui-flow-accent)' }} />
          Members ({orgMembers.length})
        </div>
        <div className="list">
          {orgMembers.map((member) => (
            <div key={member.user_id} className="list-row">
              <div>
                <div className="list-title">{member.email}</div>
                <div className="muted">{member.role}</div>
              </div>
            </div>
          ))}
          {!orgMembers.length && <div className="muted" style={{ padding: '0.75rem 0' }}>No members yet.</div>}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
