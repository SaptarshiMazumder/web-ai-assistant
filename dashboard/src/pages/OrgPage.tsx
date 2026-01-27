import { Building2 } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function OrgPage() {
  const {
    isSuperAdmin,
    activeOrgId,
    setActiveOrgId,
    orgs,
    bots,
    orgMembers,
    newOrgName,
    setNewOrgName,
    createOrg,
    setOrgStatus,
    newMemberEmail,
    setNewMemberEmail,
    newMemberRole,
    setNewMemberRole,
    addOrgMember,
    orgDisplayNameInput,
    setOrgDisplayNameInput,
    saveOrgName,
    loading,
    user,
    refreshAll,
  } = useDashboardData()

  const activeOrg = orgs.find((org) => org.org_id === activeOrgId)
  const currentOrgName = activeOrg?.name || orgDisplayNameInput || activeOrgId || 'No org selected'
  const botCountLabel = `${bots.length} Bots`
  const currentRole = orgMembers.find((member) => member.email?.toLowerCase() === user?.email?.toLowerCase())?.role || ''
  const isAllOrgsSelected = activeOrgId === '__all__'

  const currentUserName = (() => {
    const given = user?.given_name?.trim()
    const family = user?.family_name?.trim()
    const combined = `${given || ''} ${family || ''}`.trim()
    if (combined) return combined
    const name = user?.name?.trim()
    if (name) return name
    const email = user?.email?.trim()
    if (!email) return ''
    const local = email.split('@')[0]
    return local ? local.replace(/[._-]+/g, ' ').trim() : email
  })()

  const currentUserPicture = user?.picture?.trim() || ''

  const displayNameForMember = (email: string, firstName?: string | null, lastName?: string | null) => {
    const full = `${firstName || ''} ${lastName || ''}`.trim()
    if (full) return full
    const local = (email || '').split('@')[0]
    if (!local) return email || 'Member'
    const cleaned = local.replace(/[._-]+/g, ' ').trim()
    return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : email
  }

  const displayRole = (role: string) => {
    if (role === 'owner') return 'Owner'
    if (role === 'org_admin') return 'Admin'
    if (role === 'org_member') return 'User'
    return role
  }

  const avatarUrlForMember = (name: string, email: string) => {
    const label = name || email || 'User'
    return `https://ui-avatars.com/api/?name=${encodeURIComponent(label)}&background=0f172a&color=fff&size=128`
  }

  if (!activeOrgId && !isSuperAdmin) {
    return <div className="empty-panel">No organization assigned yet.</div>
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <Building2 className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Organization</span>
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

      <div className="org-grid">
        <section className="card">
          <div className="card-title">Current Team</div>
          <div className="card-subtitle">Switch between different team dashboards that you have access to.</div>
          <div className="org-select-row">
            <div className="org-select">
              {isSuperAdmin ? (
                <div className="org-select-name">
                  <select
                    className="org-select-input"
                    value={activeOrgId || ''}
                    onChange={(event) => setActiveOrgId(event.target.value)}
                  >
                    <option value="" disabled>
                      Select org
                    </option>
                    <option value="__all__">All orgs</option>
                    {orgs.map((org) => (
                      <option key={org.org_id} value={org.org_id}>
                        {org.name} ({org.org_id})
                      </option>
                    ))}
                  </select>
                  {currentRole && <span className="org-role">{currentRole}</span>}
                </div>
              ) : (
                <div className="org-select-name">
                  {currentOrgName}
                  {currentRole && <span className="org-role">{currentRole}</span>}
                </div>
              )}
              <div className="org-select-meta">{botCountLabel}</div>
            </div>
          </div>

          {isSuperAdmin && (
            <div className="org-admin-block">
              <div className="card-title">Create org</div>
              <div className="stack">
                <input value={newOrgName} onChange={(event) => setNewOrgName(event.target.value)} placeholder="Org name" />
                <button className="primary" onClick={createOrg} disabled={loading || !newOrgName.trim()}>
                  Create org
                </button>
              </div>
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-title">Rename Team</div>
          <div className="card-subtitle">
            {isAllOrgsSelected ? 'Select a single org to rename it.' : `Enter a new team name for ${currentOrgName}.`}
          </div>
          <div className="org-rename-row">
            <input
              value={orgDisplayNameInput}
              onChange={(event) => setOrgDisplayNameInput(event.target.value)}
              placeholder="Organization name"
              disabled={isAllOrgsSelected}
            />
            <button className="secondary" onClick={saveOrgName} disabled={isAllOrgsSelected || loading || !orgDisplayNameInput.trim()}>
              Update
            </button>
          </div>
        </section>
      </div>

      {activeOrgId && (
        <>
          <section className="card">
            <div className="members-header">
              <div>
                <div className="card-title">Members</div>
                <div className="card-subtitle">
                  {isAllOrgsSelected ? 'View members across all orgs.' : 'View and manage the members of this team.'}
                </div>
              </div>
            </div>

            <div className={`members-header-row members-columns ${isSuperAdmin ? 'members-columns-admin' : ''}`}>
              <div className="muted">Name</div>
              <div className="muted">Email</div>
              <div className="muted">Role</div>
              {isSuperAdmin && <div className="muted">Org</div>}
            </div>
            <div className="list">
              {orgMembers.map((member) => {
                const isCurrentUser = user?.email && member.email?.toLowerCase() === user.email.toLowerCase()
                const name =
                  isCurrentUser && currentUserName
                    ? currentUserName
                    : displayNameForMember(member.email, member.first_name, member.last_name)
                const initials = name
                  .split(' ')
                  .filter(Boolean)
                  .slice(0, 2)
                  .map((part) => part[0].toUpperCase())
                  .join('')
                const avatarUrl = isCurrentUser && currentUserPicture ? currentUserPicture : avatarUrlForMember(name, member.email)
                const orgLabel = member.org_name || member.org_id || currentOrgName
                return (
                  <div key={`${member.user_id}-${member.org_id || 'org'}`} className={`member-row members-columns ${isSuperAdmin ? 'members-columns-admin' : ''}`}>
                    <div className="member-info">
                      <div className="avatar">
                        <img
                          src={avatarUrl}
                          alt={name}
                          onLoad={(event) => event.currentTarget.parentElement?.classList.add('avatar-loaded')}
                          onError={(event) => event.currentTarget.parentElement?.classList.remove('avatar-loaded')}
                        />
                        <span className="avatar-fallback">{initials || 'U'}</span>
                      </div>
                      <div>
                        <div className="list-title">{name}</div>
                        {isCurrentUser && <div className="muted">You</div>}
                      </div>
                    </div>
                    <div className="muted">{member.email}</div>
                    <div className="member-role">{displayRole(member.role)}</div>
                    {isSuperAdmin && <div className="muted">{orgLabel}</div>}
                  </div>
                )
              })}
              {!orgMembers.length && <div className="empty">No members yet.</div>}
            </div>
          </section>

          <section className="card">
            <div className="members-header">
              <div>
                <div className="card-title">Add member</div>
                <div className="card-subtitle">Invite a new member to collaborate with this team.</div>
              </div>
              <button className="secondary" onClick={addOrgMember} disabled={isAllOrgsSelected || loading || !newMemberEmail.trim()}>
                Add member
              </button>
            </div>
            <div className="members-form">
              <input
                value={newMemberEmail}
                onChange={(event) => setNewMemberEmail(event.target.value)}
                placeholder="user@company.com"
                disabled={isAllOrgsSelected}
              />
              <select value={newMemberRole} onChange={(event) => setNewMemberRole(event.target.value)} disabled={isAllOrgsSelected}>
                <option value="org_admin">Admin</option>
                <option value="org_member">Member</option>
              </select>
            </div>
          </section>
        </>
      )}

      {isSuperAdmin && (
        <section className="card">
          <div className="card-title">Organizations</div>
          <div className="list">
            {orgs.map((org) => (
              <div key={org.org_id} className="list-row">
                <div>
                  <div className="list-title">{org.name}</div>
                  <div className="muted">{org.org_id}</div>
                </div>
                <div className="row">
                  <button className="ghost" onClick={() => setOrgStatus(org.org_id, org.status === 'active' ? 'disabled' : 'active')}>
                    {org.status === 'active' ? 'Disable' : 'Enable'}
                  </button>
                </div>
              </div>
            ))}
            {!orgs.length && <div className="empty">No orgs yet</div>}
          </div>
        </section>
      )}
    </div>
  )
}
