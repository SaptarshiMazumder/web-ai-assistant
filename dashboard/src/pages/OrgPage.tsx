import { useDashboardData } from '../hooks/useDashboardData'

export default function OrgPage() {
  const {
    isSuperAdmin,
    activeOrgId,
    orgs,
    newOrgName,
    setNewOrgName,
    createOrg,
    setOrgStatus,
    orgDisplayNameInput,
    setOrgDisplayNameInput,
    saveOrgName,
    loading,
  } = useDashboardData()

  if (!activeOrgId && !isSuperAdmin) {
    return <div className="empty-panel">No organization assigned yet.</div>
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">Organization</div>
          <div className="page-subtitle">Manage org settings and visibility.</div>
        </div>
      </div>

      <div className="card-grid">
        {isSuperAdmin ? (
          <section className="card">
            <div className="card-title">Create org</div>
            <div className="stack">
              <input value={newOrgName} onChange={(event) => setNewOrgName(event.target.value)} placeholder="Org name" />
              <button className="primary" onClick={createOrg} disabled={loading || !newOrgName.trim()}>
                Create org
              </button>
            </div>
          </section>
        ) : (
          <section className="card">
            <div className="card-title">Org name</div>
            <div className="stack">
              <input
                value={orgDisplayNameInput}
                onChange={(event) => setOrgDisplayNameInput(event.target.value)}
                placeholder="Organization name"
              />
              <button className="secondary" onClick={saveOrgName} disabled={loading || !orgDisplayNameInput.trim()}>
                Save org name
              </button>
            </div>
          </section>
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
    </div>
  )
}
