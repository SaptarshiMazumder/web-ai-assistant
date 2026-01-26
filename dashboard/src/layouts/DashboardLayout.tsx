import { useMemo, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import { sidebarConfig } from '../navigation/sidebarConfig'

export default function DashboardLayout() {
  const { user, logout, activeOrgId, setActiveOrgId, isSuperAdmin, orgs, orgDisplayName, loading, error, refreshAll } =
    useDashboardData()
  const location = useLocation()
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const activePaths = useMemo(() => {
    return sidebarConfig.map((item) => ({
      id: item.id,
      active: location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(`${item.to}/`)),
    }))
  }, [location.pathname])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo-dot" />
          <div>
            <div className="brand-title">Web AI Admin</div>
            <div className="brand-subtitle">Workspace console</div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="section-title">Active org</div>
          <div className="stack">
            {isSuperAdmin ? (
              <select value={activeOrgId || ''} onChange={(event) => setActiveOrgId(event.target.value)}>
                <option value="" disabled>
                  Select org
                </option>
                {orgs.map((org) => (
                  <option key={org.org_id} value={org.org_id}>
                    {org.name} ({org.org_id})
                  </option>
                ))}
              </select>
            ) : (
              <div className="muted">{orgDisplayName || activeOrgId || 'No org assigned'}</div>
            )}
          </div>
        </div>

        <nav className="sidebar-nav">
          {sidebarConfig.map((item) => {
            const activeEntry = activePaths.find((entry) => entry.id === item.id)
            const isActive = activeEntry?.active ?? false
            const hasChildren = Boolean(item.children?.length)
            const isExpanded = hasChildren ? expanded[item.id] ?? isActive : false
            return (
              <div key={item.id} className={`nav-group ${isActive ? 'active' : ''}`}>
                <div className="nav-row">
                  <NavLink className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} to={item.to} end={item.to === '/'}>
                    {item.label}
                  </NavLink>
                  {hasChildren && (
                    <button
                      type="button"
                      className="nav-toggle"
                      onClick={() => setExpanded((prev) => ({ ...prev, [item.id]: !isExpanded }))}
                      aria-label={`Toggle ${item.label}`}
                    >
                      {isExpanded ? '–' : '+'}
                    </button>
                  )}
                </div>
                {hasChildren && isExpanded && (
                  <div className="nav-children">
                    {item.children?.map((child) => (
                      <NavLink key={child.id} className={({ isActive }) => `nav-sublink ${isActive ? 'active' : ''}`} to={child.to}>
                        {child.label}
                      </NavLink>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </nav>

        <div className="sidebar-section sidebar-account">
          <div className="section-title">Account</div>
          <div className="stack">
            <div className="muted">{user?.email || 'Signed in'}</div>
            <button className="ghost" onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}>
              Sign out
            </button>
          </div>
        </div>
      </aside>

      <main className="content">
        <div className="content-toolbar">
          <button className="ghost" onClick={refreshAll} disabled={loading}>
            Refresh
          </button>
        </div>
        {error && <div className="alert error">{error}</div>}
        {loading && <div className="alert">Working...</div>}
        <Outlet />
      </main>
    </div>
  )
}
