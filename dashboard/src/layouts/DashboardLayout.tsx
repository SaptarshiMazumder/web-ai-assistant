import { useMemo, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import { sidebarConfig } from '../navigation/sidebarConfig'

export default function DashboardLayout() {
  const { user, logout, loading, error, refreshAll } = useDashboardData()
  const location = useLocation()
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const profileName = useMemo(() => {
    const given = (user as { given_name?: string | null } | undefined)?.given_name?.trim()
    const family = (user as { family_name?: string | null } | undefined)?.family_name?.trim()
    const combined = `${given || ''} ${family || ''}`.trim()
    if (combined) return combined
    const name = (user as { name?: string | null } | undefined)?.name?.trim()
    if (name) return name
    const email = user?.email?.trim()
    if (!email) return ''
    const local = email.split('@')[0]
    return local ? local.replace(/[._-]+/g, ' ').trim() : email
  }, [user])

  const profileEmail = user?.email?.trim() || ''
  const profileInitials =
    profileName
      .split(' ')
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0].toUpperCase())
      .join('') || 'U'

  const profilePicture =
    (user as { picture?: string | null } | undefined)?.picture?.trim() ||
    `https://ui-avatars.com/api/?name=${encodeURIComponent(profileName || profileEmail || 'User')}&background=0f172a&color=fff&size=128`

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
            <div className="account-row">
              <div className="avatar avatar-lg">
                <img
                  src={profilePicture}
                  alt={profileName || 'Profile'}
                  onLoad={(event) => event.currentTarget.parentElement?.classList.add('avatar-loaded')}
                  onError={(event) => event.currentTarget.parentElement?.classList.remove('avatar-loaded')}
                />
                <span className="avatar-fallback">{profileInitials}</span>
              </div>
              <div className="account-meta">
                <div className="account-name">{profileName || 'Signed in'}</div>
                {profileEmail && <div className="account-email">{profileEmail}</div>}
              </div>
            </div>
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
