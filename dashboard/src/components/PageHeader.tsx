import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { LogOut, Settings } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

type PageHeaderProps = {
  title: string
  subtitle?: string
  children?: React.ReactNode
}

export default function PageHeader({ title, subtitle, children }: PageHeaderProps) {
  const { user, logout } = useDashboardData()
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  const profileName = useMemo(() => {
    const given = (user as { given_name?: string | null } | undefined)?.given_name?.trim()
    const family = (user as { family_name?: string | null } | undefined)?.family_name?.trim()
    const combined = `${given || ''} ${family || ''}`.trim()
    if (combined) return combined
    const name = (user as { name?: string | null } | undefined)?.name?.trim()
    if (name) return name
    const email = user?.email?.trim()
    if (!email) return 'User'
    const local = email.split('@')[0]
    return local ? local.replace(/[._-]+/g, ' ').trim() : email
  }, [user])

  const profileEmail = user?.email?.trim() || ''
  const profileInitials = useMemo(
    () =>
      profileName
        .split(' ')
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0].toUpperCase())
        .join('') || 'U',
    [profileName]
  )

  const profilePicture =
    (user as { picture?: string | null } | undefined)?.picture?.trim() ||
    `https://ui-avatars.com/api/?name=${encodeURIComponent(profileName)}&background=6366f1&color=fff&size=128`

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      const target = e.target as Node
      if (menuRef.current && !menuRef.current.contains(target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <span className="page-title-text">{title}</span>
            </span>
          </div>
          {subtitle != null && <div className="page-subtitle">{subtitle}</div>}
        </div>
        <div className="page-actions">
          {children}
          <div className="page-header-profile" ref={menuRef}>
            <button
              type="button"
              className="page-header-profile-btn"
              onClick={() => setMenuOpen((o) => !o)}
              aria-expanded={menuOpen}
              aria-haspopup="menu"
              aria-label="Open profile menu"
            >
              <div className="avatar avatar-sm">
                <img
                  src={profilePicture}
                  alt={profileName}
                  onLoad={(e) => e.currentTarget.parentElement?.classList.add('avatar-loaded')}
                  onError={(e) => e.currentTarget.parentElement?.classList.remove('avatar-loaded')}
                />
                <span className="avatar-fallback">{profileInitials}</span>
              </div>
            </button>
            {menuOpen && (
              <div className="page-header-profile-dropdown" role="menu">
                <div className="page-header-profile-info">
                  <div className="avatar avatar-lg">
                    <img
                      src={profilePicture}
                      alt={profileName}
                      onLoad={(e) => e.currentTarget.parentElement?.classList.add('avatar-loaded')}
                      onError={(e) => e.currentTarget.parentElement?.classList.remove('avatar-loaded')}
                    />
                    <span className="avatar-fallback">{profileInitials}</span>
                  </div>
                  <div className="page-header-profile-name">{profileName}</div>
                  {profileEmail && <div className="page-header-profile-email">{profileEmail}</div>}
                </div>
                <div className="page-header-profile-actions">
                  <Link
                    className="page-header-profile-item"
                    to="/settings"
                    onClick={() => setMenuOpen(false)}
                    role="menuitem"
                  >
                    <Settings className="page-header-profile-icon" aria-hidden="true" />
                    Settings
                  </Link>
                  <button
                    type="button"
                    className="page-header-profile-item"
                    onClick={() => {
                      setMenuOpen(false)
                      logout({ logoutParams: { returnTo: window.location.origin } })
                    }}
                    role="menuitem"
                  >
                    <LogOut className="page-header-profile-icon" aria-hidden="true" />
                    Sign out
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="page-divider" />
    </>
  )
}
