import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useMatch, useNavigate } from 'react-router-dom'
import { BarChart3, Check, FlaskConical, LayoutDashboard, MoreVertical, Palette, Plus, Search, Settings, BookOpen, MessageSquare, MessageCircle } from 'lucide-react'
import { useDashboardData } from '../hooks/useDashboardData'

export default function DashboardLayout() {
  const { user, bots, selectedBotId, sources, loading, error } = useDashboardData()
  const navigate = useNavigate()
  const [botDropdownOpen, setBotDropdownOpen] = useState(false)
  const [profileMenuOpen, setProfileMenuOpen] = useState(false)
  const [botSearch, setBotSearch] = useState('')
  const botDropdownRef = useRef<HTMLDivElement>(null)
  const profileMenuRef = useRef<HTMLDivElement>(null)

  const botMatch = useMatch('/bots/:botId')
  const botMatchNested = useMatch('/bots/:botId/*')
  const botId = botMatch?.params?.botId ?? botMatchNested?.params?.botId ?? null
  const currentBot = useMemo(
    () => (botId ? bots.find((b) => b.bot_id === botId) ?? null : null),
    [botId, bots]
  )

  const mostRecentBotId = useMemo(() => {
    if (bots.length === 0) return null
    const sorted = [...bots].sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
    return sorted[0]?.bot_id ?? null
  }, [bots])

  const backToDashboardTo = mostRecentBotId ? `/bots/${mostRecentBotId}/overview` : '/dashboard'

  const botStatus = useMemo(() => {
    if (!currentBot) return null
    if (currentBot.bot_id !== selectedBotId) return 'Ready'
    if (sources.length === 0) return 'Not installed'
    return 'Ready'
  }, [currentBot, selectedBotId, sources.length])

  const filteredBots = useMemo(() => {
    const q = botSearch.trim().toLowerCase()
    if (!q) return bots
    return bots.filter(
      (b) =>
        b.display_name.toLowerCase().includes(q) || b.bot_id.toLowerCase().includes(q)
    )
  }, [bots, botSearch])

  const handleBotSelect = useCallback(
    (id: string) => {
      navigate(`/bots/${id}/overview`)
      setBotDropdownOpen(false)
      setBotSearch('')
    },
    [navigate]
  )

  const handleCreateBot = useCallback(() => {
    navigate('/create-bot')
    setBotDropdownOpen(false)
    setBotSearch('')
  }, [navigate])

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      const target = e.target as Node
      if (botDropdownRef.current && !botDropdownRef.current.contains(target)) {
        setBotDropdownOpen(false)
      }
      if (profileMenuRef.current && !profileMenuRef.current.contains(target)) {
        setProfileMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const botTabItems = useMemo(
    () =>
      botId
        ? [
            { id: 'overview', label: 'Overview', to: `/bots/${botId}/overview`, icon: LayoutDashboard },
            { id: 'knowledge', label: 'Knowledge', to: `/bots/${botId}/knowledge`, icon: BookOpen },
            { id: 'design', label: 'Design', to: `/bots/${botId}/design`, icon: Palette },
            { id: 'suggested', label: 'Suggestions', to: `/bots/${botId}/suggested-messages`, icon: MessageCircle },
            { id: 'testing', label: 'Testing', to: `/bots/${botId}/testing`, icon: FlaskConical },
            { id: 'conversations', label: 'Conversations', to: `/bots/${botId}/conversations`, icon: MessageSquare },
            { id: 'settings', label: 'Settings', to: `/bots/${botId}/settings`, icon: Settings },
            { id: 'analytics', label: 'Analytics', to: `/bots/${botId}/analytics`, icon: BarChart3 },
          ]
        : [],
    [botId]
  )

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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="bot-selector-wrap" ref={botDropdownRef}>
          {botId ? (
            <>
              <button
                type="button"
                className="bot-selector-trigger"
                onClick={() => setBotDropdownOpen((o) => !o)}
                aria-expanded={botDropdownOpen}
                aria-haspopup="listbox"
              >
                <div className="logo-dot" aria-hidden="true" />
                <div className="bot-selector-text">
                  <div className="brand-title">
                    {currentBot?.display_name ?? 'Select a bot'}
                  </div>
                  <div className="bot-selector-status">
                    {botStatus != null && (
                      <>
                        <span
                          className={`bot-status-dot ${botStatus === 'Not installed' ? 'bot-status-dot--not-installed' : 'bot-status-dot--ready'}`}
                          aria-hidden="true"
                        />
                        <span className="brand-subtitle">{botStatus}</span>
                      </>
                    )}
                  </div>
                </div>
                <span className="bot-selector-chevron" aria-hidden="true">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="m6 9 6 6 6-6" />
                  </svg>
                </span>
              </button>

              {botDropdownOpen && (
            <div className="bot-selector-dropdown" role="listbox">
              <div className="bot-selector-search-wrap">
                <Search className="bot-selector-search-icon" aria-hidden="true" />
                <input
                  type="search"
                  className="bot-selector-search"
                  placeholder="Search agent..."
                  value={botSearch}
                  onChange={(e) => setBotSearch(e.target.value)}
                  autoFocus
                  aria-label="Search agents"
                />
              </div>
              <div className="bot-selector-list">
                {filteredBots.length === 0 ? (
                  <div className="bot-selector-empty">No agents match</div>
                ) : (
                  filteredBots.map((bot) => (
                    <button
                      key={bot.bot_id}
                      type="button"
                      className="bot-selector-item"
                      onClick={() => handleBotSelect(bot.bot_id)}
                      role="option"
                      aria-selected={currentBot?.bot_id === bot.bot_id}
                    >
                      <div className="logo-dot bot-selector-item-icon" aria-hidden="true" />
                      <span className="bot-selector-item-name">{bot.display_name}</span>
                      {currentBot?.bot_id === bot.bot_id && (
                        <Check className="bot-selector-item-check" aria-hidden="true" />
                      )}
                    </button>
                  ))
                )}
              </div>
              <div className="bot-selector-footer">
                <button type="button" className="bot-selector-create primary" onClick={handleCreateBot}>
                  <Plus className="bot-selector-create-icon" aria-hidden="true" />
                  Create AI Agent
                </button>
              </div>
            </div>
          )}
            </>
          ) : (
            <Link to={backToDashboardTo} className="sidebar-back-link">
              <span className="sidebar-back-arrow" aria-hidden="true">←</span>
              Back to Dashboard
            </Link>
          )}
        </div>

        <nav className="sidebar-nav">
          {botTabItems.map((item) => {
            const Icon = item.icon
            return (
              <div key={item.id} className="nav-group">
                <div className="nav-row">
                  <NavLink
                    className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                    to={item.to}
                    end={item.id === 'overview'}
                  >
                    <span className="nav-link-content">
                      <Icon className="nav-icon" aria-hidden="true" />
                      {item.label}
                    </span>
                  </NavLink>
                </div>
              </div>
            )
          })}
        </nav>

        <div className="sidebar-section sidebar-account" ref={profileMenuRef}>
          <div className="account-row account-row--no-link">
            <div className="avatar avatar-lg">
              <img
                src={profilePicture}
                alt={profileName || 'Profile'}
                onLoad={(e) => e.currentTarget.parentElement?.classList.add('avatar-loaded')}
                onError={(e) => e.currentTarget.parentElement?.classList.remove('avatar-loaded')}
              />
              <span className="avatar-fallback">{profileInitials}</span>
            </div>
            <div className="account-meta">
              <div className="account-name">{profileName || 'Signed in'}</div>
              {profileEmail && <div className="account-email">{profileEmail}</div>}
            </div>
            <button
              type="button"
              className="profile-menu-btn"
              onClick={() => setProfileMenuOpen((o) => !o)}
              aria-expanded={profileMenuOpen}
              aria-haspopup="menu"
              aria-label="Open menu"
            >
              <MoreVertical aria-hidden="true" />
            </button>
          </div>
          {profileMenuOpen && (
            <div className="profile-menu-dropdown" role="menu">
              <Link
                className="profile-menu-item"
                to="/org"
                onClick={() => { setProfileMenuOpen(false) }}
                role="menuitem"
              >
                Team
              </Link>
              <Link
                className="profile-menu-item"
                to="/account"
                onClick={() => { setProfileMenuOpen(false) }}
                role="menuitem"
              >
                Profile
              </Link>
            </div>
          )}
        </div>
      </aside>

      <main className="content">
        <div className="content-shell">
          {error && <div className="alert error">{error}</div>}
          {loading && <div className="alert">Working...</div>}
          <Outlet />
        </div>
      </main>
    </div>
  )
}
