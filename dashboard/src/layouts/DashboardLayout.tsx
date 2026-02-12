import { useMemo } from 'react'
import { Link, NavLink, Outlet, useLocation, useMatch } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useDashboardData } from '../hooks/useDashboardData'
import ironManIcon from '../assets/icons8/iron-man.png'
import {
  primaryNavConfig,
  botsSecondaryItemsBase,
  botTabSecondaryItems,
} from '../navigation/sidebarConfig'

function getActivePrimaryId(pathname: string): string {
  if (pathname === '/' || pathname.startsWith('/dashboard')) return 'home'
  if (pathname === '/account') return 'account'
  if (pathname === '/org' || pathname.startsWith('/org/')) return 'team'
  if (pathname === '/billing' || pathname.startsWith('/billing')) return 'billing'
  if (pathname === '/settings' || pathname.startsWith('/settings')) return 'settings'
  if (pathname === '/bots' || pathname.startsWith('/bots/') || pathname.startsWith('/create-bot')) return 'bots'
  return 'home'
}

export default function DashboardLayout() {
  const { loading, error } = useDashboardData()
  const location = useLocation()

  const botMatch = useMatch('/bots/:botId')
  const botMatchNested = useMatch('/bots/:botId/*')
  const botId = botMatch?.params?.botId ?? botMatchNested?.params?.botId ?? null

  const activePrimaryId = useMemo(() => getActivePrimaryId(location.pathname), [location.pathname])

  const secondaryItems = useMemo(
    () => (botId ? botTabSecondaryItems(botId) : botsSecondaryItemsBase),
    [botId]
  )

  const showSecondaryPanel = activePrimaryId === 'bots' && !!botId

  return (
    <div className={`app-shell ${showSecondaryPanel ? 'app-shell--secondary-visible' : ''}`}>
      <aside className="sidebar-primary" aria-label="Main navigation">
        <div className="sidebar-primary-brand" aria-hidden="true">
          <img src={ironManIcon} alt="" className="sidebar-brand-icon" />
        </div>
        <nav className="sidebar-primary-nav">
          {primaryNavConfig.map((item) => {
            const Icon = item.icon
            const isActive = activePrimaryId === item.id
            const to = item.id === 'bots' ? '/bots' : item.path
            return (
              <Link
                key={item.id}
                to={to}
                className={`sidebar-primary-link ${isActive ? 'active' : ''}`}
                title={item.label}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon className="sidebar-primary-icon" aria-hidden="true" />
              </Link>
            )
          })}
        </nav>
      </aside>

      <AnimatePresence>
        {showSecondaryPanel && (
          <motion.aside
            className="sidebar"
            initial={{ opacity: 0, x: -14 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -14 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
          >
            <Link to="/bots" className="sidebar-back-link">
              <span className="sidebar-back-arrow" aria-hidden="true">←</span>
              All bots
            </Link>

            <nav className="sidebar-nav">
              {secondaryItems.map((item) => {
                const Icon = item.icon
                return (
                  <div key={item.id} className="nav-group">
                    <div className="nav-row">
                      <NavLink
                        className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                        to={item.to}
                        end={item.to === '/' || item.id === 'overview'}
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
          </motion.aside>
        )}
      </AnimatePresence>

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
