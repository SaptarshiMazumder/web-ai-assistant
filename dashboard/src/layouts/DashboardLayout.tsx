import { useEffect, useMemo, useState } from 'react'
import { Link, NavLink, Outlet, useLocation, useMatch } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'

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
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const { t } = useTranslation()

  const botMatch = useMatch('/bots/:botId')
  const botMatchNested = useMatch('/bots/:botId/*')
  const botId = botMatch?.params?.botId ?? botMatchNested?.params?.botId ?? null

  const activePrimaryId = useMemo(() => getActivePrimaryId(location.pathname), [location.pathname])

  const secondaryItems = useMemo(
    () => (botId ? botTabSecondaryItems(botId) : botsSecondaryItemsBase),
    [botId]
  )

  const showSecondaryPanel = activePrimaryId === 'bots' && !!botId

  // Close mobile sidebar on route change
  useEffect(() => {
    setMobileSidebarOpen(false)
  }, [location.pathname])

  return (
    <div className={`app-shell ${showSecondaryPanel ? 'app-shell--secondary-visible' : ''}`}>
      <aside className="sidebar-primary" aria-label={t('nav.mainNavigation', 'Main navigation')}>
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
                title={t(item.label)}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon className="sidebar-primary-icon" aria-hidden="true" />
              </Link>
            )
          })}
        </nav>
      </aside>

      {/* Backdrop overlay for mobile sidebar */}
      {showSecondaryPanel && (
        <div
          className={`sidebar-overlay ${mobileSidebarOpen ? 'mobile-sidebar-open' : ''}`}
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      <AnimatePresence>
        {showSecondaryPanel && (
          <motion.aside
            className={`sidebar ${mobileSidebarOpen ? 'mobile-sidebar-open' : ''}`}
            initial={{ opacity: 0, x: -14 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -14 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
          >
            <Link to="/bots" className="sidebar-back-link">
              <span className="sidebar-back-arrow" aria-hidden="true">←</span>
              {t('nav.allAgentsBack', 'All bots')}
            </Link>

            <div className="sidebar-content" style={{ flex: 1, overflowY: 'auto', marginRight: '-1rem', paddingRight: '1rem' }}>
              <nav className="sidebar-nav">
                {(() => {
                  // Filter out settings for the main scrollable area
                  const mainItems = secondaryItems.filter(item => item.id !== 'settings')

                  // Group items by consecutive header
                  const groups: { header?: string; items: typeof secondaryItems }[] = []
                  let currentGroup: { header?: string; items: typeof secondaryItems } | null = null

                  mainItems.forEach((item) => {
                    if (!currentGroup || currentGroup.header !== item.header) {
                      currentGroup = { header: item.header, items: [] }
                      groups.push(currentGroup)
                    }
                    currentGroup.items.push(item)
                  })

                  return groups.map((group, groupIndex) => (
                    <div key={groupIndex} className={group.header ? 'nav-category' : 'nav-group-container'}>
                      {group.header && <div className="nav-category-header">{t(group.header)}</div>}
                      {group.items.map((item) => {
                        const Icon = item.icon
                        return (
                          <div key={item.id}>
                            {item.separator && <hr className="nav-separator" />}
                            <div className="nav-group">
                              <div className="nav-row">
                                <NavLink
                                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                                  to={item.to}
                                  end={item.to === '/' || item.id === 'overview'}
                                  onClick={() => setMobileSidebarOpen(false)}
                                >
                                  <span className="nav-link-content">
                                    <Icon className="nav-icon" aria-hidden="true" />
                                    {t(item.label)}
                                  </span>
                                </NavLink>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ))
                })()}
              </nav>

            </div>

            {/* Fixed Settings Footer */}
            {secondaryItems.some(item => item.id === 'settings') && (
              <div className="sidebar-footer" style={{ flexShrink: 0, marginTop: 'auto', paddingTop: '0.5rem' }}>
                <hr className="nav-separator" style={{ margin: '0.5rem 0 1rem' }} />
                <nav className="sidebar-nav">
                  {secondaryItems.filter(item => item.id === 'settings').map((item) => {
                    const Icon = item.icon
                    return (
                      <div key={item.id} className="nav-group">
                        <div className="nav-row">
                          <NavLink
                            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                            to={item.to}
                            onClick={() => setMobileSidebarOpen(false)}
                          >
                            <span className="nav-link-content">
                              <Icon className="nav-icon" aria-hidden="true" />
                              {t(item.label)}
                            </span>
                          </NavLink>
                        </div>
                      </div>
                    )
                  })}
                </nav>
              </div>
            )}
          </motion.aside>
        )}
      </AnimatePresence>

      <main className="content">
        <div className="content-shell">
          {error && <div className="alert error">{error}</div>}
          {loading && <div className="alert">{t('common.working', 'Working...')}</div>}
          <Outlet />
        </div>
      </main>
    </div>
  )
}
