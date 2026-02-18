import { useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation, useParams } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { botTabSecondaryItems } from '../../navigation/sidebarConfig'
import PageHeader from '../../components/PageHeader'

export default function BotDetailLayout() {
  const { botId } = useParams()
  const location = useLocation()
  const { selectedBot, setSelectedBotId, isSuperAdmin, activeOrgId } = useDashboardData()

  const [mobileDropdownOpen, setMobileDropdownOpen] = useState(false)
  const [dropdownTop, setDropdownTop] = useState(0)
  const btnRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (botId) {
      setSelectedBotId(botId)
    }
  }, [botId, setSelectedBotId])

  // Close dropdown on route change
  useEffect(() => {
    setMobileDropdownOpen(false)
  }, [location.pathname])

  const secondaryItems = useMemo(
    () => (botId ? botTabSecondaryItems(botId) : []),
    [botId]
  )

  const activeSecondaryItem = useMemo(() => {
    const exact = secondaryItems.find(item => location.pathname === item.to)
    if (exact) return exact
    return secondaryItems.find(item => location.pathname.startsWith(item.to + '/')) ?? secondaryItems[0]
  }, [secondaryItems, location.pathname])

  const groups = useMemo(() => {
    const result: { header?: string; items: typeof secondaryItems }[] = []
    let currentGroup: typeof result[0] | null = null
    secondaryItems.forEach(item => {
      if (!currentGroup || currentGroup.header !== item.header) {
        currentGroup = { header: item.header, items: [] }
        result.push(currentGroup)
      }
      currentGroup.items.push(item)
    })
    return result
  }, [secondaryItems])

  const openDropdown = () => {
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect()
      setDropdownTop(rect.bottom + 6)
    }
    setMobileDropdownOpen(true)
  }

  const ActiveIcon = activeSecondaryItem?.icon

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bot details.</div>
  }

  return (
    <div className="page">
      <PageHeader title={selectedBot?.display_name || 'Bot'} />

      {/* Mobile page nav pill — below agent name, only visible on mobile */}
      {secondaryItems.length > 0 && (
        <>
          <div className="mobile-page-nav">
            <button
              ref={btnRef}
              type="button"
              className={`mobile-page-btn${mobileDropdownOpen ? ' open' : ''}`}
              onClick={mobileDropdownOpen ? () => setMobileDropdownOpen(false) : openDropdown}
              aria-expanded={mobileDropdownOpen}
            >
              {ActiveIcon && <ActiveIcon className="nav-icon" size={18} strokeWidth={1.8} aria-hidden />}
              <span className="mobile-page-btn-label">{activeSecondaryItem?.label}</span>
              <ChevronDown className={`mobile-page-chevron${mobileDropdownOpen ? ' open' : ''}`} size={15} strokeWidth={2} />
            </button>

            {mobileDropdownOpen && (
              <div
                className="mobile-page-dropdown"
                style={{ top: dropdownTop, maxHeight: `calc(100dvh - 60px - ${dropdownTop}px)` }}
              >
                {groups.map((group, i) => (
                  <div key={i} className={group.header ? 'nav-category' : ''}>
                    {group.header && <div className="nav-category-header">{group.header}</div>}
                    {group.items.map(item => {
                      const Icon = item.icon
                      return (
                        <div key={item.id}>
                          {item.separator && <hr className="nav-separator" />}
                          <NavLink
                            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                            to={item.to}
                            end={item.id === 'overview'}
                            onClick={() => setMobileDropdownOpen(false)}
                          >
                            <span className="nav-link-content">
                              <Icon className="nav-icon" aria-hidden />
                              {item.label}
                            </span>
                          </NavLink>
                        </div>
                      )
                    })}
                  </div>
                ))}
              </div>
            )}
          </div>

          {mobileDropdownOpen && (
            <div className="mobile-dropdown-backdrop" onClick={() => setMobileDropdownOpen(false)} />
          )}
        </>
      )}

      <div className="page-body page-body-centered">
        <Outlet />
      </div>
    </div>
  )
}
