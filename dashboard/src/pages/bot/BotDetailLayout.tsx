import { useEffect } from 'react'
import { NavLink, Outlet, useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotDetailLayout() {
  const { botId } = useParams()
  const { selectedBot, setSelectedBotId, isSuperAdmin, activeOrgId } = useDashboardData()

  useEffect(() => {
    if (botId) {
      setSelectedBotId(botId)
    }
  }, [botId, setSelectedBotId])

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bot details.</div>
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">{selectedBot?.display_name || 'Bot'}</div>
          <div className="page-subtitle">{selectedBot?.bot_id || 'Select a bot to view details.'}</div>
        </div>
      </div>

      <div className="tab-list">
        <NavLink className={({ isActive }) => `tab ${isActive ? 'active' : ''}`} to="overview">
          Overview
        </NavLink>
        <NavLink className={({ isActive }) => `tab ${isActive ? 'active' : ''}`} to="knowledge">
          Knowledge
        </NavLink>
        <NavLink className={({ isActive }) => `tab ${isActive ? 'active' : ''}`} to="sources">
          Sources
        </NavLink>
        <NavLink className={({ isActive }) => `tab ${isActive ? 'active' : ''}`} to="design">
          Design
        </NavLink>
        <NavLink className={({ isActive }) => `tab ${isActive ? 'active' : ''}`} to="settings">
          Settings
        </NavLink>
        <NavLink className={({ isActive }) => `tab ${isActive ? 'active' : ''}`} to="analytics">
          Analytics
        </NavLink>
      </div>

      <Outlet />
    </div>
  )
}
