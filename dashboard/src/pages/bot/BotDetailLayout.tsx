import { Bot } from 'lucide-react'
import { useEffect } from 'react'
import { Outlet, useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotDetailLayout() {
  const { botId } = useParams()
  const { selectedBot, setSelectedBotId, isSuperAdmin, activeOrgId, refreshAll, loading } = useDashboardData()

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
          <div className="page-title">
            <span className="page-title-row">
              <Bot className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">{selectedBot?.display_name || 'Bot'}</span>
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

      <div className="page-body">
        <Outlet />
      </div>
    </div>
  )
}
