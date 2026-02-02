import { Bot, Plus } from 'lucide-react'
import { useMemo } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'

function mostRecentBotId(bots: { bot_id: string; created_at: string }[]): string | null {
  if (bots.length === 0) return null
  const sorted = [...bots].sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
  return sorted[0]?.bot_id ?? null
}

export default function BotsPage() {
  const { bots, loading, isSuperAdmin, activeOrgId, refreshAll } = useDashboardData()
  const navigate = useNavigate()

  const mostRecentId = useMemo(() => mostRecentBotId(bots), [bots])

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bots.</div>
  }

  if (!loading && bots.length === 0) {
    return <Navigate to="/create-bot" replace />
  }

  if (!loading && bots.length > 0 && mostRecentId) {
    return <Navigate to={`/bots/${mostRecentId}/overview`} replace />
  }

  const canCreateBot = !isSuperAdmin || (activeOrgId && activeOrgId !== '__all__')

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="page-title">
            <span className="page-title-row">
              <Bot className="page-title-icon" aria-hidden="true" />
              <span className="page-title-divider">|</span>
              <span className="page-title-text">Bots</span>
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

      <div className="page-body page-body-narrow">
        <div className="bot-row">
          <button className="create-bot-button" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
            <Plus className="create-bot-icon" aria-hidden="true" />
            <span>Create bot</span>
          </button>

          {bots.map((bot) => (
            <button key={bot.bot_id} className="bot-card" onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}>
              <div className="list-title">{bot.display_name}</div>
              <div className="muted">{bot.bot_id}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
