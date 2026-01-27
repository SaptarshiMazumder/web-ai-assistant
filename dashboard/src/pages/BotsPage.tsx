import { Bot } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'

export default function BotsPage() {
  const { bots, newBotName, setNewBotName, createBot, loading, isSuperAdmin, activeOrgId, refreshAll } = useDashboardData()
  const navigate = useNavigate()

  const canCreateBot = !isSuperAdmin || (activeOrgId && activeOrgId !== "__all__")

  const handleCreate = async () => {
    if (!canCreateBot) {
      return
    }
    const data = await createBot()
    if (data) {
      navigate(`/bots/${data.bot_id}/overview`)
    }
  }

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bots.</div>
  }

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

      <div className="bot-row">
        <div className="bot-card bot-create-card">
          <div className="list-title">Create bot</div>
          <div className="stack">
            <input
              className="bot-input"
              value={newBotName}
              onChange={(event) => setNewBotName(event.target.value)}
              placeholder="Bot display name"
            />
            <button className="primary bot-action" onClick={handleCreate} disabled={loading || !newBotName.trim() || !canCreateBot}>
              Create bot
            </button>
          </div>
        </div>

        {bots.map((bot) => (
          <button key={bot.bot_id} className="bot-card" onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}>
            <div className="list-title">{bot.display_name}</div>
            <div className="muted">{bot.bot_id}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
