import { useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'

export default function BotsPage() {
  const { bots, newBotName, setNewBotName, createBot, loading, isSuperAdmin, activeOrgId } = useDashboardData()
  const navigate = useNavigate()

  const handleCreate = async () => {
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
          <div className="page-title">Bots</div>
          <div className="page-subtitle">Create, select, and manage bots.</div>
        </div>
      </div>

      <section className="card">
        <div className="card-title">Create bot</div>
        <div className="stack">
          <input value={newBotName} onChange={(event) => setNewBotName(event.target.value)} placeholder="Bot display name" />
          <button className="primary" onClick={handleCreate} disabled={loading || !newBotName.trim()}>
            Create bot
          </button>
        </div>
      </section>

      <section className="card">
        <div className="card-title">Bots list</div>
        {!bots.length ? (
          <div className="empty-panel">
            No bots yet.
            <div className="spacer-sm" />
            <button className="primary" onClick={handleCreate} disabled={loading || !newBotName.trim()}>
              Create bot
            </button>
          </div>
        ) : (
          <div className="list">
            {bots.map((bot) => (
              <button key={bot.bot_id} className="list-row button-row" onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}>
                <div>
                  <div className="list-title">{bot.display_name}</div>
                  <div className="muted">{bot.bot_id}</div>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
