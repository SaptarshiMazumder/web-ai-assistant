import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotOverviewTab() {
  const { selectedBot, embedSnippet, copySnippet } = useDashboardData()

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to view overview details.</div>
  }

  return (
    <div className="card-grid">
      <section className="card">
        <div className="card-title">Bot details</div>
        <div className="detail-row">
          <span>Bot ID</span>
          <code>{selectedBot.bot_id}</code>
        </div>
        <div className="detail-row">
          <span>Publishable key</span>
          <code>{selectedBot.publishable_key}</code>
        </div>
        <div className="detail-row">
          <span>Secret key</span>
          <code>{selectedBot.secret_key}</code>
        </div>
        <div className="detail-row">
          <span>Created</span>
          <span>{new Date(selectedBot.created_at).toLocaleString()}</span>
        </div>
      </section>

      <section className="card">
        <div className="card-title">Embed script</div>
        <p className="muted">Add this snippet to your client website.</p>
        <pre className="snippet">{embedSnippet}</pre>
        <button className="secondary" onClick={() => void copySnippet()} disabled={!embedSnippet}>
          Copy snippet
        </button>
      </section>
    </div>
  )
}
