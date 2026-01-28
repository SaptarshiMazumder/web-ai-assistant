import { Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotSettingsTab() {
  const { selectedBot, deleteBot, loading, error } = useDashboardData()
  const navigate = useNavigate()

  const handleDelete = async () => {
    if (!selectedBot) return
    
    const confirmed = window.confirm(
      `Are you sure you want to delete "${selectedBot.display_name}"? This action cannot be undone and will delete all associated data including domains, knowledge base, and crawl jobs.`
    )
    
    if (!confirmed) return

    const success = await deleteBot(selectedBot.bot_id)
    if (success) {
      navigate('/bots')
    }
  }

  return (
    <div className="page-body">
      <div className="card">
        <div className="card-header">
          <h3>Danger Zone</h3>
        </div>
        <div className="card-body">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <p style={{ margin: '0 0 8px 0', color: 'var(--text-secondary)' }}>
                Permanently delete this bot. This will remove all associated data including:
              </p>
              <ul style={{ margin: '0 0 16px 0', paddingLeft: '20px', color: 'var(--text-secondary)' }}>
                <li>Bot configuration</li>
                <li>All domains and verifications</li>
                <li>Knowledge base and crawled content</li>
                <li>All crawl jobs and history</li>
              </ul>
            </div>
            <button
              className="danger"
              onClick={handleDelete}
              disabled={loading || !selectedBot}
              style={{ alignSelf: 'flex-start' }}
            >
              <Trash2 size={16} style={{ marginRight: '8px' }} />
              Delete Bot
            </button>
            {error && <div className="error-message">{error}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
