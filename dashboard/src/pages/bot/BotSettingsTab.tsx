import { Trash2, AlertTriangle, ShieldAlert } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

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
    <AnimatedPage>
      <SectionHeader
        eyebrow="Configuration"
        title="Bot settings"
        subtitle="Manage your bot configuration and take destructive actions when needed."
      />

      <GlassCard>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444' }}>
          <ShieldAlert size={18} />
          Danger zone
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <p style={{ margin: '0 0 8px 0', color: 'var(--ui-flow-muted)' }}>
              Permanently delete this bot. This will remove all associated data including:
            </p>
            <ul style={{ margin: '0 0 16px 0', paddingLeft: '20px', color: 'var(--ui-flow-muted)', lineHeight: 1.8 }}>
              <li>Bot configuration</li>
              <li>All domains and verifications</li>
              <li>Knowledge base and crawled content</li>
              <li>All crawl jobs and history</li>
            </ul>
          </div>
          <UiButton
            variant="primary"
            className="danger"
            onClick={handleDelete}
            disabled={loading || !selectedBot}
            style={{ alignSelf: 'flex-start', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
          >
            <Trash2 size={16} />
            Delete Bot
          </UiButton>
          {error && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#ef4444', fontSize: '0.9rem' }}>
              <AlertTriangle size={15} />
              {error}
            </div>
          )}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
