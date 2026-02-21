import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Check, Clock, Copy, Key, Code2 } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import DashboardAnalytics from '../../components/DashboardAnalytics'
import { GlassCard, UiButton } from '../../components/ui'

type SetupIndicator = {
  id: string
  label: string
  done: boolean
  to: string
}

export default function BotOverviewTab() {
  const { botId } = useParams()
  const {
    selectedBot,
    embedSnippet,
    copySnippet,
    sources,
    domains,
    selectedBotWidgetConfig,
    getEscalationConfig,
  } = useDashboardData()

  const [escalationEnabled, setEscalationEnabled] = useState<boolean | null>(null)

  useEffect(() => {
    if (!botId) return
    getEscalationConfig(botId).then((config) => {
      setEscalationEnabled(config?.enabled ?? false)
    })
  }, [botId, getEscalationConfig])

  if (!selectedBot || !botId) {
    return <div className="empty-panel">Select a bot to view overview details.</div>
  }

  const hasSources = sources.length > 0
  const hasDesign = !!selectedBotWidgetConfig && Object.keys(selectedBotWidgetConfig).length > 0
  const suggestedMessages = (selectedBotWidgetConfig?.suggestedMessages as { label?: string }[] | undefined) ?? []
  const hasSuggestedMessages = suggestedMessages.length > 0
  const hasVerifiedDomain = domains.some((d) => !!d.verified_at)
  const escalationSetUp = escalationEnabled === true

  const setupIndicators: SetupIndicator[] = [
    { id: 'sources', label: 'Knowledge and Training', done: hasSources, to: `/bots/${botId}/knowledge` },
    { id: 'design', label: 'Design', done: hasDesign, to: `/bots/${botId}/design` },
    { id: 'suggestions', label: 'Suggested messages', done: hasSuggestedMessages, to: `/bots/${botId}/suggested-messages` },
    { id: 'deployment', label: 'Installation', done: hasVerifiedDomain, to: `/bots/${botId}/overview` },
    { id: 'escalation', label: 'Escalations', done: escalationSetUp, to: `/bots/${botId}/suggested-messages` },
  ]

  return (
    <>
      <DashboardAnalytics
        botId={selectedBot.bot_id}
        setupPills={
          <div className="summary-pills-inner">
            {setupIndicators.map((ind) => (
              <Link
                key={ind.id}
                to={ind.to}
                className="summary-pill"
                title={ind.done ? `${ind.label} is set up` : `Set up ${ind.label}`}
              >
                {ind.done ? (
                  <span className="summary-pill-icon summary-pill-icon--check" aria-hidden>
                    <Check size={13} strokeWidth={2.5} />
                  </span>
                ) : (
                  <span className="summary-pill-icon summary-pill-icon--setup" aria-hidden title="Pending">
                    <Clock size={13} strokeWidth={2.5} />
                  </span>
                )}
                <span className="summary-pill-label">{ind.label}</span>
              </Link>
            ))}
          </div>
        }
      />

      <div className="card-grid" style={{ marginTop: 16 }}>
        <GlassCard>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Key size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            Bot details
          </div>
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
        </GlassCard>

        <GlassCard>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Code2 size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            Embed script
          </div>
          <p className="muted" style={{ marginBottom: '0.75rem' }}>Add this snippet to your client website.</p>
          <pre className="snippet">{embedSnippet}</pre>
          <UiButton
            variant="secondary"
            onClick={() => void copySnippet()}
            disabled={!embedSnippet}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.75rem' }}
          >
            <Copy size={16} />
            Copy snippet
          </UiButton>
        </GlassCard>
      </div>
    </>
  )
}
