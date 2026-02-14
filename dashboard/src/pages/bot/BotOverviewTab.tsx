import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Check, Clock, Copy, Key, Code2, CalendarDays, Mail, ExternalLink, BarChart3 } from 'lucide-react'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import DashboardAnalytics from '../../components/DashboardAnalytics'
import { GlassCard, UiButton } from '../../components/ui'

type SetupIndicator = {
  id: string
  label: string
  done: boolean
  to: string
}

function formatLeadDate(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
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
    listEscalations,
  } = useDashboardData()

  const [escalationEnabled, setEscalationEnabled] = useState<boolean | null>(null)
  const [leads, setLeads] = useState<EscalationRecord[]>([])

  useEffect(() => {
    if (!botId) return
    getEscalationConfig(botId).then((config) => {
      setEscalationEnabled(config?.enabled ?? false)
    })
  }, [botId, getEscalationConfig])

  useEffect(() => {
    if (!botId) return
    listEscalations(botId, 20).then((data) => {
      setLeads(data?.escalations ?? [])
    })
  }, [botId, listEscalations])

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
    { id: 'suggestions', label: 'Suggested messages', done: hasSuggestedMessages, to: `/bots/${botId}/design` },
    { id: 'deployment', label: 'Installation', done: hasVerifiedDomain, to: `/bots/${botId}/overview` },
    { id: 'escalation', label: 'Escalations', done: escalationSetUp, to: `/bots/${botId}/escalations` },
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

      <GlassCard className="leads-section">
        <div className="leads-section-header">
          <div>
            <div className="card-title" style={{ marginBottom: '0.2rem' }}>Leads</div>
            <p className="card-subtitle" style={{ margin: 0 }}>Captured when visitors request human follow-up in chat.</p>
          </div>
          <UiButton variant="secondary" onClick={() => { }} style={{ fontSize: '0.88rem', padding: '0.4rem 0.85rem' }}>
            <Link to={`/bots/${botId}/escalations?tab=escalations`} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'inherit' }}>
              <BarChart3 size={14} />
              View all
            </Link>
          </UiButton>
        </div>
        {leads.length === 0 ? (
          <div className="muted" style={{ padding: '0.5rem 0' }}>No leads yet. Escalations will appear here once visitors request follow-up.</div>
        ) : (
          <div className="leads-table-wrap">
            <table className="leads-table">
              <thead>
                <tr>
                  <th><Mail size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />Email</th>
                  <th><CalendarDays size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />Date</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {leads.map((lead) => (
                  <tr key={lead.escalation_id}>
                    <td className="leads-email">{lead.visitor_email}</td>
                    <td className="leads-date">{formatLeadDate(lead.created_at)}</td>
                    <td>
                      <span className={`leads-status leads-status--${lead.status}`}>{lead.status}</span>
                    </td>
                    <td>
                      <Link to={`/bots/${botId}/conversations?session=${lead.session_id}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: 'var(--ui-flow-accent)', fontSize: '0.85rem', fontWeight: 500 }}>
                        <ExternalLink size={13} />
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>

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
