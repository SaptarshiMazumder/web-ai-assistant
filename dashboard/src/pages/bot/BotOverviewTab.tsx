import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Check, Clock } from 'lucide-react'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import DashboardAnalytics from '../../components/DashboardAnalytics'

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
    { id: 'suggestions', label: 'Suggested messages', done: hasSuggestedMessages, to: `/bots/${botId}/suggested-messages` },
    { id: 'deployment', label: 'Installation', done: hasVerifiedDomain, to: `/bots/${botId}/overview` },
    { id: 'escalation', label: 'Escalations', done: escalationSetUp, to: `/bots/${botId}/escalations` },
  ]

  return (
    <>
      {/* Summary: setup pills + metrics; all inside Summary block */}
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
                    <Check size={12} strokeWidth={3} />
                  </span>
                ) : (
                  <span className="summary-pill-icon summary-pill-icon--setup" aria-hidden title="Pending">
                    <Clock size={14} strokeWidth={2} />
                  </span>
                )}
                <span className="summary-pill-label">{ind.label}</span>
              </Link>
            ))}
          </div>
        }
      />

      {/* Leads section: people who entered their email (from escalations) */}
      <section className="card leads-section" style={{ marginTop: 24 }}>
        <div className="leads-section-header">
          <div className="card-title">Leads</div>
          <Link to={`/bots/${botId}/escalations?tab=escalations`} className="secondary" style={{ fontSize: '0.9rem', padding: '0.4rem 0.75rem' }}>
            View all
          </Link>
        </div>
        <p className="muted" style={{ marginTop: -8, marginBottom: 12 }}>
          Leads are captured when someone enters their email in the chat (e.g. when requesting to be contacted).
        </p>
        {leads.length === 0 ? (
          <div className="muted">No leads yet.</div>
        ) : (
          <div className="leads-table-wrap">
            <table className="leads-table">
              <thead>
                <tr>
                  <th>Email</th>
                  <th>Date</th>
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
                      <Link to={`/bots/${botId}/conversations?session=${lead.session_id}`} className="ghost" style={{ fontSize: '0.85rem' }}>
                        View conversation
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Bot details + Embed script */}
      <div className="card-grid" style={{ marginTop: 24 }}>
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
    </>
  )
}
