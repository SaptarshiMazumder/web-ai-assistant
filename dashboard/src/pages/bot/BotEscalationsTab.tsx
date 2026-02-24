import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Check, CheckCircle, Clock, RotateCcw } from 'lucide-react'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader } from '../../components/ui'

export default function BotEscalationsTab() {
  const { botId } = useParams()
  const navigate = useNavigate()
  const { selectedBot, listEscalations, updateEscalationStatus } = useDashboardData()
  const [escalations, setEscalations] = useState<EscalationRecord[]>([])
  const pageSize = 200

  useEffect(() => {
    if (!selectedBot?.bot_id) return
    setEscalations([])
    void loadEscalations()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBot?.bot_id])

  async function loadEscalations() {
    if (!selectedBot) return
    const data = await listEscalations(selectedBot.bot_id, pageSize)
    setEscalations(data.escalations || [])
  }

  function formatTime(ts?: string | null) {
    if (!ts) return ''
    const d = new Date(ts)
    if (Number.isNaN(d.getTime())) return ''
    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffMin = Math.floor(diffMs / 60000)
    if (diffMin < 1) return 'Just now'
    if (diffMin < 60) return `${diffMin} min ago`
    const diffH = Math.floor(diffMin / 60)
    if (diffH < 24) return `${diffH} hours ago`
    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)
    if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
    return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
  }

  function titleForEscalation(e: EscalationRecord) {
    return e.title || e.site_title || e.site_url || e.session_id
  }

  async function handleResolve(escalationId: string) {
    if (!selectedBot) return
    await updateEscalationStatus(selectedBot.bot_id, escalationId, 'resolved')
    setEscalations((prev) =>
      prev.map((item) =>
        item.escalation_id === escalationId ? { ...item, status: 'resolved' } : item
      )
    )
  }

  async function handleReopen(escalationId: string) {
    if (!selectedBot) return
    await updateEscalationStatus(selectedBot.bot_id, escalationId, 'open')
    setEscalations((prev) =>
      prev.map((item) =>
        item.escalation_id === escalationId ? { ...item, status: 'open' } : item
      )
    )
  }

  if (!botId) {
    return <div className="empty-panel">Select a bot to view human support requests.</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow="Support"
        title="Human support requests"
        subtitle="Review and resolve customer support requests."
      />

      <GlassCard style={{ marginTop: '0.5rem' }}>
        {escalations.length === 0 && (
          <div className="muted" style={{ padding: '1rem 0' }}>No human support requests yet.</div>
        )}
        <div className="escalation-list">
          {escalations.map((e) => (
            <button
              key={e.escalation_id}
              type="button"
              className="conversation-row"
              onClick={() => navigate(`/bots/${botId}/conversations?session=${encodeURIComponent(e.session_id)}`)}
            >
              <div className="conversation-row-top">
                <div className="conversation-title">{titleForEscalation(e)}</div>
                <div className="conversation-time">{formatTime(e.created_at)}</div>
              </div>
              <div className="conversation-meta escalation-meta-row">
                <span className="escalation-email">{e.visitor_email}</span>
                <div className="escalation-status-actions">
                  {e.status === 'resolved' ? (
                    <>
                      <span className="conversation-status resolved">
                        <CheckCircle size={14} />
                        Resolved
                      </span>
                      <button
                        type="button"
                        className="icon-pill"
                        aria-label="Undo resolve"
                        onClick={(evt) => {
                          evt.stopPropagation()
                          void handleReopen(e.escalation_id)
                        }}
                      >
                        <RotateCcw size={14} />
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="conversation-status pending">
                        <Clock size={14} />
                        Pending
                      </span>
                      <button
                        type="button"
                        className="pill-action"
                        onClick={(evt) => {
                          evt.stopPropagation()
                          void handleResolve(e.escalation_id)
                        }}
                      >
                        <Check size={14} />
                        Mark as resolved
                      </button>
                    </>
                  )}
                </div>
              </div>
              {e.details && (
                <div className="muted" style={{ marginTop: '0.35rem', fontSize: '0.85rem' }}>
                  {e.details}
                </div>
              )}
            </button>
          ))}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
