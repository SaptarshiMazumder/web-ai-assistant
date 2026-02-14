import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { CalendarDays, ExternalLink, Mail, UsersRound } from 'lucide-react'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader } from '../../components/ui'

function formatLeadDate(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export default function BotLeadsTab() {
  const { botId } = useParams()
  const { selectedBot, listEscalations } = useDashboardData()
  const [leads, setLeads] = useState<EscalationRecord[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!botId) return
    setLoading(true)
    listEscalations(botId, 100).then((data) => {
      setLeads(data?.escalations ?? [])
    }).finally(() => setLoading(false))
  }, [botId, listEscalations])

  if (!botId || !selectedBot) {
    return <div className="empty-panel">Select a bot to view leads.</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow="Contacts"
        title="Leads"
        subtitle="Captured when visitors request human follow-up in chat."
      />

      <GlassCard>
        {loading ? (
          <div className="muted" style={{ padding: '0.5rem 0' }}>Loading leads...</div>
        ) : leads.length === 0 ? (
          <div style={{ padding: '2rem 0', textAlign: 'center' }}>
            <UsersRound size={36} style={{ color: 'var(--ui-flow-muted, #94a3b8)', marginBottom: '0.75rem' }} />
            <p className="muted" style={{ margin: 0 }}>No leads yet. Escalations will appear here once visitors request follow-up.</p>
          </div>
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
                      <Link
                        to={`/bots/${botId}/conversations?session=${lead.session_id}`}
                        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: 'var(--ui-flow-accent)', fontSize: '0.85rem', fontWeight: 500 }}
                      >
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
    </AnimatedPage>
  )
}
