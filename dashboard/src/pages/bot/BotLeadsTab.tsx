import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { CalendarDays, ExternalLink, User, UsersRound } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader } from '../../components/ui'
import { getEscalationContact } from '../../utils/escalationIdentity'

function formatLeadDate(iso: string, locale?: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(locale || undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export default function BotLeadsTab() {
  const { botId } = useParams()
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)
  const { selectedBot, listEscalations } = useDashboardData()
  const [leads, setLeads] = useState<EscalationRecord[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!botId) return
    setLoading(true)
    listEscalations(botId, 100)
      .then((data) => {
        setLeads(data?.escalations ?? [])
      })
      .finally(() => setLoading(false))
  }, [botId, listEscalations])

  function leadStatusLabel(status?: string | null) {
    switch ((status || '').toLowerCase()) {
      case 'resolved':
        return tr('Resolved', '解決済み')
      case 'canceled':
        return tr('Canceled', 'キャンセル')
      case 'expired':
        return tr('Expired', '期限切れ')
      case 'open':
        return tr('Open', '未対応')
      default:
        return status || tr('Unknown', '不明')
    }
  }

  if (!botId || !selectedBot) {
    return <div className="empty-panel">{tr('Select a bot to view leads.', 'リードを表示するボットを選択してください。')}</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow={tr('Contacts', '連絡先')}
        title={tr('Leads', 'リード')}
        subtitle={tr('Captured when visitors request human follow-up in chat.', 'チャットでヒューマンサポートを依頼した訪問者の情報です。')}
      />

      <GlassCard>
        {loading ? (
          <div className="muted" style={{ padding: '0.5rem 0' }}>{tr('Loading leads...', 'リードを読み込み中...')}</div>
        ) : leads.length === 0 ? (
          <div style={{ padding: '2rem 0', textAlign: 'center' }}>
            <UsersRound size={36} style={{ color: 'var(--ui-flow-muted, #94a3b8)', marginBottom: '0.75rem' }} />
            <p className="muted" style={{ margin: 0 }}>{tr('No leads yet. Human support requests will appear here once visitors request follow-up.', 'リードはまだありません。訪問者がフォローアップを依頼すると、ここに表示されます。')}</p>
          </div>
        ) : (
          <div className="leads-table-wrap">
            <table className="leads-table">
              <thead>
                <tr>
                  <th><User size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />{tr('Contact', '連絡先')}</th>
                  <th><CalendarDays size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />{tr('Date', '日時')}</th>
                  <th>{tr('Status', 'ステータス')}</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {leads.map((lead) => {
                  const contact = getEscalationContact(lead, tr)
                  return (
                    <tr key={lead.escalation_id}>
                      <td className="leads-email">
                        <div style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.02em', color: 'var(--ui-flow-muted)' }}>{contact.label}</div>
                        <div>{contact.value}</div>
                      </td>
                      <td className="leads-date">{formatLeadDate(lead.created_at, isJa ? 'ja-JP' : undefined)}</td>
                      <td>
                        <span className={`leads-status leads-status--${lead.status}`}>
                          {leadStatusLabel(lead.status)}
                        </span>
                      </td>
                      <td>
                        <Link
                          to={`/bots/${botId}/conversations?session=${lead.linked_session_id || lead.session_id}`}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: 'var(--ui-flow-accent)', fontSize: '0.85rem', fontWeight: 500 }}
                        >
                          <ExternalLink size={13} />
                          {tr('View', '表示')}
                        </Link>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </AnimatedPage>
  )
}
