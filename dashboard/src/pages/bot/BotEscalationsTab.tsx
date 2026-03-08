import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Check, CheckCircle, Clock, RotateCcw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader } from '../../components/ui'
import { getEscalationSubtitle, getEscalationTitle } from '../../utils/escalationIdentity'

export default function BotEscalationsTab() {
  const { botId } = useParams()
  const navigate = useNavigate()
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)
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
    if (diffMin < 1) return tr('Just now', 'たった今')
    if (diffMin < 60) return isJa ? `${diffMin}分前` : `${diffMin} min ago`
    const diffH = Math.floor(diffMin / 60)
    if (diffH < 24) return isJa ? `${diffH}時間前` : `${diffH} hours ago`
    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)
    if (d.toDateString() === yesterday.toDateString()) return tr('Yesterday', '昨日')
    return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
  }

  function titleForEscalation(e: EscalationRecord) {
    return getEscalationTitle(e, tr)
  }

  function sessionTargetForEscalation(e: EscalationRecord) {
    return e.linked_session_id || e.session_id
  }

  function subtitleForEscalation(e: EscalationRecord) {
    return getEscalationSubtitle(e, tr)
  }

  function escalationStatusMeta(status?: string | null) {
    switch ((status || '').toLowerCase()) {
      case 'resolved':
        return {
          className: 'resolved',
          icon: <CheckCircle size={14} />,
          label: tr('Resolved', '解決済み'),
        }
      case 'canceled':
        return {
          className: 'ended',
          icon: <RotateCcw size={14} />,
          label: tr('Canceled', 'キャンセル'),
        }
      case 'expired':
        return {
          className: 'ended',
          icon: <Clock size={14} />,
          label: tr('Expired', '期限切れ'),
        }
      default:
        return {
          className: 'pending',
          icon: <Clock size={14} />,
          label: tr('Pending', '保留'),
        }
    }
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
    return <div className="empty-panel">{tr('Select a bot to view support requests.', 'サポート依頼を表示するボットを選択してください。')}</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow={tr('Support', 'サポート')}
        title={tr('Support requests', 'サポート依頼')}
        subtitle={tr('Review and resolve customer support requests.', '顧客のサポート依頼を確認して対応できます。')}
      />

      <GlassCard style={{ marginTop: '0.5rem' }}>
        {escalations.length === 0 && (
          <div className="muted" style={{ padding: '1rem 0' }}>{tr('No support requests yet.', 'サポート依頼はまだありません。')}</div>
        )}
        <div className="escalation-list">
          {escalations.map((e) => (
            <button
              key={e.escalation_id}
              type="button"
              className="conversation-row"
              onClick={() => navigate(`/bots/${botId}/conversations?session=${encodeURIComponent(sessionTargetForEscalation(e))}`)}
            >
              {(() => {
                const statusMeta = escalationStatusMeta(e.status)
                const isOpen = (e.status || '').toLowerCase() === 'open'
                return (
                  <>
              <div className="conversation-row-top">
                <div className="conversation-title">{titleForEscalation(e)}</div>
                <div className="conversation-time">{formatTime(e.created_at)}</div>
              </div>
              <div className="conversation-meta escalation-meta-row">
                <span className="escalation-email">{subtitleForEscalation(e)}</span>
                <div className="escalation-status-actions">
                  {!isOpen ? (
                    <>
                      <span className={`conversation-status ${statusMeta.className}`}>
                        {statusMeta.icon}
                        {statusMeta.label}
                      </span>
                      <button
                        type="button"
                        className="icon-pill"
                        aria-label={tr('Reopen support request', 'サポート依頼を再開する')}
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
                      <span className={`conversation-status ${statusMeta.className}`}>
                        {statusMeta.icon}
                        {statusMeta.label}
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
                        {tr('Mark as resolved', '解決済みにする')}
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
                  </>
                )
              })()}
            </button>
          ))}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
