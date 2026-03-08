import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AlertCircle, CheckCircle, Clock, ExternalLink, RotateCcw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useDashboardData, type EscalationRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader } from '../../components/ui'
import { getEscalationSubtitle, getEscalationTitle } from '../../utils/escalationIdentity'

function notificationStatusMeta(
  status: string | null | undefined,
  tr: (en: string, ja: string) => string
) {
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
        icon: <AlertCircle size={14} />,
        label: tr('Open', '対応中'),
      }
  }
}

export default function BotNotificationsTab() {
  const { botId } = useParams()
  const navigate = useNavigate()
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)
  const {
    selectedBot,
    listEscalations,
    markEscalationRead,
  } = useDashboardData()
  const [notifications, setNotifications] = useState<EscalationRecord[]>([])
  const [loading, setLoading] = useState(false)
  const pageSize = 200

  useEffect(() => {
    if (!selectedBot?.bot_id) return
    setNotifications([])
    void loadNotifications()
    const timer = window.setInterval(() => {
      void loadNotifications()
    }, 30000)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBot?.bot_id])

  async function loadNotifications() {
    if (!selectedBot) return
    setLoading(true)
    try {
      const data = await listEscalations(selectedBot.bot_id, pageSize)
      setNotifications(data.escalations || [])
    } finally {
      setLoading(false)
    }
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

  function titleForNotification(record: EscalationRecord) {
    return getEscalationTitle(record, tr)
  }

  function subtitleForNotification(record: EscalationRecord) {
    return getEscalationSubtitle(record, tr)
  }

  async function openNotification(record: EscalationRecord) {
    if (!botId || !selectedBot) return
    if (record.notification_is_unread) {
      const updated = await markEscalationRead(selectedBot.bot_id, record.escalation_id)
      setNotifications((prev) =>
        prev.map((item) => (item.escalation_id === record.escalation_id ? (updated || { ...item, notification_is_unread: false }) : item))
      )
    }
    const targetSessionId = record.linked_session_id || record.session_id
    navigate(`/bots/${botId}/conversations?session=${encodeURIComponent(targetSessionId)}`)
  }

  if (!botId) {
    return <div className="empty-panel">{tr('Select a bot to view notifications.', '通知を表示するボットを選択してください。')}</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow={tr('Support', 'サポート')}
        title={tr('Notifications', '通知')}
        subtitle={tr('Escalation alerts appear here and open the linked conversation.', 'サポート依頼の通知をここで確認し、該当チャットを開けます。')}
      />

      <GlassCard style={{ marginTop: '0.5rem' }}>
        {!loading && notifications.length === 0 && (
          <div className="muted" style={{ padding: '1rem 0' }}>
            {tr('No notifications yet.', '通知はまだありません。')}
          </div>
        )}
        <div className="escalation-list">
          {notifications.map((record) => {
            const statusMeta = notificationStatusMeta(record.status, tr)
            return (
              <button
                key={record.escalation_id}
                type="button"
                className={`conversation-row notification-row${record.notification_is_unread ? ' notification-row--unread' : ''}`}
                onClick={() => {
                  void openNotification(record)
                }}
              >
                <div className="conversation-row-top">
                  <div className="conversation-title">
                    {record.notification_is_unread && <span className="notification-dot" aria-hidden="true" />}
                    {titleForNotification(record)}
                  </div>
                  <div className="conversation-time">{formatTime(record.created_at)}</div>
                </div>
                <div className="conversation-meta escalation-meta-row">
                  <span className="escalation-email">{subtitleForNotification(record)}</span>
                  <div className="escalation-status-actions">
                    <span className={`notification-pill${record.notification_is_unread ? ' unread' : ''}`}>
                      {record.notification_is_unread ? tr('Unread', '未読') : tr('Read', '既読')}
                    </span>
                    <span className={`conversation-status ${statusMeta.className}`}>
                      {statusMeta.icon}
                      {statusMeta.label}
                    </span>
                  </div>
                </div>
                {record.details && (
                  <div className="muted" style={{ marginTop: '0.35rem', fontSize: '0.85rem' }}>
                    {record.details}
                  </div>
                )}
                <div className="notification-open-link">
                  <ExternalLink size={14} />
                  {tr('Open conversation', 'チャットを開く')}
                </div>
              </button>
            )
          })}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
