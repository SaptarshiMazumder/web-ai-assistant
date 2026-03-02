import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Bot, User, Search, Download, ArrowLeft, XCircle, MessageCircle, MessagesSquare } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useDashboardData } from '../../hooks/useDashboardData'
import type {
  ConversationMessageRecord,
  ConversationSessionRecord,
  EscalationRecord,
} from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

export default function BotConversationsTab() {
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)
  const { selectedBot, listConversations, searchConversations, exportConversationsCsv, listEscalations, getConversation, endConversation, takeOverConversation, getEscalationForSession } =
    useDashboardData()
  const [sessions, setSessions] = useState<ConversationSessionRecord[]>([])
  const [messages, setMessages] = useState<ConversationMessageRecord[]>([])
  const [escalation, setEscalation] = useState<EscalationRecord | null>(null)
  const [takeoverLoading, setTakeoverLoading] = useState(false)
  const [escalatedSessionIds, setEscalatedSessionIds] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [mobileView, setMobileView] = useState<'list' | 'detail'>('list')
  const [query, setQuery] = useState('')
  const [searchParams] = useSearchParams()

  useEffect(() => {
    if (!selectedBot) return
    setSessions([])
    setMessages([])
    setEscalation(null)
    setEscalatedSessionIds(new Set())
    setSelectedSession(null)
    setMobileView('list')
    void loadSessions()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBot?.bot_id])

  const pageSize = 200

  const selectedSessionRecord = useMemo(
    () => sessions.find((s) => s.session_id === selectedSession) || null,
    [sessions, selectedSession]
  )

  useEffect(() => {
    const sessionParam = searchParams.get('session')
    if (!sessionParam || !selectedBot) return
    if (selectedSession === sessionParam) return
    void openSession(sessionParam)
  }, [searchParams, sessions, selectedBot, selectedSession])

  function formatListTime(ts?: string | null) {
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

  type SessionStatus = 'active' | 'away' | 'ended' | null

  function statusForSession(s: ConversationSessionRecord | null): SessionStatus {
    if (!s) return null
    if (s.status && s.status !== 'active') return 'ended'
    const last = new Date(s.last_active_at)
    if (Number.isNaN(last.getTime())) return null
    const now = new Date()
    const diffMin = Math.floor((now.getTime() - last.getTime()) / 60000)
    if (diffMin <= 5) return 'active'
    if (diffMin <= 30) return 'away'
    return 'ended'
  }

  function statusLabel(status: SessionStatus) {
    if (status === 'active') return tr('Active', '稼働中')
    if (status === 'away') return tr('Away', '離席')
    if (status === 'ended') return tr('Session ended', '終了')
    return ''
  }

  async function loadSessions() {
    if (!selectedBot) return
    setLoading(true)
    try {
      const convData = query.trim()
        ? await searchConversations(selectedBot.bot_id, { q: query.trim(), limit: pageSize })
        : await listConversations(selectedBot.bot_id, pageSize)
      const escData = await listEscalations(selectedBot.bot_id, pageSize)
      setSessions(convData.sessions || [])
      setEscalatedSessionIds(
        new Set((escData.escalations || []).map((e) => e.session_id))
      )
    } finally {
      setLoading(false)
    }
  }

  async function openSession(sessionId: string) {
    if (!selectedBot) return
    setSelectedSession(sessionId)
    setMobileView('detail')
    setLoading(true)
    try {
      const [data, escalationInfo] = await Promise.all([
        getConversation(selectedBot.bot_id, sessionId, 200),
        getEscalationForSession(selectedBot.bot_id, sessionId),
      ])
      setMessages(data || [])
      setEscalation(escalationInfo || null)
    } finally {
      setLoading(false)
    }
  }

  // No polling: conversations update on manual refresh or re-open.

  function toDateKey(ts?: string | null) {
    if (!ts) return ''
    const d = new Date(ts)
    if (Number.isNaN(d.getTime())) return ''
    return d.toDateString()
  }

  function formatMessageTime(ts?: string | null) {
    const d = ts ? new Date(ts) : new Date()
    if (Number.isNaN(d.getTime())) return ''
    const now = new Date()
    const isToday = d.toDateString() === now.toDateString()
    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)
    const isYesterday = d.toDateString() === yesterday.toDateString()
    const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    if (isToday) return isJa ? `今日 ${time}` : `Today ${time}`
    if (isYesterday) return isJa ? `昨日 ${time}` : `Yesterday ${time}`
    return `${d.toLocaleDateString()} ${time}`
  }

  async function handleEndSession() {
    if (!selectedBot || !selectedSession) return
    await endConversation(selectedBot.bot_id, selectedSession)
    setSelectedSession(null)
    setMobileView('list')
    setMessages([])
    await loadSessions()
  }

  async function handleTakeOver() {
    if (!selectedBot || !selectedSession || !selectedSessionRecord) return
    const ch = (selectedSessionRecord.channel || '').toLowerCase()
    if (ch !== 'instagram' && ch !== 'line') return
    setTakeoverLoading(true)
    try {
      await takeOverConversation(selectedBot.bot_id, selectedSession)
      setEscalatedSessionIds((prev) => new Set([...prev, selectedSession]))
    } finally {
      setTakeoverLoading(false)
    }
  }

  function channelLabel(ch?: string | null): { emoji: string; label: string } {
    switch ((ch || '').toLowerCase()) {
      case 'web': return { emoji: '🌐', label: tr('Website', 'ウェブサイト') }
      case 'test': return { emoji: '🧪', label: tr('Test', 'テスト') }
      case 'line': return { emoji: '💬', label: tr('LINE', 'ライン') }
      case 'instagram': return { emoji: '📸', label: tr('Instagram', 'インスタグラム') }
      default: {
        const raw = (ch || tr('unknown', '不明')).trim()
        return { emoji: '💬', label: raw.charAt(0).toUpperCase() + raw.slice(1) }
      }
    }
  }


  if (!selectedBot) {
    return <div className="empty-panel">{tr('Select a bot to view conversations.', '会話を表示するボットを選択してください。')}</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow={tr('Conversations', '会話')}
        title={tr('Live inbox and replay', 'ライブ受信トレイと履歴')}
        subtitle={tr('Track active chats, human support requests, and message timelines in one place.', '進行中チャット、ヒューマンサポート依頼、メッセージ履歴をまとめて確認できます。')}
      />
      <div className={`card-grid conversation-grid${mobileView === 'detail' ? ' conversation-grid--mobile-detail' : ''}`}>
        <GlassCard className="conversation-panel conversation-panel--list">
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <MessagesSquare size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            {tr('Conversation sessions', '会話セッション')}
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={tr('Search messages...', 'メッセージを検索...')}
              style={{ flex: 1 }}
            />
            <UiButton variant="secondary" onClick={() => void loadSessions()} disabled={loading} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <Search size={14} />
              {tr('Search', '検索')}
            </UiButton>
            <UiButton
              variant="ghost"
              onClick={() => void exportConversationsCsv(selectedBot.bot_id, { q: query.trim() || null })}
              disabled={loading}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
            >
              <Download size={14} />
              {tr('Export CSV', 'CSVを出力')}
            </UiButton>
          </div>
          {loading && <div className="muted">{tr('Loading...', '読み込み中...')}</div>}
          {!loading && sessions.length === 0 && <div className="muted">{tr('No conversations yet.', '会話はまだありません。')}</div>}
          <div className="conversation-list">
            {sessions.map((s) => (
              <button
                key={s.session_id}
                type="button"
                className={`conversation-row${s.session_id === selectedSession ? ' conversation-row--selected' : ''}`}
                onClick={() => void openSession(s.session_id)}
              >
                <div className="conversation-row-top">
                  <div className="conversation-title">
                    {s.title || s.site_title || s.site_url || s.session_id}
                  </div>
                  <div className="conversation-time">{formatListTime(s.last_active_at)}</div>
                </div>
                <div className="conversation-meta">
                  {statusForSession(s) && (() => {
                    const status = statusForSession(s)
                    if (!status) return null
                    return (
                      <span
                        className={`conversation-status ${status === 'active'
                          ? 'active'
                          : status === 'away'
                            ? 'inactive'
                            : 'ended'
                          }`}
                      >
                        {statusLabel(status)}
                      </span>
                    )
                  })()}
                  {isJa ? `${s.message_count}件` : `${s.message_count} messages`}
                  <span className="conversation-pill conversation-pill--channel">
                    {channelLabel(s.channel).emoji} {channelLabel(s.channel).label}
                  </span>
                  {escalatedSessionIds.has(s.session_id) && (
                    <span className="conversation-pill conversation-pill--escalated">{tr('Support requested', 'サポート依頼')}</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </GlassCard>

        <GlassCard className="conversation-panel conversation-panel--detail">
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <MessageCircle size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            {tr('Conversation details', '会話詳細')}{messages.length ? (isJa ? `（${messages.length}件）` : ` (${messages.length} messages)`) : ''}
          </div>
          {!selectedSession && <div className="muted">{tr('Select a session to view messages.', 'メッセージを表示するセッションを選択してください。')}</div>}
          {selectedSession && (
            <>
              {selectedSessionRecord && (
                <>
                  <div className="detail-row">
                    <span>{tr('Status', 'ステータス')}</span>
                    {(() => {
                      const status = statusForSession(selectedSessionRecord)
                      return (
                        <span
                          className={`conversation-status ${status === 'active'
                            ? 'active'
                            : status === 'away'
                              ? 'inactive'
                              : status === 'ended'
                                ? 'ended'
                                : 'inactive'
                            }`}
                        >
                          {statusLabel(status)}
                        </span>
                      )
                    })()}
                  </div>
                  <div className="detail-row">
                    <span>{tr('Channel', 'チャネル')}</span>
                    <span className="conversation-pill conversation-pill--channel">
                      {channelLabel(selectedSessionRecord.channel).emoji} {channelLabel(selectedSessionRecord.channel).label}
                    </span>
                  </div>
                </>
              )}
              {escalation && (
                <div className="conversation-escalation-box conversation-escalation-box--top">
                  <div className="conversation-escalation-title">{tr('Support requested', 'サポート依頼')}</div>
                  <div className="conversation-escalation-row">
                    <span>{tr('Email', 'メール')}</span>
                    <span>{escalation.visitor_email}</span>
                  </div>
                  <div className="conversation-escalation-row">
                    <span>{tr('Status', 'ステータス')}</span>
                    <span>{escalation.status === 'resolved' ? tr('Resolved', '解決済み') : tr('Pending', '保留')}</span>
                  </div>
                  <div className="conversation-escalation-row">
                    <span>{tr('Requested', '依頼日時')}</span>
                    <span>{formatMessageTime(escalation.created_at || null)}</span>
                  </div>
                  {escalation.details && (
                    <div className="conversation-escalation-row">
                      <span>{tr('Details', '詳細')}</span>
                      <span>{escalation.details}</span>
                    </div>
                  )}
                </div>
              )}
              <div className="conversation-actions">
                <UiButton variant="secondary" onClick={() => { setSelectedSession(null); setMobileView('list') }} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                  <ArrowLeft size={14} />
                  {tr('Back to list', '一覧に戻る')}
                </UiButton>
                {(selectedSessionRecord?.channel || '').toLowerCase() === 'instagram' || (selectedSessionRecord?.channel || '').toLowerCase() === 'line' ? (
                  <UiButton
                    variant="primary"
                    onClick={() => void handleTakeOver()}
                    disabled={takeoverLoading || escalatedSessionIds.has(selectedSession)}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                  >
                    {escalatedSessionIds.has(selectedSession) ? tr('Taken over', '引き継ぎ済み') : tr('Take over', '引き継ぐ')}
                  </UiButton>
                ) : null}
                <UiButton variant="ghost" onClick={() => void handleEndSession()} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                  <XCircle size={14} />
                  {tr('End session', 'セッションを終了')}
                </UiButton>
              </div>
              <div className="conversation-messages">
                {messages.map((m, idx) => {
                  const prev = messages[idx - 1]
                  const showDate = toDateKey(m.created_at || null) !== toDateKey(prev?.created_at || null)
                  if (m.role === 'system') {
                    return (
                      <div key={m.message_id || `${m.role}-${idx}`}>
                        {showDate && (
                          <div className="conversation-date-separator">
                            {toDateKey(m.created_at) || toDateKey(new Date().toISOString())}
                          </div>
                        )}
                        <div className="conversation-system-note">{m.content}</div>
                      </div>
                    )
                  }
                  return (
                    <div key={m.message_id || `${m.role}-${idx}`}>
                      {showDate && (
                        <div className="conversation-date-separator">
                          {toDateKey(m.created_at) || toDateKey(new Date().toISOString())}
                        </div>
                      )}
                      <div className={`conversation-bubble-row conversation-bubble-row--${m.role}`}>
                        <div className={`conversation-avatar conversation-avatar--${m.role}`}>
                          {m.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                        </div>
                        <div className={`conversation-message conversation-message--${m.role}`}>
                          {m.sender_name && <div className="conversation-sender-name">{m.sender_name}</div>}
                          <div className="conversation-message-content">{m.content}</div>
                          <div className="conversation-message-time">
                            {formatMessageTime(m.created_at || null)}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
                {messages.length === 0 && !loading && <div className="muted">{tr('No messages found.', 'メッセージが見つかりません。')}</div>}
              </div>
            </>
          )}
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
