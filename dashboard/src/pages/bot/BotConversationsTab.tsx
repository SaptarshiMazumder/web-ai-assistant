import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Bot, User } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import type {
  ConversationMessageRecord,
  ConversationSessionRecord,
  EscalationRecord,
} from '../../hooks/useDashboardData'

export default function BotConversationsTab() {
  const { selectedBot, listConversations, listEscalations, getConversation, endConversation, getEscalationForSession } =
    useDashboardData()
  const [sessions, setSessions] = useState<ConversationSessionRecord[]>([])
  const [messages, setMessages] = useState<ConversationMessageRecord[]>([])
  const [escalation, setEscalation] = useState<EscalationRecord | null>(null)
  const [escalatedSessionIds, setEscalatedSessionIds] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [searchParams] = useSearchParams()

  useEffect(() => {
    if (!selectedBot) return
    setSessions([])
    setMessages([])
    setEscalation(null)
    setEscalatedSessionIds(new Set())
    setSelectedSession(null)
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
    if (diffMin < 1) return 'Just now'
    if (diffMin < 60) return `${diffMin} min ago`
    const diffH = Math.floor(diffMin / 60)
    if (diffH < 24) return `${diffH} hours ago`
    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)
    if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
    return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
  }

  function statusForSession(s: ConversationSessionRecord | null) {
    if (!s) return null
    if (s.status && s.status !== 'active') return 'Session ended'
    const last = new Date(s.last_active_at)
    if (Number.isNaN(last.getTime())) return null
    const now = new Date()
    const diffMin = Math.floor((now.getTime() - last.getTime()) / 60000)
    if (diffMin <= 5) return 'Active'
    if (diffMin <= 30) return 'Away'
    return 'Session ended'
  }

  async function loadSessions() {
    if (!selectedBot) return
    setLoading(true)
    try {
      const [convData, escData] = await Promise.all([
        listConversations(selectedBot.bot_id, pageSize),
        listEscalations(selectedBot.bot_id, pageSize),
      ])
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
    if (isToday) return `Today ${time}`
    if (isYesterday) return `Yesterday ${time}`
    return `${d.toLocaleDateString()} ${time}`
  }

  async function handleEndSession() {
    if (!selectedBot || !selectedSession) return
    await endConversation(selectedBot.bot_id, selectedSession)
    setSelectedSession(null)
    setMessages([])
    await loadSessions()
  }


  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to view conversations.</div>
  }

  return (
    <div className="conversations-page">
      <div className="card-grid conversation-grid">
      <section className="card conversation-panel conversation-panel--list">
        <div className="card-title">Conversation sessions</div>
        {loading && <div className="muted">Loading...</div>}
        {!loading && sessions.length === 0 && <div className="muted">No conversations yet.</div>}
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
                {statusForSession(s) && (
                  <span
                    className={`conversation-status ${
                      statusForSession(s) === 'Active'
                        ? 'active'
                        : statusForSession(s) === 'Away'
                        ? 'inactive'
                        : 'ended'
                    }`}
                  >
                    {statusForSession(s)}
                  </span>
                )}
                {s.message_count} messages
                {escalatedSessionIds.has(s.session_id) && (
                  <span className="conversation-pill conversation-pill--escalated">Escalated</span>
                )}
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="card conversation-panel conversation-panel--detail">
        <div className="card-title">
          Conversation details{messages.length ? ` (${messages.length} messages)` : ''}
        </div>
        {!selectedSession && <div className="muted">Select a session to view messages.</div>}
        {selectedSession && (
          <>
            {selectedSessionRecord && (
              <>
                <div className="detail-row">
                  <span>Status</span>
                  <span
                    className={`conversation-status ${
                      statusForSession(selectedSessionRecord) === 'Active'
                        ? 'active'
                        : statusForSession(selectedSessionRecord) === 'Away'
                        ? 'inactive'
                        : statusForSession(selectedSessionRecord) === 'Session ended'
                        ? 'ended'
                        : 'inactive'
                    }`}
                  >
                    {statusForSession(selectedSessionRecord)}
                  </span>
                </div>
              </>
            )}
            <div className="conversation-actions">
              <button type="button" className="secondary" onClick={() => setSelectedSession(null)}>
                Back to list
              </button>
              <button type="button" className="ghost" onClick={() => void handleEndSession()}>
                End session
              </button>
            </div>
            <div className="conversation-messages">
              {(() => {
                let escalationRendered = false
                return messages.map((m, idx) => {
                const prev = messages[idx - 1]
                const showDate = toDateKey(m.created_at || null) !== toDateKey(prev?.created_at || null)
                if (m.role === 'system') {
                  const shouldShowEscalation =
                    !escalationRendered &&
                    !!escalation?.visitor_email &&
                    m.content?.toLowerCase().includes('escalated to support')
                  if (shouldShowEscalation) {
                    escalationRendered = true
                  }
                  return (
                    <div key={m.message_id || `${m.role}-${idx}`}>
                      {showDate && (
                        <div className="conversation-date-separator">
                          {toDateKey(m.created_at) || toDateKey(new Date().toISOString())}
                        </div>
                      )}
                      <div className="conversation-system-note">{m.content}</div>
                      {shouldShowEscalation && (
                        <div className="conversation-escalation-box">
                          <div className="conversation-escalation-row">
                            <span>Email</span>
                            <span>{escalation?.visitor_email}</span>
                          </div>
                          {escalation?.details && (
                            <div className="conversation-escalation-row">
                              <span>Details</span>
                              <span>{escalation.details}</span>
                            </div>
                          )}
                        </div>
                      )}
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
              })
              })()}
              {messages.length === 0 && !loading && <div className="muted">No messages found.</div>}
            </div>
          </>
        )}
      </section>
      </div>
    </div>
  )
}
