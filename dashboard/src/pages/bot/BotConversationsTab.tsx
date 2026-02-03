import { useEffect, useMemo, useState } from 'react'
import { Bot, User, ChevronLeft, ChevronRight } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import type { ConversationMessageRecord, ConversationSessionRecord } from '../../hooks/useDashboardData'

export default function BotConversationsTab() {
  const { selectedBot, listConversations, getConversation, endConversation } = useDashboardData()
  const [sessions, setSessions] = useState<ConversationSessionRecord[]>([])
  const [messages, setMessages] = useState<ConversationMessageRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [cursorStack, setCursorStack] = useState<string[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [currentCursor, setCurrentCursor] = useState<string | null>(null)
  const [totalCount, setTotalCount] = useState<number | null>(null)

  useEffect(() => {
    if (!selectedBot) return
    setSessions([])
    setMessages([])
    setSelectedSession(null)
    setCursorStack([])
    setTotalCount(null)
    void loadSessions(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBot?.bot_id])

  const canPrev = cursorStack.length > 0
  const canNext = !!nextCursor
  const pageSize = 10
  const currentPage = cursorStack.length + 1
  const totalPages = totalCount ? Math.max(1, Math.ceil(totalCount / pageSize)) : null

  const selectedSessionRecord = useMemo(
    () => sessions.find((s) => s.session_id === selectedSession) || null,
    [sessions, selectedSession]
  )

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

  async function loadSessions(cursor: string | null) {
    if (!selectedBot) return
    setLoading(true)
    try {
      const data = await listConversations(selectedBot.bot_id, pageSize, cursor)
      setSessions(data.sessions || [])
      setNextCursor(data.next_cursor || null)
      setCurrentCursor(cursor)
      setTotalCount(typeof data.total_count === 'number' ? data.total_count : null)
    } finally {
      setLoading(false)
    }
  }

  async function openSession(sessionId: string) {
    if (!selectedBot) return
    setSelectedSession(sessionId)
    setLoading(true)
    try {
      const data = await getConversation(selectedBot.bot_id, sessionId, 200)
      setMessages(data || [])
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
    await loadSessions(currentCursor)
  }


  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to view conversations.</div>
  }

  return (
    <div className="card-grid">
      <section className="card">
        <div className="card-title">Conversation sessions</div>
        {loading && <div className="muted">Loading...</div>}
        {!loading && sessions.length === 0 && <div className="muted">No conversations yet.</div>}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.75rem' }}>
          {sessions.map((s) => (
            <button
              key={s.session_id}
              type="button"
              className="conversation-row"
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
              </div>
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.75rem' }}>
          <button
            type="button"
            className="circle-nav"
            disabled={!canPrev}
            aria-label="Previous page"
            onClick={() => {
              const stack = [...cursorStack]
              const prev = stack.pop() || null
              setCursorStack(stack)
              void loadSessions(prev)
            }}
          >
            <ChevronLeft size={18} />
          </button>
          <div className="muted" style={{ fontSize: '0.95rem' }}>
            Page {currentPage}
            {totalPages ? ` of ${totalPages}` : ''}
          </div>
          <button
            type="button"
            className="circle-nav"
            disabled={!canNext}
            aria-label="Next page"
            onClick={() => {
              if (!nextCursor) return
              if (currentCursor) setCursorStack([...cursorStack, currentCursor])
              void loadSessions(nextCursor)
            }}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </section>

      <section className="card">
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
              {messages.map((m, idx) => {
                const prev = messages[idx - 1]
                const showDate = toDateKey(m.created_at || null) !== toDateKey(prev?.created_at || null)
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
              {messages.length === 0 && !loading && <div className="muted">No messages found.</div>}
            </div>
          </>
        )}
      </section>
    </div>
  )
}
