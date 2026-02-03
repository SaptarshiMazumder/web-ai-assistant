import { useEffect, useMemo, useState } from 'react'
import { Bot, User } from 'lucide-react'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import type { ConversationMessageRecord, ConversationSessionRecord } from '../../hooks/useDashboardData'

export default function BotConversationsTab() {
  const { selectedBot, listConversations, getConversation, endConversation, orgMembers, activeOrgId } = useDashboardData()
  const { getAccessTokenSilently } = useAuth0()
  const apiBase = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin
  const [sessions, setSessions] = useState<ConversationSessionRecord[]>([])
  const [messages, setMessages] = useState<ConversationMessageRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [cursorStack, setCursorStack] = useState<string[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [currentCursor, setCurrentCursor] = useState<string | null>(null)
  const [humanName, setHumanName] = useState('')
  const [humanMessage, setHumanMessage] = useState('')
  const [isPageVisible, setIsPageVisible] = useState(true)

  useEffect(() => {
    if (!selectedBot?.bot_id) return
    const key = `webai_human_name_${activeOrgId || 'org'}_${selectedBot.bot_id}`
    const stored = localStorage.getItem(key)
    if (stored) setHumanName(stored)
  }, [selectedBot?.bot_id, activeOrgId])

  useEffect(() => {
    if (!selectedBot?.bot_id) return
    const key = `webai_human_name_${activeOrgId || 'org'}_${selectedBot.bot_id}`
    if (humanName) {
      localStorage.setItem(key, humanName)
    }
  }, [humanName, selectedBot?.bot_id, activeOrgId])

  useEffect(() => {
    const handleVisibility = () => {
      setIsPageVisible(document.visibilityState === 'visible')
    }
    handleVisibility()
    document.addEventListener('visibilitychange', handleVisibility)
    return () => document.removeEventListener('visibilitychange', handleVisibility)
  }, [])

  useEffect(() => {
    if (!selectedBot) return
    setSessions([])
    setMessages([])
    setSelectedSession(null)
    setCursorStack([])
    void loadSessions(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBot?.bot_id])

  const canPrev = cursorStack.length > 0
  const canNext = !!nextCursor

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
    if (diffMin <= 30) return 'Inactive'
    return 'Session ended'
  }

  async function loadSessions(cursor: string | null) {
    if (!selectedBot) return
    setLoading(true)
    try {
      const data = await listConversations(selectedBot.bot_id, 50, cursor)
      setSessions(data.sessions || [])
      setNextCursor(data.next_cursor || null)
      setCurrentCursor(cursor)
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

  useEffect(() => {
    if (!selectedBot || !selectedSession) return
    if (!isPageVisible) return
    const status = statusForSession(selectedSessionRecord)
    if (status !== 'Active') return
    let active = true
    const poll = async () => {
      if (!active) return
      const data = await getConversation(selectedBot.bot_id, selectedSession, 200)
      if (active) setMessages(data || [])
    }
    void poll()
    const timer = window.setInterval(() => {
      void poll()
    }, 4000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [selectedBot, selectedSession, getConversation, isPageVisible, selectedSessionRecord])

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

  async function sendHumanReply() {
    if (!selectedBot || !selectedSession) return
    const name = humanName.trim()
    const msg = humanMessage.trim()
    if (!msg) return
    const token = await getAccessTokenSilently()
    const orgParam = activeOrgId && activeOrgId !== '__all__' ? `?org_id=${encodeURIComponent(activeOrgId)}` : ''
    const path = `/v1/org/bots/${selectedBot.bot_id}/conversations/${encodeURIComponent(selectedSession)}/human-reply${orgParam}`
    try {
      const resp = await fetch(`${apiBase}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ sender_name: name || 'Agent', message: msg }),
      })
      if (!resp.ok) {
        let detail = resp.statusText
        try {
          const data = (await resp.json()) as { detail?: string }
          detail = data.detail || detail
        } catch {
          // ignore
        }
        throw new Error(detail)
      }
      setHumanMessage('')
      await openSession(selectedSession)
    } catch (err) {
      console.error('Failed to send human reply:', err)
    }
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
                        : statusForSession(s) === 'Inactive'
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
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
          <button
            type="button"
            className="secondary"
            disabled={!canPrev}
            onClick={() => {
              const stack = [...cursorStack]
              const prev = stack.pop() || null
              setCursorStack(stack)
              void loadSessions(prev)
            }}
          >
            Previous
          </button>
          <button
            type="button"
            className="secondary"
            disabled={!canNext}
            onClick={() => {
              if (!nextCursor) return
              if (currentCursor) setCursorStack([...cursorStack, currentCursor])
              void loadSessions(nextCursor)
            }}
          >
            Next
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
                        : statusForSession(selectedSessionRecord) === 'Inactive'
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
            {statusForSession(selectedSessionRecord) === 'Active' ? (
              <div className="conversation-reply-row">
                <input
                  className="conversation-reply-input"
                  list="orgMembers"
                  placeholder="Name"
                  value={humanName}
                  onChange={(e) => setHumanName(e.target.value)}
                />
                <span className="conversation-reply-sep">|</span>
                <input
                  className="conversation-reply-input"
                  placeholder="Type a reply..."
                  value={humanMessage}
                  onChange={(e) => setHumanMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      void sendHumanReply()
                    }
                  }}
                />
                <button type="button" className="primary" onClick={() => void sendHumanReply()}>
                  Send
                </button>
                <datalist id="orgMembers">
                  {orgMembers.map((m) => (
                    <option
                      key={m.user_id}
                      value={`${m.first_name || ''} ${m.last_name || ''}`.trim() || m.email}
                    />
                  ))}
                </datalist>
              </div>
            ) : (
              <div className="muted" style={{ marginTop: '0.75rem' }}>
                Replying is available while the session is active.
              </div>
            )}
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
