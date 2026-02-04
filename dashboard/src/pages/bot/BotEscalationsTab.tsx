import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { Check, CheckCircle, Clock, RotateCcw } from 'lucide-react'
import { useDashboardData, type EscalationConfig, type EscalationRecord } from '../../hooks/useDashboardData'

const DEFAULT_CONFIG: EscalationConfig = {
  enabled: false,
  notify_enabled: false,
  notification_emails: '',
}

export default function BotEscalationsTab() {
  const { botId } = useParams()
  const navigate = useNavigate()
  const {
    selectedBot,
    getEscalationConfig,
    saveEscalationConfig,
    listEscalations,
    updateEscalationStatus,
  } = useDashboardData()
  const [activeTab, setActiveTab] = useState<'settings' | 'escalations'>('settings')
  const [config, setConfig] = useState<EscalationConfig>(DEFAULT_CONFIG)
  const [saving, setSaving] = useState(false)
  const [escalations, setEscalations] = useState<EscalationRecord[]>([])
  const [searchParams] = useSearchParams()
  const pageSize = 200

  useEffect(() => {
    if (!selectedBot?.bot_id) return
    void (async () => {
      const data = await getEscalationConfig(selectedBot.bot_id)
      if (data) setConfig(data)
    })()
  }, [selectedBot?.bot_id, getEscalationConfig])

  useEffect(() => {
    if (!selectedBot?.bot_id) return
    setEscalations([])
    if (activeTab === 'escalations') {
      void loadEscalations()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBot?.bot_id, activeTab])

  useEffect(() => {
    const tab = searchParams.get('tab')
    if (tab === 'escalations' || tab === 'settings') {
      setActiveTab(tab)
    }
  }, [searchParams])

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

  async function handleSave() {
    if (!selectedBot || saving) return
    setSaving(true)
    try {
      const saved = await saveEscalationConfig(selectedBot.bot_id, config)
      if (saved) setConfig(saved)
    } finally {
      setSaving(false)
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
    return <div className="empty-panel">Select a bot to manage escalations.</div>
  }

  return (
    <div className="card">
      <div className="card-title">Escalations</div>
      <div className="tab-row" style={{ marginTop: '0.75rem' }}>
        <button
          type="button"
          className={`tab-button ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
        <button
          type="button"
          className={`tab-button ${activeTab === 'escalations' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('escalations')
            void loadEscalations()
          }}
        >
          Escalated conversations
        </button>
      </div>

      {activeTab === 'settings' && (
        <div style={{ marginTop: '1rem', display: 'grid', gap: '1rem' }}>
          <div className="card">
            <div className="card-title" style={{ fontSize: '1rem' }}>Enable escalations</div>
            <p className="card-subtitle" style={{ marginTop: '0.25rem' }}>
              Allow visitors to escalate to support. Visitors will be prompted to enter their email.
            </p>
            <label className="toggle">
              <input
                type="checkbox"
                checked={config.enabled}
                onChange={(e) => setConfig((prev) => ({ ...prev, enabled: e.target.checked }))}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          <div className="card">
            <div className="card-title" style={{ fontSize: '1rem' }}>Enable email notifications</div>
            <p className="card-subtitle" style={{ marginTop: '0.25rem' }}>
              Receive an email notification when a visitor escalates.
            </p>
            <label className="toggle">
              <input
                type="checkbox"
                checked={config.notify_enabled}
                onChange={(e) => setConfig((prev) => ({ ...prev, notify_enabled: e.target.checked }))}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          <div className="design-form-field design-form-field-full">
            <label className="design-form-label">Escalation notification email(s)</label>
            <input
              type="text"
              className="design-form-input"
              placeholder="team@company.com; support@company.com"
              value={config.notification_emails}
              onChange={(e) => setConfig((prev) => ({ ...prev, notification_emails: e.target.value }))}
            />
            <span className="design-form-hint">
              Separate multiple emails with semicolons (;).
            </span>
          </div>

          <div className="flow-actions">
            <button type="button" className="primary" onClick={() => void handleSave()} disabled={saving}>
              {saving ? 'Saving...' : 'Save settings'}
            </button>
          </div>
        </div>
      )}

      {activeTab === 'escalations' && (
        <div style={{ marginTop: '1rem' }}>
          {escalations.length === 0 && <div className="muted">No escalations yet.</div>}
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
        </div>
      )}
    </div>
  )
}
