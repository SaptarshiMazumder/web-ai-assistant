import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { Check, CheckCircle, Clock, RotateCcw, Bell, BellRing, Mail, Settings2, MessageSquare } from 'lucide-react'
import { useDashboardData, type EscalationConfig, type EscalationRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton, SegmentedTabs } from '../../components/ui'

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
    <AnimatedPage>
      <SectionHeader
        eyebrow="Support"
        title="Escalations"
        subtitle="Manage handoff rules and resolve customer requests from one timeline."
      />

      <div style={{ marginBottom: '1.5rem' }}>
        <SegmentedTabs
          value={activeTab}
          onChange={(val) => {
            setActiveTab(val)
            if (val === 'escalations') void loadEscalations()
          }}
          options={[
            { id: 'settings', label: 'Settings', icon: <Settings2 size={16} /> },
            { id: 'escalations', label: 'Escalated conversations', icon: <MessageSquare size={16} /> },
          ]}
          ariaLabel="Escalation tabs"
        />
      </div>

      {activeTab === 'settings' && (
        <GlassCard style={{ display: 'grid', gap: '1.25rem', marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <Bell size={18} style={{ color: 'var(--ui-flow-accent)' }} />
              <div>
                <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>Enable escalations</div>
                <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                  Allow visitors to escalate to support with their email.
                </p>
              </div>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={config.enabled}
                onChange={(e) => setConfig((prev) => ({ ...prev, enabled: e.target.checked }))}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <BellRing size={18} style={{ color: 'var(--ui-flow-accent)' }} />
              <div>
                <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>Email notifications</div>
                <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                  Receive notifications when visitors escalate.
                </p>
              </div>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={config.notify_enabled}
                onChange={(e) => setConfig((prev) => ({ ...prev, notify_enabled: e.target.checked }))}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 500, fontSize: '0.9rem', marginBottom: '0.4rem', color: 'var(--ui-flow-text)' }}>
              <Mail size={15} style={{ color: 'var(--ui-flow-accent)' }} />
              Notification email(s)
            </label>
            <input
              type="text"
              placeholder="team@company.com; support@company.com"
              value={config.notification_emails}
              onChange={(e) => setConfig((prev) => ({ ...prev, notification_emails: e.target.value }))}
              style={{ width: '100%' }}
            />
            <span className="muted" style={{ fontSize: '0.82rem', display: 'block', marginTop: '0.3rem' }}>
              Separate multiple emails with semicolons (;).
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <UiButton variant="primary" onClick={() => void handleSave()} disabled={saving}>
              {saving ? 'Saving...' : 'Save settings'}
            </UiButton>
          </div>
        </GlassCard>
      )}

      {activeTab === 'escalations' && (
        <GlassCard style={{ marginTop: '0.5rem' }}>
          {escalations.length === 0 && <div className="muted" style={{ padding: '1rem 0' }}>No escalations yet.</div>}
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
      )}
    </AnimatedPage >
  )
}
