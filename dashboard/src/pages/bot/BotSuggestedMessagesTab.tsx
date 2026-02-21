import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Bell, BellRing, Check, Loader2, MessageSquare, Save, Sparkles } from 'lucide-react'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  widgetConfigToState,
  stateToWidgetConfig,
  type WidgetDesignState,
  type SuggestedMessageConfig,
} from '../../components/WidgetDesignForm'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import { useDashboardData, type EscalationConfig } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, GlassField, SectionHeader, UiButton } from '../../components/ui'

const SAVED_FEEDBACK_MS = 2000
const DEFAULT_ESCALATION_BTN_LABEL = 'Request human support'

const DEFAULT_ESCALATION_CONFIG: EscalationConfig = {
  enabled: false,
  notify_enabled: false,
  notification_emails: '',
}

export default function BotSuggestedMessagesTab() {
  const { botId } = useParams()
  const {
    selectedBot,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    loading,
    generateSuggestedMessages,
    generatingSuggestions,
    getEscalationConfig,
    saveEscalationConfig,
  } = useDashboardData()
  const [state, setState] = useState<WidgetDesignState>(() => DEFAULT_WIDGET_DESIGN_STATE)
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)

  // Escalation settings state
  const [escalationConfig, setEscalationConfig] = useState<EscalationConfig>(DEFAULT_ESCALATION_CONFIG)
  const [escalationBtnLabel, setEscalationBtnLabel] = useState(DEFAULT_ESCALATION_BTN_LABEL)
  const [savingEscalation, setSavingEscalation] = useState(false)
  const [escalationSavedJustNow, setEscalationSavedJustNow] = useState(false)
  const [escalationToggling, setEscalationToggling] = useState(false)

  useEffect(() => {
    const next = widgetConfigToState(selectedBotWidgetConfig ?? null)
    setState(next)
    // Sync escalation button label from existing escalate message
    const existingEscalateMsg = next.suggestedMessages.find((m) => m.type === 'escalate')
    if (existingEscalateMsg) setEscalationBtnLabel(existingEscalateMsg.label || DEFAULT_ESCALATION_BTN_LABEL)
  }, [selectedBotWidgetConfig])

  useEffect(() => {
    if (!botId) return
    void getEscalationConfig(botId).then((data) => {
      if (data) setEscalationConfig(data)
    })
  }, [botId, getEscalationConfig])

  const update = useCallback(<K extends keyof WidgetDesignState>(key: K, value: WidgetDesignState[K]) => {
    setState((prev) => ({ ...prev, [key]: value }))
  }, [])

  // Only ai_response messages go into the editor; escalate is managed by the escalation section
  const aiMessages = state.suggestedMessages.filter((m) => m.type !== 'escalate')
  const escalateMessages = state.suggestedMessages.filter((m) => m.type === 'escalate')

  const handleAiMessagesChange = (next: SuggestedMessageConfig[]) => {
    update('suggestedMessages', [...next, ...escalateMessages])
  }

  const handleToggleEnabled = async (enabled: boolean) => {
    if (!botId) return
    update('suggestedMessagesEnabled', enabled)
    setSaving(true)
    try {
      const newState = { ...state, suggestedMessagesEnabled: enabled }
      await saveWidgetConfig(botId, stateToWidgetConfig(newState))
    } finally {
      setSaving(false)
    }
  }

  const handleSave = async () => {
    if (!botId || saving || savedJustNow) return
    setSaving(true)
    setSavedJustNow(false)
    try {
      await saveWidgetConfig(botId, stateToWidgetConfig(state))
      setSavedJustNow(true)
      setTimeout(() => setSavedJustNow(false), SAVED_FEEDBACK_MS)
    } finally {
      setSaving(false)
    }
  }

  const handleAutoGenerate = async () => {
    if (!botId || generatingSuggestions) return
    await generateSuggestedMessages(botId)
  }

  const handleEscalationToggle = async (enabled: boolean) => {
    if (!botId || escalationToggling) return
    setEscalationToggling(true)
    setEscalationConfig((prev) => ({ ...prev, enabled }))

    try {
      if (enabled) {
        const hasEscalateMsg = state.suggestedMessages.some((m) => m.type === 'escalate')
        if (!hasEscalateMsg) {
          const newMsg: SuggestedMessageConfig = {
            id: `suggest_escalate_${Date.now()}`,
            label: escalationBtnLabel,
            type: 'escalate',
          }
          const newMessages = [...state.suggestedMessages, newMsg]
          update('suggestedMessages', newMessages)
          await saveWidgetConfig(botId, stateToWidgetConfig({ ...state, suggestedMessages: newMessages }))
        }
        await saveEscalationConfig(botId, { ...escalationConfig, enabled: true })
      } else {
        const filtered = state.suggestedMessages.filter((m) => m.type !== 'escalate')
        update('suggestedMessages', filtered)
        await saveWidgetConfig(botId, stateToWidgetConfig({ ...state, suggestedMessages: filtered }))
        await saveEscalationConfig(botId, { ...escalationConfig, enabled: false })
      }
    } catch {
      setEscalationConfig((prev) => ({ ...prev, enabled: !enabled }))
    } finally {
      setEscalationToggling(false)
    }
  }

  const handleSaveEscalation = async () => {
    if (!botId || savingEscalation || escalationSavedJustNow) return
    setSavingEscalation(true)
    try {
      // Save escalation config
      const saved = await saveEscalationConfig(botId, escalationConfig)
      if (saved) setEscalationConfig(saved)

      // Also update the escalate message label in suggestedMessages if it changed
      if (escalationConfig.enabled) {
        const updatedMessages = state.suggestedMessages.map((m) =>
          m.type === 'escalate' ? { ...m, label: escalationBtnLabel } : m
        )
        update('suggestedMessages', updatedMessages)
        await saveWidgetConfig(botId, stateToWidgetConfig({ ...state, suggestedMessages: updatedMessages }))
      }

      setEscalationSavedJustNow(true)
      setTimeout(() => setEscalationSavedJustNow(false), SAVED_FEEDBACK_MS)
    } finally {
      setSavingEscalation(false)
    }
  }

  if (!botId) return <div className="empty-panel">Select a bot.</div>
  if (loading && !selectedBot) return <div className="empty-panel">Loading...</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">Loading...</div>

  const isGenerating = generatingSuggestions
  const isEnabled = state.suggestedMessagesEnabled

  const saveAction = (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <UiButton
        variant="secondary"
        onClick={() => void handleAutoGenerate()}
        disabled={isGenerating || saving || !isEnabled}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
      >
        {isGenerating ? (
          <>
            <Loader2 size={15} className="spin" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles size={15} />
            Auto-generate
          </>
        )}
      </UiButton>
      <UiButton
        variant="primary"
        onClick={() => void handleSave()}
        disabled={saving || savedJustNow || isGenerating || !isEnabled}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
      >
        {saving ? (
          'Saving...'
        ) : savedJustNow ? (
          <>
            <Check size={18} strokeWidth={2.5} aria-hidden />
            <span>Saved</span>
          </>
        ) : (
          <>
            <Save size={16} />
            Save
          </>
        )}
      </UiButton>
    </div>
  )

  return (
    <AnimatedPage>
      <SectionHeader
        title="Suggested messages"
        subtitle="Quick actions shown when the chat opens. Auto-generate from your trained content or add manually."
      />

      {/* Enable/Disable toggle */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
        <div>
          <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>Suggested messages</div>
          <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
            Quick actions shown when the chat opens.
          </p>
        </div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={isEnabled}
            onChange={(e) => void handleToggleEnabled(e.target.checked)}
            disabled={saving}
          />
          <span className="toggle-slider" />
        </label>
      </div>

      {isGenerating && (
        <div
          style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            background: 'var(--flow-surface-alt, #fef3f0)',
            border: '1px solid var(--flow-border, #f2d8d2)',
            borderRadius: 'var(--flow-radius, 10px)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontSize: '0.9rem',
            color: 'var(--flow-text, #1e293b)',
          }}
        >
          <Loader2 size={18} className="spin" style={{ color: 'var(--flow-primary, #e8614d)' }} />
          Generating suggested messages from your trained content...
        </div>
      )}

      <div style={{ marginTop: '1rem', opacity: isEnabled ? 1 : 0.45, pointerEvents: isEnabled ? 'auto' : 'none' }}>
        <GlassCard>
          <SuggestedMessagesEditor
            suggestedMessages={aiMessages}
            onChange={handleAiMessagesChange}
            title=""
            subtitle=""
            addButtonPlacement="bottom"
            maxItems={10}
            actions={saveAction}
          />
        </GlassCard>
      </div>

      {/* ── Section 2: Escalation Settings ── */}
      <div style={{ marginTop: '2rem' }}>
        <SectionHeader
          title="Escalation settings"
          subtitle="Let visitors request human support and notify your team by email."
        />

        <GlassCard style={{ display: 'grid', gap: '1.25rem', marginTop: '0.5rem' }}>
          {/* Enable escalations toggle */}
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
                checked={escalationConfig.enabled}
                onChange={(e) => void handleEscalationToggle(e.target.checked)}
                disabled={escalationToggling}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          {escalationConfig.enabled && (
            <>
              {/* Escalation button config */}
              <div style={{ padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.75rem' }}>
                  <MessageSquare size={18} style={{ color: 'var(--ui-flow-accent)' }} />
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>Escalation button</div>
                    <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                      A button shown in the chat that lets visitors request human support.
                    </p>
                  </div>
                </div>
                <GlassField label="Button label" style={{ marginTop: '1rem' }}>
                  <input
                    type="text"
                    placeholder="Request human support"
                    value={escalationBtnLabel}
                    onChange={(e) => setEscalationBtnLabel(e.target.value)}
                  />
                </GlassField>
              </div>

              {/* Email notifications toggle */}
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
                    checked={escalationConfig.notify_enabled}
                    onChange={(e) =>
                      setEscalationConfig((prev) => ({ ...prev, notify_enabled: e.target.checked }))
                    }
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              <GlassField label="Notification email(s)" helper="Separate multiple emails with semicolons (;).">
                <input
                  type="text"
                  placeholder="team@company.com; support@company.com"
                  value={escalationConfig.notification_emails}
                  onChange={(e) =>
                    setEscalationConfig((prev) => ({ ...prev, notification_emails: e.target.value }))
                  }
                />
              </GlassField>
            </>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <UiButton
              variant="primary"
              onClick={() => void handleSaveEscalation()}
              disabled={savingEscalation || escalationSavedJustNow}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              {savingEscalation ? (
                'Saving...'
              ) : escalationSavedJustNow ? (
                <>
                  <Check size={18} strokeWidth={2.5} aria-hidden />
                  <span>Saved</span>
                </>
              ) : (
                <>
                  <Save size={16} />
                  Save
                </>
              )}
            </UiButton>
          </div>
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
