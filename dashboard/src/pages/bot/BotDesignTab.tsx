import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { Check, Paintbrush } from 'lucide-react'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  WidgetDesignForm,
  widgetConfigToState,
  stateToWidgetConfig,
  type WidgetDesignState,
} from '../../components/WidgetDesignForm'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, SectionHeader, UiButton } from '../../components/ui'

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin
const SAVED_FEEDBACK_MS = 2000
const DEFAULT_PERSONA_ID = 'default-assistant'

type PersonaItem = {
  id: string
  name: string
  category: string
  emoji: string
}

type AgentConfig = {
  model_id?: string | null
  instructions?: string | null
  temperature?: number | null
  persona_id?: string | null
}

function withOrg(path: string, orgId: string | null): string {
  if (!orgId || orgId === '__all__') return path
  const suffix = `org_id=${encodeURIComponent(orgId)}`
  return path.includes('?') ? `${path}&${suffix}` : `${path}?${suffix}`
}

export default function BotDesignTab() {
  const { botId } = useParams()
  const { getAccessTokenSilently } = useAuth0()
  const { selectedBot, selectedBotWidgetConfig, saveWidgetConfig, loading, activeOrgId } = useDashboardData()
  const [state, setState] = useState<WidgetDesignState>(() => DEFAULT_WIDGET_DESIGN_STATE)
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)

  // Persona state
  const [personas, setPersonas] = useState<PersonaItem[]>([])
  const [personaId, setPersonaId] = useState<string>(DEFAULT_PERSONA_ID)
  const [personaSaving, setPersonaSaving] = useState(false)
  const [personaSaved, setPersonaSaved] = useState(false)

  useEffect(() => {
    setState(widgetConfigToState(selectedBotWidgetConfig ?? null))
  }, [selectedBotWidgetConfig])

  useEffect(() => {
    if (selectedBot?.display_name && state.widgetTitle === 'Chat') {
      setState((prev) => ({ ...prev, widgetTitle: selectedBot.display_name.trim() }))
    }
  }, [selectedBot?.display_name, state.widgetTitle])

  // Load personas
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/v1/personas`)
        if (res.ok) {
          const data = await res.json() as { personas: PersonaItem[] }
          if (!cancelled) setPersonas(data.personas)
        }
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [])

  // Load current persona from agent config
  useEffect(() => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    let cancelled = false
    void (async () => {
      try {
        const token = await getAccessTokenSilently()
        const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
        const res = await fetch(`${API_BASE}${path}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json() as AgentConfig
          if (!cancelled) setPersonaId((data.persona_id || DEFAULT_PERSONA_ID).trim())
        }
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [botId, activeOrgId, getAccessTokenSilently])

  useEffect(() => {
    if (!personas.length) return
    if (!personas.some((p) => p.id === personaId)) {
      setPersonaId(DEFAULT_PERSONA_ID)
    }
  }, [personas, personaId])

  const handlePersonaChange = async (newPersonaId: string) => {
    setPersonaId(newPersonaId)
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    setPersonaSaving(true)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
      // Get existing config
      const getRes = await fetch(`${API_BASE}${path}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const existing = getRes.ok ? (await getRes.json() as AgentConfig) : {}
      const payload: AgentConfig = {
        model_id: existing.model_id || undefined,
        temperature: existing.temperature ?? undefined,
        persona_id: newPersonaId,
      }
      await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload),
      })
      setPersonaSaved(true)
      setTimeout(() => setPersonaSaved(false), 2000)
    } catch { /* ignore */ }
    finally { setPersonaSaving(false) }
  }

  const update = useCallback(<K extends keyof WidgetDesignState>(key: K, value: WidgetDesignState[K]) => {
    setState((prev) => ({ ...prev, [key]: value }))
  }, [])

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

  if (!botId) {
    return <div className="empty-panel">Select a bot to edit design.</div>
  }

  if (loading && !selectedBot) {
    return <div className="empty-panel">Loading...</div>
  }

  if (selectedBot?.bot_id !== botId) {
    return <div className="empty-panel">Loading...</div>
  }

  const currentPersona =
    personas.find((p) => p.id === personaId) ||
    personas.find((p) => p.id === DEFAULT_PERSONA_ID)

  return (
    <AnimatedPage>

      <SectionHeader
        title="Design the chat widget"
        subtitle="Customize how the widget appears. Changes update the preview on the right."
      />

      {/* Persona quick selector */}
      <div className="ui-glass-card" style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
          <div>
            <h4 style={{ margin: '0 0 0.25rem 0', fontSize: '0.95rem' }}>Agent Persona</h4>
            <p className="muted" style={{ margin: 0, fontSize: '0.82rem' }}>
              {currentPersona
                ? `${currentPersona.emoji} ${currentPersona.name} — shapes how your AI communicates`
                : 'Default persona is active.'}
            </p>
          </div>
          <div className="persona-selector-compact">
            <select
              className="persona-selector-dropdown"
              value={personaId}
              onChange={(e) => void handlePersonaChange(e.target.value)}
              disabled={personaSaving}
            >
              {personas.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.emoji} {p.name}
                </option>
              ))}
            </select>
            {personaSaving && <span className="muted" style={{ fontSize: '0.8rem' }}>Saving...</span>}
            {personaSaved && <span style={{ fontSize: '0.8rem', color: '#16a34a' }}>Saved!</span>}
          </div>
        </div>
      </div>

      <WidgetDesignForm
        value={state}
        onChange={update}
        showSuggestedMessages={false}
        actions={
          <UiButton
            variant="primary"
            onClick={() => void handleSave()}
            disabled={saving || savedJustNow}
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
                <Paintbrush size={16} />
                Save design
              </>
            )}
          </UiButton>
        }
      />
    </AnimatedPage>
  )
}
