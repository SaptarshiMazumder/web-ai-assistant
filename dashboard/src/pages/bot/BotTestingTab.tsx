import { useAuth0 } from '@auth0/auth0-react'
import { useCallback, useMemo, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

const DEFAULT_MODELS = [
  { value: '', label: 'Default' },
  { value: 'gemini-2.0-flash-001', label: 'gemini-2.0-flash-001' },
  { value: 'gemini-1.5-flash', label: 'gemini-1.5-flash' },
]

const DEFAULT_INSTRUCTIONS = `## Role
You are a friendly and helpful AI chatbot who helps users with their inquiries, issues, and requests. Listen attentively, understand their needs, and assist them using the information provided. If a question is unclear, ask clarifying questions. End replies with a positive note.

## Instructions
These instructions allow you to customize the behavior, tone and personality of the agent and its responses.`

type AgentConfig = {
  model_id?: string | null
  instructions?: string | null
  temperature?: number | null
}

function withOrg(path: string, orgId: string | null): string {
  if (!orgId || orgId === '__all__') return path
  const suffix = `org_id=${encodeURIComponent(orgId)}`
  return path.includes('?') ? `${path}&${suffix}` : `${path}?${suffix}`
}

/** Build iframe URL for the real widget so Testing tab shows the same widget as on the website. */
function buildWidgetIframeSrc(
  publishableKey: string,
  widgetConfig: Record<string, unknown> | null
): string {
  const params = new URLSearchParams()
  params.set('pk', publishableKey)
  params.set('apiBase', API_BASE)
  params.set('siteUrl', window.location.origin)
  params.set('siteTitle', 'Dashboard Testing')
  const merged = widgetConfig && typeof widgetConfig === 'object' ? { ...widgetConfig } : {}
  for (const key of Object.keys(merged)) {
    const v = merged[key]
    if (v !== undefined && v !== null && v !== '') params.set(key, String(v))
  }
  return `${API_BASE}/widget/iframe.html?${params.toString()}`
}

export default function BotTestingTab() {
  const { botId } = useParams()
  const { getAccessTokenSilently } = useAuth0()
  const { selectedBot, activeOrgId, selectedBotWidgetConfig } = useDashboardData()

  const [agentConfig, setAgentConfig] = useState<AgentConfig>({})
  const [modelId, setModelId] = useState('')
  const [instructions, setInstructions] = useState(DEFAULT_INSTRUCTIONS)
  const [temperature, setTemperature] = useState(0.2)
  const [configLoading, setConfigLoading] = useState(true)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveError, setSaveError] = useState<string | null>(null)

  const widgetIframeSrc = useMemo(
    () =>
      selectedBot?.publishable_key
        ? buildWidgetIframeSrc(selectedBot.publishable_key, selectedBotWidgetConfig ?? null)
        : '',
    [selectedBot?.publishable_key, selectedBotWidgetConfig]
  )

  const loadConfig = useCallback(async () => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    setConfigLoading(true)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
      const res = await fetch(`${API_BASE}${path}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error(res.statusText)
      const data = (await res.json()) as AgentConfig
      setAgentConfig(data)
      setModelId(data.model_id ?? '')
      setInstructions((data.instructions ?? '').trim() || DEFAULT_INSTRUCTIONS)
      setTemperature(
        typeof data.temperature === 'number' && data.temperature >= 0 && data.temperature <= 1
          ? data.temperature
          : 0.2
      )
    } catch {
      setAgentConfig({})
      setModelId('')
      setInstructions(DEFAULT_INSTRUCTIONS)
      setTemperature(0.2)
    } finally {
      setConfigLoading(false)
    }
  }, [botId, activeOrgId, getAccessTokenSilently])

  useEffect(() => {
    void loadConfig()
  }, [loadConfig])

  const handleSave = async () => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    setSaveStatus('saving')
    setSaveError(null)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
      const payload: AgentConfig = {}
      if (modelId.trim()) payload.model_id = modelId.trim()
      if (instructions.trim()) payload.instructions = instructions.trim()
      if (temperature !== 0.2) payload.temperature = temperature
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error((await res.json())?.detail || res.statusText)
      setSaveStatus('saved')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch (e) {
      setSaveStatus('error')
      setSaveError(e instanceof Error ? e.message : 'Failed to save')
    }
  }

  const handleReset = () => {
    setModelId(agentConfig.model_id ?? '')
    setInstructions((agentConfig.instructions ?? '').trim() || DEFAULT_INSTRUCTIONS)
    setTemperature(
      typeof agentConfig.temperature === 'number' ? agentConfig.temperature : 0.2
    )
  }

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to test.</div>
  }

  return (
    <div className="testing-page">
      <div className="testing-left">
        <div className="testing-config-card">
          <h3 className="testing-config-title">Agent configuration</h3>

          <div className="testing-field">
            <label className="testing-label">AI Model</label>
            <select
              className="testing-select"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={configLoading}
            >
              {DEFAULT_MODELS.map((m) => (
                <option key={m.value || 'default'} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
            <p className="testing-hint">
              This AI model will be used to generate answers and perform actions by your agent.
            </p>
          </div>

          <div className="testing-field">
            <label className="testing-label">Instructions (Prompt)</label>
            <textarea
              className="testing-textarea"
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="System prompt / instructions for the agent..."
              disabled={configLoading}
              rows={10}
            />
          </div>

          <div className="testing-field">
            <label className="testing-label">Model Temperature</label>
            <div className="testing-temperature-row">
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={configLoading}
                className="testing-slider"
              />
              <span className="testing-temperature-value">{temperature}</span>
            </div>
            <p className="testing-hint">
              Control the randomness of the agent response. Lower values are more predictable,
              higher values more random.
            </p>
          </div>

          <div className="testing-actions">
            <button type="button" className="ghost" onClick={handleReset} disabled={configLoading}>
              Reset
            </button>
            <button
              type="button"
              className="primary"
              onClick={() => void handleSave()}
              disabled={configLoading}
            >
              {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'Saved' : 'Save'}
            </button>
          </div>
        </div>
      </div>

      <div className="testing-right">
        <div className="testing-widget-wrap">
          <p className="testing-widget-label">
            Same widget as on your website — colors, style, and scroll match.
          </p>
          {widgetIframeSrc ? (
            <iframe
              key={widgetIframeSrc}
              src={widgetIframeSrc}
              title="Chat widget"
              className="testing-widget-iframe"
            />
          ) : (
            <div className="testing-widget-placeholder">No publishable key for this bot.</div>
          )}
        </div>
        {saveError && <div className="testing-chat-error">{saveError}</div>}
      </div>
    </div>
  )
}
