import { useAuth0 } from '@auth0/auth0-react'
import { useCallback, useMemo, useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { WIDGET_SIZE_DIMENSIONS } from '../../constants/widgetSizes'
import { useDashboardData, type SourceRecord, type DomainRecord, type AvailabilityJobRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, SectionHeader, UiButton } from '../../components/ui'

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
  widgetConfig: Record<string, unknown> | null,
  siteUrl: string,
  siteTitle: string
): string {
  const params = new URLSearchParams()
  params.set('pk', publishableKey)
  params.set('apiBase', API_BASE)
  params.set('siteUrl', siteUrl)
  params.set('siteTitle', siteTitle)
  const merged = widgetConfig && typeof widgetConfig === 'object' ? { ...widgetConfig } : {}
  for (const key of Object.keys(merged)) {
    const v = merged[key]
    if (v !== undefined && v !== null && v !== '') {
      if (key === 'suggestedMessages' && Array.isArray(v)) {
        params.set('suggestedMessages', JSON.stringify(v))
      } else {
        params.set(key, String(v))
      }
    }
  }
  return `${API_BASE}/widget/iframe.html?${params.toString()}`
}

function pickPrimaryDomain(domains: DomainRecord[], botId?: string | null): DomainRecord | null {
  if (!botId) return null
  const botDomains = domains.filter((d) => d.bot_id === botId)
  if (!botDomains.length) return null
  return botDomains.find((d) => d.status === 'verified') ?? botDomains[0]
}

function pickSourceOrigin(sources: SourceRecord[], botId?: string | null): { origin: string | null; host: string | null } {
  if (!botId) return { origin: null, host: null }
  const urlSource = sources.find(
    (s) => s.bot_id === botId && s.type === 'url' && typeof s.config?.url === 'string'
  )
  const rawUrl = typeof urlSource?.config?.url === 'string' ? urlSource.config.url.trim() : ''
  if (!rawUrl) return { origin: null, host: null }
  try {
    const parsed = new URL(rawUrl.startsWith('http') ? rawUrl : `https://${rawUrl}`)
    return { origin: parsed.origin, host: parsed.hostname }
  } catch {
    return { origin: null, host: null }
  }
}

export default function BotTestingTab() {
  const { botId } = useParams()
  const { getAccessTokenSilently } = useAuth0()
  const {
    selectedBot,
    activeOrgId,
    selectedBotWidgetConfig,
    domains,
    sources,
    startAvailabilityJob,
    listAvailabilityJobs,
    getAvailabilityJob,
    getAvailabilityRaw,
  } = useDashboardData()

  const [agentConfig, setAgentConfig] = useState<AgentConfig>({})
  const [modelId, setModelId] = useState('')
  const [instructions, setInstructions] = useState(DEFAULT_INSTRUCTIONS)
  const [temperature, setTemperature] = useState(0.2)
  const [configLoading, setConfigLoading] = useState(true)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveError, setSaveError] = useState<string | null>(null)
  const [availabilityUrl, setAvailabilityUrl] = useState('')
  const [checkIn, setCheckIn] = useState('')
  const [checkOut, setCheckOut] = useState('')
  const [adults, setAdults] = useState(2)
  const [children, setChildren] = useState(0)
  const [rooms, setRooms] = useState(1)
  const [maxSeconds, setMaxSeconds] = useState(60)
  const [availabilityQuestion, setAvailabilityQuestion] = useState('')
  const [availabilityJob, setAvailabilityJob] = useState<AvailabilityJobRecord | null>(null)
  const [availabilityJobs, setAvailabilityJobs] = useState<AvailabilityJobRecord[]>([])
  const [availabilityError, setAvailabilityError] = useState<string | null>(null)
  const availabilityPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [rawAvailabilityFormat, setRawAvailabilityFormat] = useState<'text' | 'html' | 'debug'>('text')
  const [rawAvailabilityContent, setRawAvailabilityContent] = useState<string | null>(null)
  const [rawAvailabilityLoading, setRawAvailabilityLoading] = useState(false)

  const { siteUrl, siteTitle } = useMemo(() => {
    const primaryDomain = pickPrimaryDomain(domains, selectedBot?.bot_id)
    const sourceInfo = pickSourceOrigin(sources, selectedBot?.bot_id)
    const domainHost = primaryDomain?.hostname || null
    const url = domainHost ? `https://${domainHost}` : (sourceInfo.origin || window.location.origin)
    const title =
      domainHost ||
      sourceInfo.host ||
      selectedBot?.display_name ||
      document.title ||
      'Website'
    return { siteUrl: url, siteTitle: title }
  }, [domains, sources, selectedBot?.bot_id, selectedBot?.display_name])

  // Pre-fill availability URL: prefer bookingTestUrl from Knowledge tab, else siteUrl
  useEffect(() => {
    const bookingUrl =
      selectedBotWidgetConfig && typeof selectedBotWidgetConfig === 'object'
        ? (selectedBotWidgetConfig.bookingTestUrl as string)
        : null
    const url = (typeof bookingUrl === 'string' && bookingUrl.trim()) || siteUrl
    if (url) setAvailabilityUrl((prev) => prev || url.trim())
  }, [availabilityUrl, siteUrl, selectedBotWidgetConfig])

  const widgetIframeSrc = useMemo(() => {
    if (!selectedBot?.publishable_key) return ''
    return buildWidgetIframeSrc(
      selectedBot.publishable_key,
      selectedBotWidgetConfig ?? null,
      siteUrl,
      siteTitle
    )
  }, [selectedBot?.publishable_key, selectedBotWidgetConfig, siteUrl, siteTitle])

  const widgetSize = (selectedBotWidgetConfig?.size as 'small' | 'medium' | 'large') || 'medium'
  const widgetDims = WIDGET_SIZE_DIMENSIONS[widgetSize] ?? WIDGET_SIZE_DIMENSIONS.medium

  useEffect(() => {
    function endSession() {
      if (!selectedBot) return
      const sessionKey = `webai_session_${selectedBot.publishable_key}`
      const sessionId = localStorage.getItem(sessionKey)
      if (!sessionId) return
      fetch(`${API_BASE}/v1/pk/${encodeURIComponent(selectedBot.publishable_key)}/conversations/${encodeURIComponent(sessionId)}/end`, {
        method: 'POST',
        keepalive: true,
      }).catch(() => {})
    }
    window.addEventListener('beforeunload', endSession)
    return () => window.removeEventListener('beforeunload', endSession)
  }, [selectedBot])

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

  const loadAvailabilityJobs = useCallback(async () => {
    if (!selectedBot) return
    const jobs = await listAvailabilityJobs(selectedBot.bot_id)
    setAvailabilityJobs(jobs)
    if (!availabilityJob && jobs.length > 0) {
      setAvailabilityJob(jobs[0])
    }
  }, [selectedBot, listAvailabilityJobs, availabilityJob])

  useEffect(() => {
    void loadAvailabilityJobs()
  }, [loadAvailabilityJobs])

  useEffect(() => {
    if (!selectedBot || !availabilityJob) return
    const status = (availabilityJob.status || '').toLowerCase()
    if (status === 'done' || status === 'error') return
    if (availabilityPollRef.current) window.clearInterval(availabilityPollRef.current)
    availabilityPollRef.current = window.setInterval(async () => {
      const updated = await getAvailabilityJob(selectedBot.bot_id, availabilityJob.job_id)
      if (updated) {
        setAvailabilityJob(updated)
      }
    }, 4000)
    return () => {
      if (availabilityPollRef.current) {
        window.clearInterval(availabilityPollRef.current)
        availabilityPollRef.current = null
      }
    }
  }, [selectedBot, availabilityJob, getAvailabilityJob])

  useEffect(() => {
    setRawAvailabilityContent(null)
  }, [availabilityJob?.job_id])

  const handleRunAvailability = async () => {
    if (!selectedBot || !availabilityUrl.trim()) return
    setAvailabilityError(null)
    const payload: {
      url: string
      check_in?: string
      check_out?: string
      adults?: number
      children?: number
      rooms?: number
      max_seconds?: number
      question?: string
    } = {
      url: availabilityUrl.trim(),
      max_seconds: maxSeconds,
      question: availabilityQuestion.trim() || undefined,
    }
    if (checkIn && checkOut) {
      payload.check_in = checkIn
      payload.check_out = checkOut
      payload.adults = adults
      payload.children = children
      payload.rooms = rooms
    }
    const created = await startAvailabilityJob(selectedBot.bot_id, payload)
    if (!created) {
      setAvailabilityError('Failed to start availability job')
      return
    }
    setAvailabilityJob(created)
    void loadAvailabilityJobs()
  }

  const handleLoadRawAvailability = async (format: 'text' | 'html' | 'debug') => {
    if (!selectedBot || !availabilityJob) return
    setRawAvailabilityLoading(true)
    setRawAvailabilityFormat(format)
    const raw = await getAvailabilityRaw(selectedBot.bot_id, availabilityJob.job_id, format)
    setRawAvailabilityContent(raw?.content ?? null)
    setRawAvailabilityLoading(false)
  }

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to test.</div>
  }

  return (
    <AnimatedPage className="testing-page">
      <SectionHeader
        eyebrow="Testing Lab"
        title="Configure and test your AI runtime"
        subtitle="Tune model behavior and validate hotel availability flows in one polished workspace."
      />
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
            <UiButton variant="ghost" onClick={handleReset} disabled={configLoading}>Reset</UiButton>
            <UiButton variant="primary" onClick={() => void handleSave()} disabled={configLoading}>
              {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'Saved' : 'Save'}
            </UiButton>
          </div>
        </div>

        {selectedBotWidgetConfig?.businessType === 'hotel' && (
        <div className="testing-config-card" style={{ marginTop: '1.5rem' }}>
          <h3 className="testing-config-title">Availability agent test</h3>
          <div className="testing-field">
            <label className="testing-label">Hotel URL</label>
            <input
              className="testing-input"
              type="url"
              value={availabilityUrl}
              onChange={(e) => setAvailabilityUrl(e.target.value)}
              placeholder="Paste full booking URL, or base URL + dates below (Agoda, Expedia, Booking.com, etc.)"
            />
          </div>
          <div className="testing-field" style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 140px' }}>
              <label className="testing-label">Check-in</label>
              <input
                className="testing-input"
                type="date"
                value={checkIn}
                onChange={(e) => setCheckIn(e.target.value)}
              />
            </div>
            <div style={{ flex: '1 1 140px' }}>
              <label className="testing-label">Check-out</label>
              <input
                className="testing-input"
                type="date"
                value={checkOut}
                onChange={(e) => setCheckOut(e.target.value)}
              />
            </div>
          </div>
          <div className="testing-field" style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 80px' }}>
              <label className="testing-label">Adults</label>
              <input
                className="testing-input"
                type="number"
                min={1}
                value={adults}
                onChange={(e) => setAdults(Number(e.target.value))}
              />
            </div>
            <div style={{ flex: '1 1 80px' }}>
              <label className="testing-label">Children</label>
              <input
                className="testing-input"
                type="number"
                min={0}
                value={children}
                onChange={(e) => setChildren(Number(e.target.value))}
              />
            </div>
            <div style={{ flex: '1 1 80px' }}>
              <label className="testing-label">Rooms</label>
              <input
                className="testing-input"
                type="number"
                min={1}
                value={rooms}
                onChange={(e) => setRooms(Number(e.target.value))}
              />
            </div>
            <div style={{ flex: '1 1 120px' }}>
              <label className="testing-label">Max seconds</label>
              <input
                className="testing-input"
                type="number"
                min={15}
                max={180}
                value={maxSeconds}
                onChange={(e) => setMaxSeconds(Number(e.target.value))}
              />
            </div>
          </div>
          <div className="testing-field">
            <label className="testing-label">Question</label>
            <textarea
              className="testing-textarea"
              rows={3}
              value={availabilityQuestion}
              onChange={(e) => setAvailabilityQuestion(e.target.value)}
              placeholder="e.g. Is there availability for these dates? What are the cheapest room options?"
            />
          </div>
          <div className="testing-actions">
                <UiButton variant="primary" onClick={() => void handleRunAvailability()} disabled={!availabilityUrl.trim()}>
                  Run availability check
                </UiButton>
          </div>
          {availabilityError && <div className="alert error">{availabilityError}</div>}
          {availabilityJob && (
            <div className="testing-field" style={{ marginTop: '1rem' }}>
              <div className="muted" style={{ marginBottom: '0.5rem' }}>
                Status: <strong>{availabilityJob.status}</strong>
              </div>
              {availabilityJob.question && (
                <div className="muted" style={{ marginBottom: '0.5rem' }}>
                  Question: {availabilityJob.question}
                </div>
              )}
              <div className="testing-actions" style={{ justifyContent: 'flex-start', gap: '0.5rem' }}>
                <UiButton
                  variant="ghost"
                  onClick={() => void handleLoadRawAvailability('text')}
                  disabled={rawAvailabilityLoading}
                >
                  {rawAvailabilityLoading && rawAvailabilityFormat === 'text' ? 'Loading...' : 'Load raw text'}
                </UiButton>
                <UiButton
                  variant="ghost"
                  onClick={() => void handleLoadRawAvailability('html')}
                  disabled={rawAvailabilityLoading}
                >
                  {rawAvailabilityLoading && rawAvailabilityFormat === 'html' ? 'Loading...' : 'Load raw HTML'}
                </UiButton>
                <UiButton
                  variant="ghost"
                  onClick={() => void handleLoadRawAvailability('debug')}
                  disabled={rawAvailabilityLoading}
                >
                  {rawAvailabilityLoading && rawAvailabilityFormat === 'debug' ? 'Loading...' : 'Load debug log'}
                </UiButton>
              </div>
              {availabilityJob.summary && (
                <div className="alert info" style={{ whiteSpace: 'pre-wrap' }}>
                  {availabilityJob.summary}
                </div>
              )}
              {availabilityJob.last_error && (
                <div className="alert error" style={{ whiteSpace: 'pre-wrap' }}>
                  {availabilityJob.last_error}
                </div>
              )}
              {rawAvailabilityContent && (
                <pre className="testing-raw-block">{rawAvailabilityContent}</pre>
              )}
              {availabilityJob.screenshots_dir && (
                <div className="muted" style={{ marginTop: '0.5rem' }}>
                  Screenshots: {availabilityJob.screenshots_dir}
                </div>
              )}
            </div>
          )}
          {availabilityJobs.length > 1 && (
            <div className="testing-field" style={{ marginTop: '1rem' }}>
              <label className="testing-label">Recent runs</label>
              <select
                className="testing-select"
                value={availabilityJob?.job_id || ''}
                onChange={(e) => {
                  const selected = availabilityJobs.find((j) => j.job_id === e.target.value)
                  if (selected) setAvailabilityJob(selected)
                }}
              >
                {availabilityJobs.map((j) => (
                  <option key={j.job_id} value={j.job_id}>
                    {new Date(j.created_at).toLocaleString()} — {j.status}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        )}
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
              style={{ width: widgetDims.width, height: widgetDims.height }}
            />
          ) : (
            <div
              className="testing-widget-placeholder"
              style={{ width: widgetDims.width, height: widgetDims.height }}
            >
              No publishable key for this bot.
            </div>
          )}
        </div>
        {saveError && <div className="testing-chat-error">{saveError}</div>}
      </div>
    </AnimatedPage>
  )
}
