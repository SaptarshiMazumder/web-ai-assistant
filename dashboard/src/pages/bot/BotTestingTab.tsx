import { useAuth0 } from '@auth0/auth0-react'
import { useCallback, useMemo, useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { WIDGET_SIZE_DIMENSIONS } from '../../constants/widgetSizes'
import { useDashboardData, type SourceRecord, type DomainRecord, type AvailabilityJobRecord } from '../../hooks/useDashboardData'
import { AnimatedPage, SectionHeader, UiButton } from '../../components/ui'

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin
const DEFAULT_PERSONA_ID = 'default-assistant'
const LEGACY_DEFAULT_INSTRUCTIONS = `## Role
You are a friendly and helpful AI chatbot who helps users with their inquiries, issues, and requests. Listen attentively, understand their needs, and assist them using the information provided. If a question is unclear, ask clarifying questions. End replies with a positive note.

## Instructions
These instructions allow you to customize the behavior, tone and personality of the agent and its responses.`
const VANILLA_PROMPT_EN = `## Personality
You are a helpful, clear, and professional AI assistant for this business.

## About the Business
You represent the business and assist visitors with their questions and needs.

## Response Rules
- MANDATORY: Detect the language of the user's input and respond in that same language.
- Use bullet points when listing multiple items that belong to the same category.
- You may use markdown bold (**text**) to emphasize important items sparingly.
- Provide detailed, helpful explanations.
- If you do not know something, say so and suggest checking the website.
- End responses on a positive, welcoming note.`
const VANILLA_PROMPT_JA = `## パーソナリティ
あなたはこのビジネスのAIアシスタントです。丁寧でわかりやすく、親しみのある口調で案内してください。

## ビジネスについて
あなたはこのビジネスの代表として、訪問者の質問やニーズをサポートします。

## 応答ルール
- 必須：ユーザーの入力言語を検出し、同じ言語で応答してください。
- 同じカテゴリに属する複数の項目を列挙する場合は箇条書きを使ってください。
- 重要な箇所の強調には太字（**text**）を必要最小限で使えます。
- 具体的で役立つ説明を提供してください。
- 不明な点は正直に伝え、必要に応じてサイト確認を提案してください。
- 最後は前向きで歓迎的な一言で締めてください。`

const DEFAULT_MODELS = [
  { value: '', label: 'Default' },
  { value: 'gemini-2.0-flash-001', label: 'gemini-2.0-flash-001' },
  { value: 'gemini-1.5-flash', label: 'gemini-1.5-flash' },
]

type CustomPersonaConfig = {
  id?: string
  name?: string
  category?: string
  emoji?: string
  system_prompt?: string
  description?: string
  [key: string]: unknown
}

type AgentConfig = {
  model_id?: string | null
  instructions?: string | null
  temperature?: number | null
  persona_id?: string | null
  custom_personas?: CustomPersonaConfig[] | null
}

type PersonaItem = {
  id: string
  name: string
  category: string
  emoji: string
  system_prompt: string
}

function normalizePersona(input: CustomPersonaConfig | null | undefined): PersonaItem | null {
  const id = String(input?.id || '').trim()
  const systemPrompt = String(input?.system_prompt || '').trim()
  if (!id || !systemPrompt) return null
  return {
    id,
    name: String(input?.name || 'Custom Persona').trim() || 'Custom Persona',
    category: String(input?.category || 'Custom').trim() || 'Custom',
    emoji: String(input?.emoji || '💬').trim() || '💬',
    system_prompt: systemPrompt,
  }
}

function normalizePromptText(value: string): string {
  return value.replace(/\r\n/g, '\n').trim()
}

function isLegacyDefaultInstructions(value: string): boolean {
  return normalizePromptText(value) === normalizePromptText(LEGACY_DEFAULT_INSTRUCTIONS)
}

function getVanillaPromptForLanguage(lang: string): string {
  return lang === 'ja' ? VANILLA_PROMPT_JA : VANILLA_PROMPT_EN
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
  const { t, i18n } = useTranslation()
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

  // Derive bot content language from widget config, fallback to dashboard UI language
  const botLanguage = (() => {
    const lang = selectedBotWidgetConfig?.language
    if (lang === 'ja' || lang === 'jp') return 'ja'
    if (lang === 'en') return 'en'
    // No explicit bot language set — use dashboard UI language as fallback
    return i18n.language?.startsWith('ja') ? 'ja' : 'en'
  })()

  const [agentConfig, setAgentConfig] = useState<AgentConfig>({})
  const [modelId, setModelId] = useState('')
  const [instructions, setInstructions] = useState(() => getVanillaPromptForLanguage(botLanguage))
  const [instructionsOverridden, setInstructionsOverridden] = useState(false)
  const [temperature, setTemperature] = useState(0.2)
  const [personaId, setPersonaId] = useState<string>(DEFAULT_PERSONA_ID)
  const [personas, setPersonas] = useState<PersonaItem[]>([]) // built-in personas
  const [customPersonas, setCustomPersonas] = useState<CustomPersonaConfig[]>([])
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

  const [isGenerating, setIsGenerating] = useState(false)

  const languageFallbackPrompt = useMemo(
    () => getVanillaPromptForLanguage(botLanguage),
    [botLanguage]
  )

  const allPersonas = useMemo(() => {
    const normalizedCustom = customPersonas
      .map((p) => normalizePersona(p))
      .filter((p): p is PersonaItem => p !== null)
    const fallbackDefaultPersona: PersonaItem = {
      id: DEFAULT_PERSONA_ID,
      name: botLanguage === 'ja' ? 'デフォルト' : 'Default',
      category: botLanguage === 'ja' ? 'プロフェッショナル' : 'Professional',
      emoji: '💬',
      system_prompt: languageFallbackPrompt,
    }
    const builtin = personas.length > 0 ? personas : [fallbackDefaultPersona]
    return [...normalizedCustom, ...builtin]
  }, [customPersonas, personas, botLanguage, languageFallbackPrompt])

  const selectedPersona = useMemo(
    () => allPersonas.find((p) => p.id === personaId) ?? null,
    [allPersonas, personaId]
  )

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
      }).catch(() => { })
    }
    window.addEventListener('beforeunload', endSession)
    return () => window.removeEventListener('beforeunload', endSession)
  }, [selectedBot])

  // Load personas catalog (re-fetch when bot language changes)
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/v1/personas?lang=${botLanguage}`)
        if (res.ok) {
          const data = await res.json() as { personas: PersonaItem[] }
          if (!cancelled) setPersonas(data.personas)
        }
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [botLanguage])

  // When persona is selected, populate the instructions with its system prompt
  const handlePersonaSelect = useCallback((selectedId: string) => {
    setPersonaId(selectedId)
    const persona = allPersonas.find((p) => p.id === selectedId)
    if (persona) {
      setInstructionsOverridden(false)
      setInstructions(persona.system_prompt)
    }
  }, [allPersonas])

  useEffect(() => {
    if (!allPersonas.length) return
    if (!allPersonas.some((p) => p.id === personaId)) {
      setPersonaId(DEFAULT_PERSONA_ID)
    }
  }, [allPersonas, personaId])

  // Keep instructions synced to persona when no explicit override is stored.
  useEffect(() => {
    if (instructionsOverridden) return
    if (selectedPersona?.system_prompt?.trim()) {
      setInstructions(selectedPersona.system_prompt)
      return
    }
    setInstructions((prev) => prev.trim() || languageFallbackPrompt)
  }, [instructionsOverridden, selectedPersona, languageFallbackPrompt])

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
      const persistedInstructions = (data.instructions ?? '').trim()
      const effectiveInstructions = isLegacyDefaultInstructions(persistedInstructions)
        ? ''
        : persistedInstructions
      setAgentConfig(data)
      setModelId(data.model_id ?? '')
      setInstructions(effectiveInstructions || languageFallbackPrompt)
      setInstructionsOverridden(Boolean(effectiveInstructions))
      setTemperature(
        typeof data.temperature === 'number' && data.temperature >= 0 && data.temperature <= 1
          ? data.temperature
          : 0.2
      )
      setPersonaId((data.persona_id || DEFAULT_PERSONA_ID).trim())
      setCustomPersonas(Array.isArray(data.custom_personas) ? data.custom_personas : [])
    } catch {
      setAgentConfig({})
      setModelId('')
      setInstructions(languageFallbackPrompt)
      setInstructionsOverridden(false)
      setTemperature(0.2)
      setPersonaId(DEFAULT_PERSONA_ID)
      setCustomPersonas([])
    } finally {
      setConfigLoading(false)
    }
  }, [botId, activeOrgId, getAccessTokenSilently, languageFallbackPrompt])

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
      if (instructionsOverridden && instructions.trim()) payload.instructions = instructions.trim()
      if (temperature !== 0.2) payload.temperature = temperature
      payload.persona_id = personaId || DEFAULT_PERSONA_ID
      if (customPersonas.length > 0) payload.custom_personas = customPersonas
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error((await res.json())?.detail || res.statusText)
      setAgentConfig(payload)
      setSaveStatus('saved')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch (e) {
      setSaveStatus('error')
      setSaveError(e instanceof Error ? e.message : 'Failed to save')
    }
  }

  const handleReset = () => {
    const resetPersonaId = (agentConfig.persona_id || DEFAULT_PERSONA_ID).trim()
    const resetInstructions = (agentConfig.instructions ?? '').trim()
    const effectiveResetInstructions = isLegacyDefaultInstructions(resetInstructions)
      ? ''
      : resetInstructions
    const resetPersonaPrompt = allPersonas.find((p) => p.id === resetPersonaId)?.system_prompt || languageFallbackPrompt
    setModelId(agentConfig.model_id ?? '')
    setInstructions(effectiveResetInstructions || resetPersonaPrompt)
    setInstructionsOverridden(Boolean(effectiveResetInstructions))
    setTemperature(
      typeof agentConfig.temperature === 'number' ? agentConfig.temperature : 0.2
    )
    setPersonaId(resetPersonaId)
  }

  const handleGeneratePrompt = async () => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    setIsGenerating(true)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/generate-default-prompt?lang=${encodeURIComponent(botLanguage)}`, activeOrgId)
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || res.statusText)
      }
      const data = await res.json() as { prompt: string }
      if (data.prompt) {
        setInstructions(data.prompt)
        setInstructionsOverridden(true)
      }
    } catch (e) {
      setInstructions((prev) => prev.trim() || selectedPersona?.system_prompt || languageFallbackPrompt)
      setInstructionsOverridden(false)
      alert("Failed to generate prompt: " + (e instanceof Error ? e.message : String(e)))
    } finally {
      setIsGenerating(false)
    }
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
    return <div className="empty-panel">{t('botTesting.selectBotToTest', 'Select a bot to test.')}</div>
  }

  return (
    <AnimatedPage>
      <div className="testing-page">
        <div style={{ gridColumn: '1 / -1' }}>
          <SectionHeader
            eyebrow={t('botTesting.title', 'Testing Lab')}
            title={t('botTesting.subtitle', 'Configure and test your AI runtime')}
            subtitle={t('botTesting.description', 'Tune model behavior and validate hotel availability flows in one polished workspace.')}
          />
        </div>
        <div className="testing-left">
          <div className="testing-config-card ui-glass-card">
            <h3 className="testing-config-title">{t('botTesting.agentConfiguration', 'Agent configuration')}</h3>

            <div className="testing-field">
              <label className="testing-label">{t('botTesting.aiModel', 'AI Model')}</label>
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
                {t('botTesting.aiModelHint', 'This AI model will be used to generate answers and perform actions by your agent.')}
              </p>
            </div>

            <div className="testing-field">
              <label className="testing-label">{t('botTesting.activePersona', 'Active Persona')}</label>
              <div className="persona-selector-compact">
                <select
                  className="persona-selector-dropdown"
                  value={personaId}
                  onChange={(e) => handlePersonaSelect(e.target.value)}
                  disabled={configLoading}
                >
                  {allPersonas.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.emoji} {p.name} — {p.category}
                    </option>
                  ))}
                </select>
              </div>
              <p className="testing-hint">
                {t('botTesting.activePersonaHint', 'Select a persona to shape the AI\'s tone and personality.')}
              </p>
            </div>

            <div className="testing-field">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <label className="testing-label">{t('botTesting.instructions', 'Instructions (Prompt)')}</label>
                {personaId === DEFAULT_PERSONA_ID && (
                  <UiButton
                    variant="ghost"
                    onClick={() => void handleGeneratePrompt()}
                    disabled={isGenerating || configLoading}
                    style={{ fontSize: '0.75rem', padding: '2px 8px', height: 'auto', minHeight: 'unset' }}
                  >
                    {isGenerating ? t('botTesting.generating', 'Generating...') : t('botTesting.generateFromWebsite', '⚡ Generate from website')}
                  </UiButton>
                )}
                {personaId && selectedPersona && personaId !== DEFAULT_PERSONA_ID && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--ui-flow-accent, #e4587a)', fontWeight: 500 }}>
                    {t('botTesting.basedOn', 'Based on {{name}}', { name: selectedPersona.name })}
                  </span>
                )}
              </div>
              <textarea
                className="testing-textarea"
                value={instructions}
                onChange={(e) => {
                  setInstructions(e.target.value)
                  setInstructionsOverridden(true)
                }}
                placeholder={t('botTesting.instructionsPlaceholder', 'System prompt / instructions for the agent...')}
                disabled={configLoading}
                rows={10}
              />
              <p className="testing-hint">
                {t('botTesting.instructionsHint', 'The above starts from your selected persona. Edit it to customize further.')}
              </p>
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
                {t('botTesting.temperatureHint', 'Control the randomness of the agent response. Lower values are more predictable, higher values more random.')}
              </p>
            </div>

            <div className="testing-actions">
              <UiButton variant="ghost" onClick={handleReset} disabled={configLoading}>{t('botTesting.reset', 'Reset')}</UiButton>
              <UiButton variant="primary" onClick={() => void handleSave()} disabled={configLoading}>
                {saveStatus === 'saving' ? t('botTesting.saving', 'Saving...') : saveStatus === 'saved' ? t('botTesting.saved', 'Saved') : t('botTesting.save', 'Save')}
              </UiButton>
            </div>
          </div>

          {/* Hotel business type features removed, content hidden or removed as per user goal */}
          {false && (
            <div className="testing-config-card ui-glass-card" style={{ marginTop: '1.5rem' }}>
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
                    Status: <strong>{availabilityJob?.status}</strong>
                  </div>
                  {availabilityJob?.question && (
                    <div className="muted" style={{ marginBottom: '0.5rem' }}>
                      Question: {availabilityJob?.question}
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
                  {availabilityJob?.summary && (
                    <div className="alert info" style={{ whiteSpace: 'pre-wrap' }}>
                      {availabilityJob?.summary}
                    </div>
                  )}
                  {availabilityJob?.last_error && (
                    <div className="alert error" style={{ whiteSpace: 'pre-wrap' }}>
                      {availabilityJob?.last_error}
                    </div>
                  )}
                  {rawAvailabilityContent && (
                    <pre className="testing-raw-block">{rawAvailabilityContent}</pre>
                  )}
                  {availabilityJob?.screenshots_dir && (
                    <div className="muted" style={{ marginTop: '0.5rem' }}>
                      Screenshots: {availabilityJob?.screenshots_dir}
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
            {widgetIframeSrc ? (
              <iframe
                key={widgetIframeSrc}
                src={widgetIframeSrc}
                title="Chat widget"
                className="testing-widget-iframe"
                style={{ width: '100%', maxWidth: widgetDims.width, height: widgetDims.height }}
              />
            ) : (
              <div
                className="testing-widget-placeholder"
                style={{ width: '100%', maxWidth: widgetDims.width, height: widgetDims.height }}
              >
                {t('botTesting.noPublishableKey', 'No publishable key for this bot.')}
              </div>
            )}
          </div>
          {saveError && <div className="testing-chat-error">{saveError}</div>}
        </div>
      </div>
    </AnimatedPage>
  )
}
