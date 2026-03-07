import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export type BotSummary = {
  bot_id: string
  org_id?: string
  display_name: string
  publishable_key: string
  secret_key: string
  created_at: string
  updated_at: string
}

export type BotCreateResponse = {
  bot_id: string
  display_name: string
  publishable_key: string
  secret_key: string
}

export type DomainRecord = {
  bot_id: string
  hostname: string
  status: string
  verification_token: string
  verification_url?: string
  verified_at?: string | null
  created_at: string
  updated_at: string
}

export type JobRecord = {
  job_id: string
  url: string
  hostname: string
  stage: string
  pages_crawled: number
  docs_count: number
  gcs_prefix: string
  last_error: string
  created_at: string
  updated_at: string
  /** URLs discovered and indexed by this crawl job */
  crawled_urls?: string[]
  source_id?: string | null
}

export type SourceRecord = {
  source_id: string
  bot_id: string
  type: string
  config: Record<string, unknown>
  display_name?: string | null
  created_at: string
  updated_at: string
  sync_enabled?: boolean
  sync_frequency?: string
  sync_time_utc?: string
  sync_timezone?: string
  last_synced_at?: string | null
}

export type SyncSettings = {
  sync_enabled: boolean
  sync_frequency: string
  sync_time_utc: string
  sync_timezone: string
}

export type PdfSourceUploadItem = {
  source: SourceRecord
  job_id: string
  status: string
}

export type PdfSourceUploadResponse = {
  bot_id: string
  items: PdfSourceUploadItem[]
}

export type TextSourceEntry = {
  title?: string
  content: string
}

export type TextSourceUploadItem = {
  source_id: string
  job_id: string
  status: string
}

export type TextSourceUploadResponse = {
  bot_id: string
  items: TextSourceUploadItem[]
}

export type DocsSourceUploadItem = {
  source_id: string
  job_id: string
  status: string
}

export type DocsSourceUploadResponse = {
  bot_id: string
  items: DocsSourceUploadItem[]
}

/** Optional widget config for embed snippet (create-bot flow or custom embed). */
export type EmbedSnippetConfig = {
  position?: string
  primaryColor?: string
  title?: string
  size?: string
  placeholder?: string
  footerMessage?: string
  theme?: string
  textColor?: string
  launcherIconUrl?: string
  launcherText?: string
  headerIconUrl?: string
  shareIconUrl?: string
  maxHeight?: number
  fontSize?: string
  headerSize?: string
  autoPopupWelcome?: string
  autoScrollNewMessages?: boolean
  displaySourcesInMessages?: boolean
  sourcesLabel?: string
}

export type IndexStatus = {
  status: string
  stage?: string
  pages_crawled?: number
  last_crawled_url?: string
  docs_count?: number
  last_error?: string
  gcs_prefix?: string
  updated_at?: string
}

export type DiscoveryJobRecord = {
  job_id: string
  bot_id: string
  root_url: string
  method: string
  status: string
  discovered_urls: string[]
  discovered_count: number
  error?: string | null
  created_at: string
  updated_at: string
}

export type TopicJobRecord = {
  job_id: string
  org_id: string
  bot_id: string
  status: string
  stage: string
  gcs_prefix?: string | null
  docs_count: number
  topics_count: number
  last_error?: string | null
  created_at: string
  updated_at: string
}

export type AvailabilityJobRecord = {
  job_id: string
  org_id: string
  bot_id: string
  url: string
  status: string
  question?: string | null
  summary?: string | null
  raw_text_path?: string | null
  raw_html_path?: string | null
  last_error?: string | null
  max_seconds: number
  steps_count: number
  screenshots_dir?: string | null
  created_at: string
  updated_at: string
}

export type BookingLinkJobRecord = {
  job_id: string
  bot_id: string
  index_job_id?: string | null
  root_url: string
  status: string
  links: Record<string, unknown>[]
  error?: string | null
  created_at: string
  updated_at: string
}

export type JobPipelineStepRecord = {
  run_id: string
  step_index: number
  job_id: string
  runner_ref: string
  on_failure: string
  status: string
  progress_pct: number
  current_stage_key?: string | null
  current_message?: string | null
  attempt: number
  celery_task_id?: string | null
  linked_job_type?: string | null
  linked_job_id?: string | null
  output: Record<string, unknown>
  last_error?: string | null
  started_at?: string | null
  completed_at?: string | null
  created_at: string
  updated_at: string
}

export type JobPipelineEventRecord = {
  event_id: string
  run_id: string
  step_index?: number | null
  event_type: string
  stage_key?: string | null
  message?: string | null
  progress_pct?: number | null
  details: Record<string, unknown>
  created_at: string
}

export type JobPipelineRunRecord = {
  run_id: string
  org_id: string
  bot_id: string
  workflow_id: string
  trigger: string
  status: string
  current_step_index: number
  progress_pct: number
  current_step_id?: string | null
  current_stage_key?: string | null
  current_message?: string | null
  context: Record<string, unknown>
  last_error?: string | null
  created_at: string
  updated_at: string
  steps: JobPipelineStepRecord[]
  events: JobPipelineEventRecord[]
}

export type PlatformConfigJobPipelineWorkflow = {
  workflowId: string
  default: string[]
  platformOverrides: Record<string, string[]>
}

export type PlatformConfigPayload = {
  platforms: Array<{
    id: string
    widget_key: string
    domain_key: string
    label: string
    url_placeholder?: string
    availableSuggestedMessageTypes?: string[]
  }>
  defaultSuggestedMessages: Array<{ id: string; label: string; type: string; prompt?: string }>
  defaultAvailableSuggestedMessageTypes: string[]
  jobPipelineWorkflow: PlatformConfigJobPipelineWorkflow
}

export type ConversationSessionRecord = {
  session_id: string
  bot_id: string
  channel: string
  status: string
  title?: string | null
  site_url?: string | null
  site_title?: string | null
  message_count: number
  started_at: string
  last_active_at: string
  ended_at?: string | null
}

export type ConversationCitation = {
  url: string
  snippet: string
}

export type ConversationMessageRecord = {
  message_id: string
  session_id: string
  bot_id: string
  role: string
  sender_name?: string | null
  content: string
  citations?: ConversationCitation[]
  created_at: string
}

export type EscalationConfig = {
  enabled: boolean
  notify_enabled: boolean
  notify_website: boolean
  notify_instagram: boolean
  notify_line: boolean
  notification_emails: string
}

export type EscalationRecord = {
  escalation_id: string
  bot_id: string
  session_id: string
  visitor_email: string
  details?: string | null
  status: string
  created_at: string
  title?: string | null
  site_url?: string | null
  site_title?: string | null
  last_active_at?: string | null
  session_status?: string | null
}

export type AnalyticsSummary = {
  start_day: string
  end_day: string
  conversations: number
  messages_user: number
  messages_bot: number
  escalations: number
  unique_visitors_est: number
  messages_per_conversation: number
  escalation_rate: number
  positive_feedback: number
  negative_feedback: number
}

export type UsagePoint = {
  day: string
  conversations: number
  messages_user: number
  messages_bot: number
  escalations: number
  unique_visitors_est: number
}

export type AnalyticsTimeseries = {
  start_day: string
  end_day: string
  points: UsagePoint[]
}

export type TopSourceItem = { source_url: string; count: number }
export type TopSources = { start_day: string; end_day: string; items: TopSourceItem[] }

export type TopicItem = { topic: string; count: number }
export type Topics = { start_day: string; end_day: string; items: TopicItem[] }

export type ExtractedTopic = {
  topic_id: string
  topic: string
  category?: string | null
  confidence: number
  source_urls: string[]
  source_url?: string | null
  origin?: string
  occurrence_count: number
  is_active: boolean
  extracted_at?: string | null
  updated_at?: string | null
}

export type ExtractedTopicsResponse = {
  bot_id: string
  topics: ExtractedTopic[]
  total_count: number
}

export type TopicUsageItem = {
  topic_id: string
  topic: string
  category: string
  question_count: number
  source_url?: string | null
  origin?: string
}

export type TopicUsageSummary = {
  bot_id: string
  topics: TopicUsageItem[]
  total_questions: number
}

export type TopicQuestionItem = {
  id: string
  topic_id: string
  session_id: string
  message_id?: string | null
  question_text?: string | null
  asked_at: string
  session_title?: string | null
}

export type TopicQuestionsResponse = {
  topic_id: string
  questions: TopicQuestionItem[]
}

export type ConversationSearchSessionRecord = ConversationSessionRecord & { snippet?: string | null }

export type OrgSummary = {
  org_id: string
  name: string
  status: string
  created_at: string
  updated_at: string
}

export type OrgMember = {
  user_id: string
  email: string
  first_name?: string | null
  last_name?: string | null
  role: string
  org_id?: string
  org_name?: string
  created_at: string
  updated_at: string
}

type TokenClaims = Record<string, unknown>

type DashboardData = {
  user:
  | {
    email?: string | null
    name?: string | null
    given_name?: string | null
    family_name?: string | null
    picture?: string | null
  }
  | undefined
  logout: (options?: { logoutParams?: { returnTo?: string } }) => void
  bots: BotSummary[]
  selectedBotId: string | null
  selectedBot: BotSummary | null
  domains: DomainRecord[]
  jobs: JobRecord[]
  sources: SourceRecord[]
  indexStatus: IndexStatus | null
  loading: boolean
  botsLoadedOnce: boolean
  error: string | null
  setError: (value: string | null) => void
  newBotName: string
  setNewBotName: (value: string) => void
  newDomain: string
  setNewDomain: (value: string) => void
  crawlUrl: string
  setCrawlUrl: (value: string) => void
  activeCrawlUrl: string
  orgs: OrgSummary[]
  activeOrgId: string | null
  setActiveOrgId: (value: string) => void
  orgMembers: OrgMember[]
  newOrgName: string
  setNewOrgName: (value: string) => void
  orgDisplayName: string
  orgDisplayNameInput: string
  setOrgDisplayNameInput: (value: string) => void
  newMemberEmail: string
  setNewMemberEmail: (value: string) => void
  newMemberRole: string
  setNewMemberRole: (value: string) => void
  isSuperAdmin: boolean
  embedSnippet: string
  buildEmbedSnippet: (config?: EmbedSnippetConfig) => string
  selectedBotWidgetConfig: Record<string, unknown> | null
  saveWidgetConfig: (botId: string, config: Record<string, unknown>) => Promise<void>
  loadBots: () => Promise<void>
  loadBotDetail: (botId: string) => Promise<void>
  loadDomains: (botId: string) => Promise<void>
  loadJobs: (botId: string) => Promise<void>
  loadSources: (botId: string) => Promise<void>
  createSource: (botId: string, type: string, config: Record<string, unknown>, displayName?: string | null) => Promise<SourceRecord | null>
  uploadPdfSources: (botId: string, files: File[], displayName?: string | null) => Promise<PdfSourceUploadResponse | null>
  uploadTextSources: (botId: string, entries: TextSourceEntry[]) => Promise<TextSourceUploadResponse | null>
  uploadDocsSources: (botId: string, files: File[]) => Promise<DocsSourceUploadResponse | null>
  deleteSource: (botId: string, sourceId: string) => Promise<void>
  createBot: (displayName?: string, orgIdOverride?: string | null) => Promise<BotCreateResponse | null>
  loadOrgs: () => Promise<void>
  loadSelfOrgs: () => Promise<void>
  loadOrgInfo: (orgId: string) => Promise<void>
  saveOrgName: () => Promise<void>
  createOrg: () => Promise<void>
  setOrgStatus: (orgId: string, status: 'active' | 'disabled') => Promise<void>
  loadOrgMembers: (orgId: string) => Promise<void>
  addOrgMember: () => Promise<void>
  clearGcs: () => Promise<void>
  clearRag: () => Promise<void>
  addDomain: () => Promise<void>
  verifyDomain: (hostname: string) => Promise<void>
  startCrawl: () => Promise<void>
  startCrawlForSource: (botId: string, sourceId: string) => Promise<void>
  syncSource: (botId: string, sourceId: string) => Promise<void>
  updateSourceSyncSettings: (botId: string, sourceId: string, settings: SyncSettings) => Promise<SourceRecord | null>
  queueCrawlUrls: (botId: string, urls: string[]) => Promise<string | null>
  cancelCrawl: () => Promise<void>
  cancelIndexJob: (botId: string, cancelUrl: string) => Promise<void>
  refreshStatus: (url: string) => Promise<void>
  getJobStatus: (botId: string, jobId: string) => Promise<IndexStatus | null>
  copySnippet: (snippet?: string) => Promise<void>
  refreshAll: () => void
  setSelectedBotId: (value: string | null) => void
  discoverUrls: (
    url: string,
    discoveryMethod?: string,
    onEvent?: (evt: { type: string;[key: string]: unknown }) => void,
    signal?: AbortSignal,
    options?: { max_duration_sec?: number }
  ) => Promise<{ urls: string[]; error?: string; methodUsed?: string; failureReason?: string }>
  startBackgroundDiscovery: (botId: string, url: string, method: string) => Promise<void>
  cancelDiscoveryJob: (botId: string, jobId: string) => Promise<{ status: string } | null>
  listDiscoveryJobs: (botId: string) => Promise<DiscoveryJobRecord[]>
  getDiscoveryJob: (botId: string, jobId: string) => Promise<DiscoveryJobRecord | null>
  listBookingLinkJobs: (botId: string) => Promise<BookingLinkJobRecord[]>
  getBookingLinkJob: (botId: string, jobId: string) => Promise<BookingLinkJobRecord | null>
  getLatestJobPipeline: (botId: string) => Promise<JobPipelineRunRecord | null>
  resumeJobPipeline: (botId: string, runId: string) => Promise<JobPipelineRunRecord | null>
  startAvailabilityJob: (
    botId: string,
    payload: {
      url: string
      check_in?: string
      check_out?: string
      adults?: number
      children?: number
      rooms?: number
      max_seconds?: number
      question?: string
    }
  ) => Promise<AvailabilityJobRecord | null>
  listAvailabilityJobs: (botId: string) => Promise<AvailabilityJobRecord[]>
  getAvailabilityJob: (botId: string, jobId: string) => Promise<AvailabilityJobRecord | null>
  getAvailabilityRaw: (
    botId: string,
    jobId: string,
    format?: 'text' | 'html' | 'debug',
    maxChars?: number
  ) => Promise<{ format: string; content: string } | null>
  deleteBot: (botId: string) => Promise<boolean>
  renameBot: (botId: string, displayName: string) => Promise<boolean>
  listConversations: (
    botId: string,
    limit?: number,
    cursor?: string | null
  ) => Promise<{ sessions: ConversationSessionRecord[]; next_cursor?: string | null; total_count?: number | null }>
  searchConversations: (
    botId: string,
    args: {
      q?: string | null
      from_day?: string | null
      to_day?: string | null
      status?: string | null
      channel?: string | null
      has_escalation?: boolean | null
      site_url?: string | null
      limit?: number
      cursor?: string | null
    }
  ) => Promise<{ sessions: ConversationSearchSessionRecord[]; next_cursor?: string | null; total_count?: number | null }>
  exportConversationsCsv: (
    botId: string,
    args: {
      q?: string | null
      from_day?: string | null
      to_day?: string | null
      status?: string | null
      channel?: string | null
      has_escalation?: boolean | null
      site_url?: string | null
    }
  ) => Promise<void>
  getConversation: (
    botId: string,
    sessionId: string,
    limit?: number
  ) => Promise<ConversationMessageRecord[]>
  endConversation: (botId: string, sessionId: string) => Promise<void>
  takeOverConversation: (botId: string, sessionId: string) => Promise<void>
  getEscalationConfig: (botId: string) => Promise<EscalationConfig | null>
  saveEscalationConfig: (botId: string, config: EscalationConfig) => Promise<EscalationConfig | null>
  getEscalationCounts: (botId: string) => Promise<{ total: number; open: number } | null>
  recomputeAnalytics: (botId: string, args?: { range?: string; from_day?: string | null; to_day?: string | null }) => Promise<boolean>
  getAnalyticsSummary: (botId: string, args?: { range?: string; from_day?: string | null; to_day?: string | null }) => Promise<AnalyticsSummary | null>
  getAnalyticsTimeseries: (botId: string, args?: { range?: string; from_day?: string | null; to_day?: string | null }) => Promise<AnalyticsTimeseries | null>
  getAnalyticsTopSources: (
    botId: string,
    args?: { range?: string; from_day?: string | null; to_day?: string | null; limit?: number }
  ) => Promise<TopSources | null>
  listEscalations: (
    botId: string,
    limit?: number,
    cursor?: string | null
  ) => Promise<{ escalations: EscalationRecord[]; next_cursor?: string | null; total_count?: number | null }>
  getEscalationForSession: (botId: string, sessionId: string) => Promise<EscalationRecord | null>
  updateEscalationStatus: (botId: string, escalationId: string, status: 'open' | 'resolved') => Promise<void>
  getExtractedTopics: (botId: string, activeOnly?: boolean, limit?: number) => Promise<ExtractedTopicsResponse | null>
  extractTopics: (botId: string, clearExisting?: boolean) => Promise<ExtractedTopicsResponse | null>
  updateExtractedTopic: (botId: string, topicId: string, updates: { is_active?: boolean; category?: string }) => Promise<ExtractedTopic | null>
  createExtractedTopic: (botId: string, topic: string, category?: string) => Promise<ExtractedTopic | null>
  deleteExtractedTopic: (botId: string, topicId: string) => Promise<boolean>
  getTopicUsageSummary: (botId: string) => Promise<TopicUsageSummary | null>
  computeTopicMappings: (botId: string) => Promise<{ new_mappings: number } | null>
  getTopicQuestions: (botId: string, topicId: string, limit?: number) => Promise<TopicQuestionsResponse | null>
  syncUrlBankTopics: (botId: string, urlBank: Array<{ label: string; url: string }>) => Promise<ExtractedTopicsResponse | null>
  generateSuggestedMessages: (botId: string) => Promise<unknown[] | null>
  generatingSuggestions: boolean
  fetchPlatformSuggestedMessages: (platform: string, lang?: string) => Promise<Array<{ id: string; label: string; type: string; prompt?: string }>>
  fetchPlatformConfig: (lang?: string) => Promise<PlatformConfigPayload>
}

const DashboardDataContext = createContext<DashboardData | undefined>(undefined)

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin
const ALL_ORGS_ID = "__all__"
const terminalStages = new Set(['done', 'error', 'cancelled', 'import_submitted'])

function isAuthError(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toLowerCase()
  return (
    msg.includes('refresh token') ||
    msg.includes('missing refresh token') ||
    msg.includes('unauthorized') ||
    msg.includes('token expired') ||
    msg.includes('login_required') ||
    msg.includes('invalid token') ||
    msg.includes('authentication required') ||
    msg.includes('401')
  )
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  token?: string,
  onAuthError?: () => void
): Promise<T> {
  const initHeaders = init?.headers
  const headerEntries =
    initHeaders instanceof Headers ? Object.fromEntries(initHeaders.entries()) : (initHeaders as Record<string, string> | undefined)
  const isFormDataBody = typeof FormData !== 'undefined' && init?.body instanceof FormData
  const method = String(init?.method || 'GET').toUpperCase()
  const isReadRequest = method === 'GET' || method === 'HEAD'
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    ...(isReadRequest ? { cache: 'no-store' as RequestCache } : {}),
    headers: {
      ...(isFormDataBody ? {} : { 'Content-Type': 'application/json' }),
      ...(headerEntries || {}),
      ...(isReadRequest ? { 'Cache-Control': 'no-cache', Pragma: 'no-cache' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  })
  if (!res.ok) {
    if (res.status === 401 && onAuthError) {
      onAuthError()
    }
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      detail = body.detail || detail
    } catch {
      // ignore json parse error
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export function DashboardDataProvider({ children }: { children: React.ReactNode }) {
  const [bots, setBots] = useState<BotSummary[]>([])
  const [selectedBotId, setSelectedBotId] = useState<string | null>(null)
  const [selectedBot, setSelectedBot] = useState<BotSummary | null>(null)
  const [selectedBotWidgetConfig, setSelectedBotWidgetConfig] = useState<Record<string, unknown> | null>(null)
  const [domains, setDomains] = useState<DomainRecord[]>([])
  const [jobs, setJobs] = useState<JobRecord[]>([])
  const [sources, setSources] = useState<SourceRecord[]>([])
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [botsLoadedOnce, setBotsLoadedOnce] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newBotName, setNewBotName] = useState('')
  const [newDomain, setNewDomain] = useState('')
  const [crawlUrl, setCrawlUrl] = useState('')
  const [activeCrawlUrl, setActiveCrawlUrl] = useState('')
  const [orgs, setOrgs] = useState<OrgSummary[]>([])
  const [activeOrgId, setActiveOrgId] = useState<string | null>(null)
  const [orgMembers, setOrgMembers] = useState<OrgMember[]>([])
  const [newOrgName, setNewOrgName] = useState('')
  const [orgDisplayName, setOrgDisplayName] = useState('')
  const [orgDisplayNameInput, setOrgDisplayNameInput] = useState('')
  const [newMemberEmail, setNewMemberEmail] = useState('')
  const [newMemberRole, setNewMemberRole] = useState('org_admin')

  const [isSuperAdmin, setIsSuperAdmin] = useState(false)
  const prevActiveOrgIdRef = useRef<string | null>(null)

  const [generatingSuggestions, setGeneratingSuggestions] = useState(false)

  const { getAccessTokenSilently, getIdTokenClaims, user, logout, isAuthenticated, loginWithRedirect } = useAuth0()

  const setErrorSafe = useCallback(
    (value: string | null) => {
      if (value != null && isAuthError(new Error(value))) {
        loginWithRedirect()
        return
      }
      setError(value)
    },
    [loginWithRedirect]
  )

  const embedSnippet = useMemo(() => {
    if (!selectedBot) return ''
    return `<script async src="${API_BASE}/widget/widget.js" data-bot-key="${selectedBot.publishable_key}" data-api-base="${API_BASE}"></script>`
  }, [selectedBot])

  function buildEmbedSnippet(_config?: EmbedSnippetConfig): string {
    if (!selectedBot) return ''
    return `<script async src="${API_BASE}/widget/widget.js" data-bot-key="${selectedBot.publishable_key}" data-api-base="${API_BASE}"></script>`
  }

  async function saveWidgetConfig(botId: string, config: Record<string, unknown>): Promise<void> {
    const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
    const result = await fetchAuthedJson<{ widget_config?: Record<string, unknown> }>(
      withOrgParam(`/v1/org/bots/${botId}/widget-config`, orgOverride),
      {
        method: 'PUT',
        body: JSON.stringify(config),
      }
    )
    const returnedWidgetConfig = result?.widget_config
    if (
      returnedWidgetConfig &&
      typeof returnedWidgetConfig === 'object' &&
      !Array.isArray(returnedWidgetConfig)
    ) {
      setSelectedBotWidgetConfig(returnedWidgetConfig)
      return
    }
    setSelectedBotWidgetConfig((prev) => ({ ...(prev || {}), ...(config || {}) }))
  }

  async function fetchAuthedJson<T>(path: string, init?: RequestInit): Promise<T> {
    let token: string
    try {
      token = await getAccessTokenSilently()
    } catch (err) {
      if (isAuthError(err)) {
        loginWithRedirect()
      }
      throw err
    }
    return fetchJson<T>(path, init, token, loginWithRedirect)
  }

  function withOrgParam(path: string, orgIdOverride?: string | null) {
    const orgId = orgIdOverride ?? activeOrgId
    if (!orgId || orgId === ALL_ORGS_ID) return path
    const suffix = `org_id=${encodeURIComponent(orgId)}`
    return path.includes('?') ? `${path}&${suffix}` : `${path}?${suffix}`
  }

  async function loadBots() {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      if (isSuperAdmin && activeOrgId === ALL_ORGS_ID) {
        const orgIds = orgs.map((org) => org.org_id)
        const results = await Promise.all(
          orgIds.map(async (orgId) => {
            try {
              const data = await fetchAuthedJson<{ bots: BotSummary[] }>(withOrgParam('/v1/org/bots', orgId))
              return data.bots
            } catch {
              return []
            }
          })
        )
        const merged = results.flat()
        const unique = Array.from(new Map(merged.map((bot) => [bot.bot_id, bot])).values())
        setBots(unique)
      } else {
        const data = await fetchAuthedJson<{ bots: BotSummary[] }>(withOrgParam('/v1/org/bots'))
        setBots(data.bots)
        if (!activeOrgId && data.bots.length) {
          setActiveOrgId(data.bots[0].org_id || null)
        }
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
      setBotsLoadedOnce(true)
    }
  }

  async function loadBotDetail(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const data = await fetchAuthedJson<{ bot: BotSummary; widget_config?: Record<string, unknown> }>(
        withOrgParam(`/v1/org/bots/${botId}`, orgOverride)
      )
      if (!activeOrgId && data.bot.org_id) {
        setActiveOrgId(data.bot.org_id)
      }
      setSelectedBot(data.bot)
      setSelectedBotWidgetConfig(data.widget_config ?? null)
    } catch (err) {
      setError((err as Error).message)
      setSelectedBot(null)
    } finally {
      setLoading(false)
    }
  }

  async function loadDomains(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<{ bot_id: string; domains: DomainRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/domains`, orgOverride)
      )
      setDomains(data.domains)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadJobs(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<{ bot_id: string; jobs: JobRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/jobs`, orgOverride)
      )
      setJobs(data.jobs)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadSources(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<{ bot_id: string; sources: SourceRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/sources`, orgOverride)
      )
      setSources(data.sources || [])
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function createSource(
    botId: string,
    type: string,
    config: Record<string, unknown>,
    displayName?: string | null
  ): Promise<SourceRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<SourceRecord>(withOrgParam(`/v1/org/bots/${botId}/sources`, orgOverride), {
        method: 'POST',
        body: JSON.stringify({ type, config, display_name: displayName || null }),
      })
      await loadSources(botId)
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function uploadPdfSources(
    botId: string,
    files: File[],
    displayName?: string | null
  ): Promise<PdfSourceUploadResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    const valid = (files || []).filter(Boolean)
    if (!valid.length) return null
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const form = new FormData()
      for (const f of valid) {
        form.append('files', f, f.name)
      }
      if (displayName && displayName.trim()) {
        form.append('display_name', displayName.trim())
      }
      const data = await fetchAuthedJson<PdfSourceUploadResponse>(
        withOrgParam(`/v1/org/bots/${botId}/sources/pdf`, orgOverride),
        { method: 'POST', body: form }
      )
      await loadSources(botId)
      await loadJobs(botId)
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function uploadTextSources(
    botId: string,
    entries: TextSourceEntry[]
  ): Promise<TextSourceUploadResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    const valid = (entries || []).filter((e) => (e.content || '').trim())
    if (!valid.length) return null
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<TextSourceUploadResponse>(
        withOrgParam(`/v1/org/bots/${botId}/sources/text`, orgOverride),
        { method: 'POST', body: JSON.stringify({ entries: valid }), headers: { 'Content-Type': 'application/json' } }
      )
      await loadSources(botId)
      await loadJobs(botId)
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function uploadDocsSources(
    botId: string,
    files: File[]
  ): Promise<DocsSourceUploadResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    const valid = (files || []).filter(Boolean)
    if (!valid.length) return null
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const form = new FormData()
      for (const f of valid) {
        form.append('files', f, f.name)
      }
      const data = await fetchAuthedJson<DocsSourceUploadResponse>(
        withOrgParam(`/v1/org/bots/${botId}/sources/docs`, orgOverride),
        { method: 'POST', body: form }
      )
      await loadSources(botId)
      await loadJobs(botId)
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function deleteSource(botId: string, sourceId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${botId}/sources/${sourceId}`, orgOverride), {
        method: 'DELETE',
      })
      await loadSources(botId)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function createBot(displayName?: string, orgIdOverride?: string | null): Promise<BotCreateResponse | null> {
    const name = (displayName ?? newBotName).trim()
    if (!name) return null
    const effectiveOrgId = orgIdOverride ?? activeOrgId
    if (isSuperAdmin && (!effectiveOrgId || effectiveOrgId === ALL_ORGS_ID)) {
      setError("Select an organization to create a bot")
      return null
    }
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<BotCreateResponse>(withOrgParam('/v1/org/bots', effectiveOrgId), {
        method: 'POST',
        body: JSON.stringify({ display_name: name }),
      })
      if (!displayName) setNewBotName('')
      await loadBots()
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    } finally {
      setLoading(false)
    }
  }

  async function loadOrgs() {
    if (!isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<{ orgs: OrgSummary[] }>('/v1/admin/orgs')
      setOrgs(data.orgs)
      if (!activeOrgId) {
        setActiveOrgId(ALL_ORGS_ID)
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadSelfOrgs() {
    if (isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<{ org_ids: string[] }>('/v1/org/self')
      const ids = data.org_ids || []
      if (!activeOrgId && ids.length) {
        setActiveOrgId(ids[0])
      }
      if (ids.length) {
        const orgList = await Promise.all(
          ids.map(async (orgId) => {
            try {
              return await fetchAuthedJson<OrgSummary>(`/v1/org/info?org_id=${encodeURIComponent(orgId)}`)
            } catch {
              return {
                org_id: orgId,
                name: orgId,
                status: 'active',
                created_at: '',
                updated_at: '',
              } as OrgSummary
            }
          })
        )
        setOrgs(orgList)
      } else {
        setOrgs([])
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadOrgInfo(orgId: string) {
    if (isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<OrgSummary>(`/v1/org/info?org_id=${encodeURIComponent(orgId)}`)
      setOrgDisplayName(data.name)
      setOrgDisplayNameInput(data.name)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function saveOrgName() {
    if (!activeOrgId || !orgDisplayNameInput.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<OrgSummary>(`/v1/org/name?org_id=${encodeURIComponent(activeOrgId)}`, {
        method: 'POST',
        body: JSON.stringify({ name: orgDisplayNameInput.trim() }),
      })
      setOrgDisplayName(data.name)
      setOrgDisplayNameInput(data.name)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function createOrg() {
    if (!newOrgName.trim()) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson('/v1/admin/orgs', {
        method: 'POST',
        body: JSON.stringify({ name: newOrgName.trim() }),
      })
      setNewOrgName('')
      await loadOrgs()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function setOrgStatus(orgId: string, status: 'active' | 'disabled') {
    if (!isSuperAdmin) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson(`/v1/admin/orgs/${orgId}/${status === 'active' ? 'enable' : 'disable'}`, {
        method: 'POST',
      })
      await loadOrgs()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function loadOrgMembers(orgId: string) {
    try {
      if (isSuperAdmin && orgId === ALL_ORGS_ID) {
        let orgList = orgs
        if (!orgList.length) {
          const data = await fetchAuthedJson<{ orgs: OrgSummary[] }>('/v1/admin/orgs')
          orgList = data.orgs || []
          setOrgs(orgList)
        }
        if (!orgList.length) {
          setOrgMembers([])
          return
        }
        const responses = await Promise.all(
          orgList.map(async (org) => {
            try {
              const data = await fetchAuthedJson<{ org_id: string; members: OrgMember[] }>(`/v1/admin/orgs/${org.org_id}/members`)
              return (data.members || []).map((member) => ({
                ...member,
                org_id: org.org_id,
                org_name: org.name,
              }))
            } catch {
              return [] as OrgMember[]
            }
          })
        )
        const merged = responses.flat()
        setOrgMembers(merged)
        return
      }
      const path = isSuperAdmin ? `/v1/admin/orgs/${orgId}/members` : `/v1/org/members?org_id=${encodeURIComponent(orgId)}`
      const data = await fetchAuthedJson<{ org_id: string; members: OrgMember[] }>(path)
      if (isSuperAdmin) {
        const org = orgs.find((entry) => entry.org_id === orgId)
        setOrgMembers(
          (data.members || []).map((member) => ({
            ...member,
            org_id: orgId,
            org_name: org?.name,
          }))
        )
      } else {
        setOrgMembers(data.members)
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function addOrgMember() {
    if (!activeOrgId || !newMemberEmail.trim()) return
    setLoading(true)
    setError(null)
    try {
      const path = isSuperAdmin ? `/v1/admin/orgs/${activeOrgId}/members` : `/v1/org/members?org_id=${encodeURIComponent(activeOrgId)}`
      await fetchAuthedJson(path, {
        method: 'POST',
        body: JSON.stringify({ email: newMemberEmail.trim(), role: newMemberRole }),
      })
      setNewMemberEmail('')
      await loadOrgMembers(activeOrgId)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function clearGcs() {
    if (!isSuperAdmin) return
    if (!window.confirm('Delete all crawled content from GCS? This cannot be undone.')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson('/v1/admin/reset/gcs', { method: 'POST' })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function clearRag() {
    if (!isSuperAdmin) return
    if (!window.confirm('Delete all RAG corpora? This cannot be undone.')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson('/v1/admin/reset/rag', { method: 'POST' })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function addDomain() {
    if (!selectedBot || !newDomain.trim() || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/domains`, orgOverride), {
        method: 'POST',
        body: JSON.stringify({ hostname: newDomain.trim() }),
      })
      setNewDomain('')
      await loadDomains(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function verifyDomain(hostname: string) {
    if (!selectedBot || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/domains/${hostname}/verify`, orgOverride), {
        method: 'POST',
      })
      await loadDomains(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function startCrawl() {
    if (!selectedBot || !crawlUrl.trim() || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index`, orgOverride), {
        method: 'POST',
        body: JSON.stringify({ url: crawlUrl.trim() }),
      })
      setActiveCrawlUrl(crawlUrl.trim())
      await refreshStatus(crawlUrl.trim())
      await loadJobs(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function startCrawlForSource(botId: string, sourceId: string) {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${botId}/sources/${sourceId}/crawl-single`, orgOverride), {
        method: 'POST',
      })
      await loadJobs(botId)
      await loadSources(botId)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function syncSource(botId: string, sourceId: string) {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${botId}/sources/${sourceId}/sync`, orgOverride), {
        method: 'POST',
      })
      await loadJobs(botId)
      await loadSources(botId)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function updateSourceSyncSettings(botId: string, sourceId: string, settings: SyncSettings): Promise<SourceRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<SourceRecord>(
        withOrgParam(`/v1/org/bots/${botId}/sources/${sourceId}/sync-settings`, orgOverride),
        {
          method: 'PUT',
          body: JSON.stringify(settings),
        }
      )
      await loadSources(botId)
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function queueCrawlUrls(botId: string, urls: string[]): Promise<string | null> {
    if (isSuperAdmin && !activeOrgId) return null
    setLoading(true)
    setError(null)
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const cleaned = urls.map((url) => url.trim()).filter(Boolean)
      const data = await fetchAuthedJson<{ job_id: string }>(withOrgParam(`/v1/org/bots/${botId}/index/batch`, orgOverride), {
        method: 'POST',
        body: JSON.stringify({ urls: cleaned }),
      })
      return data.job_id || null
    } catch (err) {
      setError((err as Error).message)
      return null
    } finally {
      setLoading(false)
    }
  }

  async function startBackgroundDiscovery(botId: string, url: string, method: string): Promise<void> {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      await fetchAuthedJson<{ job_id: string; status: string }>(
        withOrgParam(`/v1/org/bots/${botId}/discovery-jobs`, orgOverride),
        {
          method: 'POST',
          body: JSON.stringify({ url: url.trim(), method: (method || 'auto').toLowerCase() }),
        }
      )
    } catch {
      // Fire-and-forget; do not block or surface error to create-bot flow
    }
  }

  async function cancelDiscoveryJob(
    botId: string,
    jobId: string
  ): Promise<{ status: string } | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      return await fetchAuthedJson<{ status: string }>(
        withOrgParam(`/v1/org/bots/${botId}/discovery-jobs/${encodeURIComponent(jobId)}/cancel`, orgOverride),
        { method: 'POST' }
      )
    } catch {
      return null
    }
  }

  async function listDiscoveryJobs(botId: string): Promise<DiscoveryJobRecord[]> {
    if (isSuperAdmin && !activeOrgId) return []
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const data = await fetchAuthedJson<{ jobs: DiscoveryJobRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/discovery-jobs`, orgOverride)
      )
      return data.jobs || []
    } catch {
      return []
    }
  }



  async function listBookingLinkJobs(botId: string): Promise<BookingLinkJobRecord[]> {
    if (isSuperAdmin && !activeOrgId) return []
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/booking-links`, orgOverride)
      const data = await fetchAuthedJson<{ jobs: BookingLinkJobRecord[] }>(path)
      return data.jobs || []
    } catch (err) {
      setError((err as Error).message)
      return []
    }
  }

  async function getBookingLinkJob(botId: string, jobId: string): Promise<BookingLinkJobRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/booking-links/${encodeURIComponent(jobId)}`, orgOverride)
      return await fetchAuthedJson<BookingLinkJobRecord>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function getLatestJobPipeline(botId: string): Promise<JobPipelineRunRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/job-pipelines/latest`, orgOverride)
      const data = await fetchAuthedJson<{ bot_id: string; run?: JobPipelineRunRecord | null }>(path)
      return data.run || null
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function resumeJobPipeline(botId: string, runId: string): Promise<JobPipelineRunRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/job-pipelines/${encodeURIComponent(runId)}/resume`, orgOverride)
      const data = await fetchAuthedJson<{ bot_id: string; run: JobPipelineRunRecord }>(path, { method: 'POST' })
      return data.run || null
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function startAvailabilityJob(
    botId: string,
    payload: {
      url: string
      check_in?: string
      check_out?: string
      adults?: number
      children?: number
      rooms?: number
      max_seconds?: number
      question?: string
    }
  ): Promise<AvailabilityJobRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/availability`, orgOverride)
      return await fetchAuthedJson<AvailabilityJobRecord>(path, {
        method: 'POST',
        body: JSON.stringify(payload),
      })
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function listAvailabilityJobs(botId: string): Promise<AvailabilityJobRecord[]> {
    if (isSuperAdmin && !activeOrgId) return []
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/availability`, orgOverride)
      const data = await fetchAuthedJson<{ jobs: AvailabilityJobRecord[] }>(path)
      return data.jobs || []
    } catch (err) {
      setError((err as Error).message)
      return []
    }
  }

  async function getAvailabilityJob(
    botId: string,
    jobId: string
  ): Promise<AvailabilityJobRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/availability/${encodeURIComponent(jobId)}`, orgOverride)
      return await fetchAuthedJson<AvailabilityJobRecord>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function getAvailabilityRaw(
    botId: string,
    jobId: string,
    format: 'text' | 'html' | 'debug' = 'text',
    maxChars = 0
  ): Promise<{ format: string; content: string } | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const params = new URLSearchParams()
      params.set('format', format)
      if (maxChars > 0) params.set('max_chars', String(maxChars))
      const path = withOrgParam(`/v1/org/bots/${botId}/availability/${jobId}/raw?${params.toString()}`, orgOverride)
      return await fetchAuthedJson<{ format: string; content: string }>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function getDiscoveryJob(botId: string, jobId: string): Promise<DiscoveryJobRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      return await fetchAuthedJson<DiscoveryJobRecord>(
        withOrgParam(`/v1/org/bots/${botId}/discovery-jobs/${jobId}`, orgOverride)
      )
    } catch {
      return null
    }
  }

  async function listConversations(
    botId: string,
    limit = 50,
    cursor: string | null = null
  ): Promise<{ sessions: ConversationSessionRecord[]; next_cursor?: string | null; total_count?: number | null }> {
    if (isSuperAdmin && !activeOrgId) return { sessions: [] }
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const cursorParam = cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''
      const path = withOrgParam(`/v1/org/bots/${botId}/conversations?limit=${limit}${cursorParam}`, orgOverride)
      return await fetchAuthedJson<{ sessions: ConversationSessionRecord[]; next_cursor?: string | null; total_count?: number | null }>(path)
    } catch (err) {
      setError((err as Error).message)
      return { sessions: [] }
    }
  }

  async function searchConversations(
    botId: string,
    args: {
      q?: string | null
      from_day?: string | null
      to_day?: string | null
      status?: string | null
      channel?: string | null
      has_escalation?: boolean | null
      site_url?: string | null
      limit?: number
      cursor?: string | null
    }
  ): Promise<{ sessions: ConversationSearchSessionRecord[]; next_cursor?: string | null; total_count?: number | null }> {
    if (isSuperAdmin && !activeOrgId) return { sessions: [] }
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const qp = new URLSearchParams()
      if (args.q) qp.set('q', args.q)
      if (args.from_day) qp.set('from_day', args.from_day)
      if (args.to_day) qp.set('to_day', args.to_day)
      if (args.status) qp.set('status', args.status)
      if (args.channel) qp.set('channel', args.channel)
      if (args.site_url) qp.set('site_url', args.site_url)
      if (args.has_escalation != null) qp.set('has_escalation', String(args.has_escalation))
      if (args.limit != null) qp.set('limit', String(args.limit))
      if (args.cursor) qp.set('cursor', args.cursor)
      const path = withOrgParam(`/v1/org/bots/${botId}/conversations/search?${qp.toString()}`, orgOverride)
      return await fetchAuthedJson<{ sessions: ConversationSearchSessionRecord[]; next_cursor?: string | null; total_count?: number | null }>(
        path
      )
    } catch (err) {
      setError((err as Error).message)
      return { sessions: [] }
    }
  }

  async function exportConversationsCsv(
    botId: string,
    args: {
      q?: string | null
      from_day?: string | null
      to_day?: string | null
      status?: string | null
      channel?: string | null
      has_escalation?: boolean | null
      site_url?: string | null
    }
  ): Promise<void> {
    if (isSuperAdmin && !activeOrgId) return
    const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
    const qp = new URLSearchParams()
    if (args.q) qp.set('q', args.q)
    if (args.from_day) qp.set('from_day', args.from_day)
    if (args.to_day) qp.set('to_day', args.to_day)
    if (args.status) qp.set('status', args.status)
    if (args.channel) qp.set('channel', args.channel)
    if (args.site_url) qp.set('site_url', args.site_url)
    if (args.has_escalation != null) qp.set('has_escalation', String(args.has_escalation))
    const path = withOrgParam(`/v1/org/bots/${botId}/conversations/export.csv?${qp.toString()}`, orgOverride)
    const token = await getAccessTokenSilently()
    const res = await fetch(`${API_BASE}${path}`, { headers: { Authorization: `Bearer ${token}` } })
    if (!res.ok) throw new Error(res.statusText)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversations_${botId}.csv`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  async function recomputeAnalytics(
    botId: string,
    args: { range?: string; from_day?: string | null; to_day?: string | null } = {}
  ): Promise<boolean> {
    if (isSuperAdmin && !activeOrgId) return false
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const qp = new URLSearchParams()
      qp.set('range', args.range || '30d')
      if (args.from_day) qp.set('from_day', args.from_day)
      if (args.to_day) qp.set('to_day', args.to_day)
      const path = withOrgParam(`/v1/org/bots/${botId}/analytics/recompute?${qp.toString()}`, orgOverride)
      await fetchAuthedJson<{ ok: boolean }>(path, { method: 'POST' })
      return true
    } catch (err) {
      setError((err as Error).message)
      return false
    }
  }

  async function getAnalyticsSummary(
    botId: string,
    args: { range?: string; from_day?: string | null; to_day?: string | null } = {}
  ): Promise<AnalyticsSummary | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const qp = new URLSearchParams()
      qp.set('range', args.range || '30d')
      if (args.from_day) qp.set('from_day', args.from_day)
      if (args.to_day) qp.set('to_day', args.to_day)
      const path = withOrgParam(`/v1/org/bots/${botId}/analytics/summary?${qp.toString()}`, orgOverride)
      return await fetchAuthedJson<AnalyticsSummary>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function getAnalyticsTimeseries(
    botId: string,
    args: { range?: string; from_day?: string | null; to_day?: string | null } = {}
  ): Promise<AnalyticsTimeseries | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const qp = new URLSearchParams()
      qp.set('range', args.range || '30d')
      if (args.from_day) qp.set('from_day', args.from_day)
      if (args.to_day) qp.set('to_day', args.to_day)
      const path = withOrgParam(`/v1/org/bots/${botId}/analytics/timeseries?${qp.toString()}`, orgOverride)
      return await fetchAuthedJson<AnalyticsTimeseries>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function getAnalyticsTopSources(
    botId: string,
    args: { range?: string; from_day?: string | null; to_day?: string | null; limit?: number } = {}
  ): Promise<TopSources | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const limit = args.limit ?? 10
      const qp = new URLSearchParams()
      qp.set('range', args.range || '30d')
      qp.set('limit', String(limit))
      if (args.from_day) qp.set('from_day', args.from_day)
      if (args.to_day) qp.set('to_day', args.to_day)
      const path = withOrgParam(
        `/v1/org/bots/${botId}/analytics/top-sources?${qp.toString()}`,
        orgOverride
      )
      return await fetchAuthedJson<TopSources>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }


  async function getConversation(
    botId: string,
    sessionId: string,
    limit = 200
  ): Promise<ConversationMessageRecord[]> {
    if (isSuperAdmin && !activeOrgId) return []
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(
        `/v1/org/bots/${botId}/conversations/${encodeURIComponent(sessionId)}?limit=${limit}`,
        orgOverride
      )
      const data = await fetchAuthedJson<{ messages: ConversationMessageRecord[] }>(path)
      return data.messages || []
    } catch (err) {
      setError((err as Error).message)
      return []
    }
  }

  async function endConversation(botId: string, sessionId: string): Promise<void> {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(
        `/v1/org/bots/${botId}/conversations/${encodeURIComponent(sessionId)}/end`,
        orgOverride
      )
      await fetchAuthedJson(path, { method: 'POST' })
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function takeOverConversation(botId: string, sessionId: string): Promise<void> {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(
        `/v1/org/bots/${botId}/conversations/${encodeURIComponent(sessionId)}/takeover`,
        orgOverride
      )
      await fetchAuthedJson(path, { method: 'POST' })
    } catch (err) {
      setError((err as Error).message)
      throw err
    }
  }

  async function getEscalationConfig(botId: string): Promise<EscalationConfig | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/escalation-config`, orgOverride)
      return await fetchAuthedJson<EscalationConfig>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function saveEscalationConfig(botId: string, config: EscalationConfig): Promise<EscalationConfig | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/escalation-config`, orgOverride)
      return await fetchAuthedJson<EscalationConfig>(path, {
        method: 'PUT',
        body: JSON.stringify(config),
      })
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function getEscalationCounts(botId: string): Promise<{ total: number; open: number } | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/escalations/counts`, orgOverride)
      return await fetchAuthedJson<{ bot_id: string; total: number; open: number }>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function listEscalations(
    botId: string,
    limit = 10,
    cursor: string | null = null
  ): Promise<{ escalations: EscalationRecord[]; next_cursor?: string | null; total_count?: number | null }> {
    if (isSuperAdmin && !activeOrgId) return { escalations: [] }
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const cursorParam = cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''
      const path = withOrgParam(`/v1/org/bots/${botId}/escalations?limit=${limit}${cursorParam}`, orgOverride)
      return await fetchAuthedJson<{ escalations: EscalationRecord[]; next_cursor?: string | null; total_count?: number | null }>(path)
    } catch (err) {
      setError((err as Error).message)
      return { escalations: [] }
    }
  }

  async function getEscalationForSession(botId: string, sessionId: string): Promise<EscalationRecord | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/escalations/${encodeURIComponent(sessionId)}`, orgOverride)
      return await fetchAuthedJson<EscalationRecord>(path)
    } catch {
      return null
    }
  }

  async function updateEscalationStatus(
    botId: string,
    escalationId: string,
    status: 'open' | 'resolved'
  ): Promise<void> {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/escalations/${encodeURIComponent(escalationId)}/status`, orgOverride)
      await fetchAuthedJson(path, { method: 'POST', body: JSON.stringify({ status }) })
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function getExtractedTopics(
    botId: string,
    activeOnly: boolean = false,
    limit: number = 100
  ): Promise<ExtractedTopicsResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const params = new URLSearchParams()
      if (activeOnly) params.set('active_only', 'true')
      params.set('limit', String(limit))
      const path = withOrgParam(`/v1/org/bots/${botId}/extracted-topics?${params.toString()}`, orgOverride)
      return await fetchAuthedJson<ExtractedTopicsResponse>(path)
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function extractTopics(
    botId: string,
    clearExisting: boolean = false
  ): Promise<ExtractedTopicsResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/extracted-topics/extract`, orgOverride)
      return await fetchAuthedJson<ExtractedTopicsResponse>(path, {
        method: 'POST',
        body: JSON.stringify({ clear_existing: clearExisting }),
      })
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function updateExtractedTopic(
    botId: string,
    topicId: string,
    updates: { is_active?: boolean; category?: string }
  ): Promise<ExtractedTopic | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/extracted-topics/${encodeURIComponent(topicId)}`, orgOverride)
      return await fetchAuthedJson<ExtractedTopic>(path, {
        method: 'PATCH',
        body: JSON.stringify(updates),
      })
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function createExtractedTopic(
    botId: string,
    topic: string,
    category?: string
  ): Promise<ExtractedTopic | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/extracted-topics`, orgOverride)
      return await fetchAuthedJson<ExtractedTopic>(path, {
        method: 'POST',
        body: JSON.stringify({ topic, category: category || null }),
      })
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function deleteExtractedTopic(
    botId: string,
    topicId: string
  ): Promise<boolean> {
    if (isSuperAdmin && !activeOrgId) return false
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/extracted-topics/${encodeURIComponent(topicId)}`, orgOverride)
      await fetchAuthedJson(path, { method: 'DELETE' })
      return true
    } catch (err) {
      setError((err as Error).message)
      return false
    }
  }

  async function getTopicUsageSummary(botId: string): Promise<TopicUsageSummary | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/topics/usage-summary`, orgOverride)
      return await fetchAuthedJson<TopicUsageSummary>(path)
    } catch (err) {
      return null
    }
  }

  async function computeTopicMappings(botId: string): Promise<{ new_mappings: number } | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/topics/compute-mappings`, orgOverride)
      return await fetchAuthedJson<{ new_mappings: number }>(path, { method: 'POST', body: '{}' })
    } catch (err) {
      return null
    }
  }

  async function getTopicQuestions(botId: string, topicId: string, limit = 50): Promise<TopicQuestionsResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/topics/${encodeURIComponent(topicId)}/questions?limit=${limit}`, orgOverride)
      return await fetchAuthedJson<TopicQuestionsResponse>(path)
    } catch (err) {
      return null
    }
  }

  async function syncUrlBankTopics(
    botId: string,
    urlBank: Array<{ label: string; url: string }>
  ): Promise<ExtractedTopicsResponse | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/topics/sync-url-bank`, orgOverride)
      return await fetchAuthedJson<ExtractedTopicsResponse>(path, {
        method: 'POST',
        body: JSON.stringify({ url_bank: urlBank }),
      })
    } catch (err) {
      return null
    }
  }

  async function discoverUrls(
    url: string,
    discoveryMethod: string = 'auto',
    onEvent?: (evt: { type: string;[key: string]: unknown }) => void,
    signal?: AbortSignal,
    options?: { max_duration_sec?: number }
  ): Promise<{ urls: string[]; error?: string; methodUsed?: string; failureReason?: string }> {
    if (isSuperAdmin && !activeOrgId) return { urls: [] }
    setLoading(true)
    setError(null)
    const normalizeDiscoveryUrl = (raw: string): string => {
      const v = (raw || '').trim()
      if (!v) return ''
      try {
        const u = new URL(v)
        let path = u.pathname || '/'
        if (!path) path = '/'
        if (path !== '/' && !path.endsWith('/')) {
          const leaf = path.split('/').pop() || ''
          if (!/\.[a-z0-9]{1,8}$/i.test(leaf)) path = `${path}/`
        }
        u.pathname = path
        u.hash = ''
        return u.toString()
      } catch {
        return v
      }
    }
    const discoveryDedupeKey = (raw: string): string => {
      const v = (raw || '').trim()
      if (!v) return ''
      try {
        const u = new URL(v)
        const path = u.pathname || '/'
        const pathNoSlash = path === '/' ? '/' : (path.replace(/\/+$/, '') || '/')
        return `${u.origin}${pathNoSlash}${u.search}`
      } catch {
        return v
      }
    }
    const hasPathTrailingSlash = (raw: string): boolean => {
      try {
        return new URL(raw).pathname.endsWith('/')
      } catch {
        return raw.endsWith('/')
      }
    }

    const collectedByKey = new Map<string, string>()
    const collectedOrder: string[] = []
    const pushUnique = (candidate: string): boolean => {
      const normalized = normalizeDiscoveryUrl(candidate)
      if (!normalized) return false
      const key = discoveryDedupeKey(normalized)
      if (!key) return false
      const existing = collectedByKey.get(key)
      if (!existing) {
        collectedByKey.set(key, normalized)
        collectedOrder.push(key)
        return true
      }
      // If both variants exist, keep trailing-slash version.
      if (!hasPathTrailingSlash(existing) && hasPathTrailingSlash(normalized)) {
        collectedByKey.set(key, normalized)
      }
      return false
    }
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const token = await getAccessTokenSilently()
      const body: { url: string; method: string; max_duration_sec?: number } = { url, method: discoveryMethod }
      if (options?.max_duration_sec != null) body.max_duration_sec = options.max_duration_sec

      const res = await fetch(`${API_BASE}${withOrgParam('/v1/org/url-discovery/stream', orgOverride)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(body),
        signal,
      })

      if (!res.ok) {
        let detail = res.statusText
        try {
          const body = (await res.json()) as { detail?: string }
          detail = body.detail || detail
        } catch {
          // ignore
        }
        throw new Error(detail)
      }

      if (!res.body) {
        throw new Error('No response body from discovery stream')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalError: string | undefined
      let finalMethod: string | undefined
      let finalFailureReason: string | undefined

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue
          let evt: { type: string;[key: string]: unknown }
          try {
            evt = JSON.parse(trimmed) as { type: string;[key: string]: unknown }
          } catch {
            continue
          }

          if (evt.type === 'discovered' && typeof evt.url === 'string') {
            const normalizedUrl = normalizeDiscoveryUrl(evt.url)
            if (!normalizedUrl) continue
            const isNew = pushUnique(normalizedUrl)
            if (isNew && onEvent) onEvent({ ...evt, url: normalizedUrl })
          } else if (evt.type === 'done') {
            if (Array.isArray(evt.urls)) {
              for (const u of evt.urls) {
                if (typeof u === 'string') pushUnique(u)
              }
            }
            const urls = collectedOrder
              .map((k) => collectedByKey.get(k))
              .filter((u): u is string => !!u)
            if (onEvent) onEvent({ ...evt, urls })
          } else {
            if (onEvent) onEvent(evt)
          }

          if (evt.type === 'error' && typeof evt.message === 'string') {
            finalError = evt.message
          }

          if (evt.type === 'done') {
            if (typeof evt.method_used === 'string') finalMethod = evt.method_used
            if (typeof evt.failure_reason === 'string') finalFailureReason = evt.failure_reason
          }

          if (typeof evt.method_used === 'string') finalMethod = evt.method_used
          if (typeof evt.failure_reason === 'string') finalFailureReason = evt.failure_reason
        }
      }

      const collected = collectedOrder
        .map((k) => collectedByKey.get(k))
        .filter((u): u is string => !!u)
      return { urls: collected, error: finalError, methodUsed: finalMethod, failureReason: finalFailureReason }
    } catch (err) {
      const e = err as Error & { name?: string }
      if (e.name === 'AbortError') {
        setLoading(false)
        const collected = collectedOrder
          .map((k) => collectedByKey.get(k))
          .filter((u): u is string => !!u)
        return { urls: collected }
      }
      const errorMsg = e.message
      setError(errorMsg)
      return { urls: [], error: errorMsg }
    } finally {
      setLoading(false)
    }
  }

  async function deleteBot(botId: string): Promise<boolean> {
    if (isSuperAdmin && !activeOrgId) return false
    setLoading(true)
    setError(null)
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${botId}`, orgOverride), {
        method: 'DELETE',
      })
      // If deleted bot was selected, clear selection
      if (selectedBotId === botId) {
        setSelectedBotId(null)
      }
      await loadBots()
      return true
    } catch (err) {
      setError((err as Error).message)
      return false
    } finally {
      setLoading(false)
    }
  }

  async function renameBot(botId: string, displayName: string): Promise<boolean> {
    if (isSuperAdmin && !activeOrgId) return false
    const name = (displayName || '').trim()
    if (!name) return false
    setLoading(true)
    setError(null)
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${botId}`, orgOverride), {
        method: 'PATCH',
        body: JSON.stringify({ display_name: name }),
      })
      await loadBots()
      if (selectedBotId === botId) {
        await loadBotDetail(botId)
      }
      return true
    } catch (err) {
      setError((err as Error).message)
      return false
    } finally {
      setLoading(false)
    }
  }

  async function cancelCrawl() {
    if (!selectedBot || !activeCrawlUrl || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index/cancel`, orgOverride), {
        method: 'POST',
        body: JSON.stringify({ url: activeCrawlUrl }),
      })
      await refreshStatus(activeCrawlUrl)
      await loadJobs(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function cancelIndexJob(botId: string, cancelUrl: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${botId}/index/cancel`, orgOverride), {
        method: 'POST',
        body: JSON.stringify({ url: cancelUrl }),
      })
      await loadJobs(botId)
      await loadSources(botId)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function refreshStatus(url: string) {
    if (!selectedBot || (isSuperAdmin && !activeOrgId)) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index/status?url=${encodeURIComponent(url)}`, orgOverride)
      const status = await fetchAuthedJson<IndexStatus>(path)
      setIndexStatus(status)

    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function getJobStatus(botId: string, jobId: string): Promise<IndexStatus | null> {
    if (isSuperAdmin && !activeOrgId) return null
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/index/status?job_id=${encodeURIComponent(jobId)}`, orgOverride)
      const status = await fetchAuthedJson<IndexStatus>(path)
      return status
    } catch (err) {
      setError((err as Error).message)
      return null
    }
  }

  async function fetchPlatformConfig(
    lang?: string
  ): Promise<PlatformConfigPayload> {
    try {
      const langParam = lang ? `?lang=${encodeURIComponent(lang)}` : ''
      const path = withOrgParam(`/v1/org/platform-config${langParam}`)
      const result = await fetchAuthedJson<{
        platforms?: Array<{
          id: string
          widget_key: string
          domain_key: string
          label: string
          url_placeholder?: string
          availableSuggestedMessageTypes?: string[]
        }>
        defaultSuggestedMessages?: Array<{ id: string; label: string; type: string; prompt?: string }>
        defaultAvailableSuggestedMessageTypes?: string[]
        jobPipelineWorkflow?: {
          workflowId?: string
          default?: string[]
          platformOverrides?: Record<string, string[]>
        }
      }>(path)
      const rawWorkflow = result.jobPipelineWorkflow
      const workflowDefault = Array.isArray(rawWorkflow?.default)
        ? (rawWorkflow.default as unknown[])
            .filter((stepId) => typeof stepId === 'string')
            .map((stepId) => String(stepId).trim())
            .filter(Boolean)
        : []
      const rawOverrides = rawWorkflow?.platformOverrides
      const workflowOverrides: Record<string, string[]> = {}
      if (rawOverrides && typeof rawOverrides === 'object') {
        for (const [platformId, rawSteps] of Object.entries(rawOverrides)) {
          const pid = String(platformId || '').trim().toLowerCase()
          if (!pid || !Array.isArray(rawSteps)) continue
          const steps = rawSteps.filter((stepId) => typeof stepId === 'string').map((stepId) => stepId.trim()).filter(Boolean)
          if (steps.length > 0) workflowOverrides[pid] = steps
        }
      }
      return {
        platforms: result.platforms ?? [],
        defaultSuggestedMessages: result.defaultSuggestedMessages ?? [],
        defaultAvailableSuggestedMessageTypes: result.defaultAvailableSuggestedMessageTypes ?? ['ai_response'],
        jobPipelineWorkflow: {
          workflowId: String(rawWorkflow?.workflowId || 'default').trim() || 'default',
          default: workflowDefault,
          platformOverrides: workflowOverrides,
        },
      }
    } catch {
      return {
        platforms: [],
        defaultSuggestedMessages: [],
        defaultAvailableSuggestedMessageTypes: ['ai_response'],
        jobPipelineWorkflow: { workflowId: 'default', default: [], platformOverrides: {} },
      }
    }
  }

  async function fetchPlatformSuggestedMessages(
    platform: string,
    lang?: string
  ): Promise<Array<{ id: string; label: string; type: string; prompt?: string }>> {
    try {
      const langParam = lang ? `&lang=${encodeURIComponent(lang)}` : ''
      const path = withOrgParam(`/v1/org/platform-suggested-messages?platform=${encodeURIComponent(platform)}${langParam}`)
      const result = await fetchAuthedJson<{ suggestedMessages?: Array<{ id: string; label: string; type: string; prompt?: string }> }>(path)
      return result.suggestedMessages ?? []
    } catch {
      return []
    }
  }

  async function generateSuggestedMessages(botId: string): Promise<unknown[] | null> {
    if (isSuperAdmin && !activeOrgId) return null
    setGeneratingSuggestions(true)
    try {
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${botId}/generate-suggested-messages`, orgOverride)
      const result = await fetchAuthedJson<{ suggestedMessages: unknown[] }>(path, { method: 'POST' })
      if (result.suggestedMessages?.length) {
        setSelectedBotWidgetConfig((prev) => ({
          ...(prev || {}),
          suggestedMessages: result.suggestedMessages,
        }))
        return result.suggestedMessages
      }
      return null
    } catch {
      return null
    } finally {
      setGeneratingSuggestions(false)
    }
  }

  async function copySnippet(snippet?: string) {
    const text = snippet ?? embedSnippet
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  function refreshAll() {
    if (activeOrgId || !isSuperAdmin) {
      void loadBots()
      if (selectedBotId) {
        void loadBotDetail(selectedBotId)
        void loadDomains(selectedBotId)
        void loadJobs(selectedBotId)
        void loadSources(selectedBotId)
      }
    }
    if (isSuperAdmin) {
      void loadOrgs()
      if (activeOrgId) {
        void loadOrgMembers(activeOrgId)
      }
    }
  }

  useEffect(() => {
    if (!isAuthenticated) return
    void getIdTokenClaims().then((claims: TokenClaims | undefined) => {
      const roles =
        (claims?.roles as string[]) ||
        (claims?.permissions as string[]) ||
        (claims && (claims['https://web-ai/roles'] as string[])) ||
        []
      const env = (import.meta as { env: Record<string, string> }).env
      const adminEmails = (env.VITE_SUPER_ADMIN_EMAILS || '')
        .split(',')
        .map((entry) => entry.trim().toLowerCase())
        .filter(Boolean)
      const claimEmail = (claims?.email as string | undefined)?.toLowerCase()
      const superAdmin =
        roles.includes('super_admin') || roles.includes('owner') || (claimEmail ? adminEmails.includes(claimEmail) : false)
      setIsSuperAdmin(superAdmin)

      const orgClaimKey = (import.meta as { env: Record<string, string> }).env.VITE_AUTH_ORG_CLAIM
      const claimValue = orgClaimKey ? (claims?.[orgClaimKey] as string | string[] | undefined) : undefined
      const orgsFromClaim =
        claimValue ||
        (claims?.org_id as string | string[] | undefined) ||
        (claims?.org as string | string[] | undefined) ||
        (claims?.organization_id as string | string[] | undefined)
      const ids = Array.isArray(orgsFromClaim) ? orgsFromClaim : orgsFromClaim ? [orgsFromClaim] : []
      if (!activeOrgId && ids.length === 1) {
        setActiveOrgId(ids[0])
      }
    })
  }, [isAuthenticated, getIdTokenClaims, activeOrgId])

  useEffect(() => {
    if (!isAuthenticated) return
    if (isSuperAdmin) {
      void loadOrgs()
    } else {
      void loadSelfOrgs()
    }
  }, [isAuthenticated, isSuperAdmin])

  useEffect(() => {
    if (!isAuthenticated || (isSuperAdmin && !activeOrgId)) return
    void loadBots()
  }, [isAuthenticated, activeOrgId, isSuperAdmin, orgs.length])

  useEffect(() => {
    if (activeOrgId) {
      setError(null)
    }
  }, [activeOrgId])

  useEffect(() => {
    if (!activeOrgId) return
    // Only reset bot state when SWITCHING orgs, not on initial resolution (null → value)
    if (prevActiveOrgIdRef.current !== null && prevActiveOrgIdRef.current !== activeOrgId) {
      setBots([])
      setSelectedBotId(null)
      setSelectedBot(null)
      setDomains([])
      setJobs([])
      setIndexStatus(null)
      setActiveCrawlUrl('')
    }
    prevActiveOrgIdRef.current = activeOrgId
  }, [activeOrgId])

  useEffect(() => {
    if (!activeOrgId) return
    void loadOrgMembers(activeOrgId)
  }, [isSuperAdmin, activeOrgId])

  useEffect(() => {
    if (!activeOrgId || isSuperAdmin) return
    void loadOrgInfo(activeOrgId)
  }, [activeOrgId, isSuperAdmin])

  useEffect(() => {
    if (!selectedBotId || (isSuperAdmin && !activeOrgId)) return
    void loadBotDetail(selectedBotId)
    void loadDomains(selectedBotId)
    void loadJobs(selectedBotId)
    void loadSources(selectedBotId)
    setIndexStatus(null)
    setActiveCrawlUrl('')
  }, [selectedBotId, activeOrgId])

  useEffect(() => {
    if (!selectedBot || !activeCrawlUrl) return
    if (indexStatus?.stage && terminalStages.has(indexStatus.stage)) return
    const timer = window.setInterval(() => {
      void refreshStatus(activeCrawlUrl)
    }, 4000)
    return () => window.clearInterval(timer)
  }, [selectedBot, activeCrawlUrl, indexStatus?.stage])

  const value: DashboardData = {
    user,
    logout,
    bots,
    selectedBotId,
    selectedBot,
    domains,
    jobs,
    sources,
    indexStatus,
    loading,
    botsLoadedOnce,
    error,
    setError: setErrorSafe,
    newBotName,
    setNewBotName,
    newDomain,
    setNewDomain,
    crawlUrl,
    setCrawlUrl,
    activeCrawlUrl,
    orgs,
    activeOrgId,
    setActiveOrgId,
    orgMembers,
    newOrgName,
    setNewOrgName,
    orgDisplayName,
    orgDisplayNameInput,
    setOrgDisplayNameInput,
    newMemberEmail,
    setNewMemberEmail,
    newMemberRole,
    setNewMemberRole,
    isSuperAdmin,
    embedSnippet,
    buildEmbedSnippet,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    loadBots,
    loadBotDetail,
    loadDomains,
    loadJobs,
    loadSources,
    createSource,
    uploadPdfSources,
    uploadTextSources,
    uploadDocsSources,
    deleteSource,
    createBot,
    loadOrgs,
    loadSelfOrgs,
    loadOrgInfo,
    saveOrgName,
    createOrg,
    setOrgStatus,
    loadOrgMembers,
    addOrgMember,
    clearGcs,
    clearRag,
    addDomain,
    verifyDomain,
    startCrawl,
    startCrawlForSource,
    syncSource,
    updateSourceSyncSettings,
    queueCrawlUrls,
    cancelCrawl,
    cancelIndexJob,
    refreshStatus,
    getJobStatus,
    copySnippet,
    refreshAll,
    setSelectedBotId,
    discoverUrls,
    startBackgroundDiscovery,
    cancelDiscoveryJob,
    listDiscoveryJobs,
    getDiscoveryJob,
    listBookingLinkJobs,
    getBookingLinkJob,
    getLatestJobPipeline,
    resumeJobPipeline,
    startAvailabilityJob,
    listAvailabilityJobs,
    getAvailabilityJob,
    getAvailabilityRaw,
    deleteBot,
    renameBot,
    listConversations,
    searchConversations,
    exportConversationsCsv,
    getConversation,
    endConversation,
    takeOverConversation,
    recomputeAnalytics,
    getAnalyticsSummary,
    getAnalyticsTimeseries,
    getAnalyticsTopSources,
    getEscalationConfig,
    saveEscalationConfig,
    getEscalationCounts,
    listEscalations,
    getEscalationForSession,
    updateEscalationStatus,
    getExtractedTopics,
    extractTopics,
    updateExtractedTopic,
    createExtractedTopic,
    deleteExtractedTopic,
    getTopicUsageSummary,
    computeTopicMappings,
    getTopicQuestions,
    syncUrlBankTopics,
    generateSuggestedMessages,
    generatingSuggestions,
    fetchPlatformSuggestedMessages,
    fetchPlatformConfig,
  }

  return React.createElement(DashboardDataContext.Provider, { value }, children)
}

export function useDashboardData() {
  const context = useContext(DashboardDataContext)
  if (!context) {
    throw new Error('useDashboardData must be used within DashboardDataProvider')
  }
  return context
}
