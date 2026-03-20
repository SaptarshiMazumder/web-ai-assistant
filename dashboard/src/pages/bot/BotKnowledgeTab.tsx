import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Trash2 } from 'lucide-react'
import {
  useDashboardData,
  type AvailabilityJobRecord,
  type BookingLinkJobRecord,
  type JobPipelineRunRecord,
} from '../../hooks/useDashboardData'
import {
  buildUnifiedSourcesProgress,
  type AdditionalSourcesJobStatus,
  type AdditionalSourcesProgressRun,
  type UnifiedSourcesProgress,
} from './sourcesProgressModel'
import {
  categorizeUrls,
  getAllExpandablePaths,
  getAllUrlsFromCategory,
  getCategoryDisplayPath,
  getCategoryUrlCount,
  getNormalizedUrlKey,
  type UrlCategory,
} from '../createBot/urlCategorizer'
import {
  AnimatedPage,
  GlassCard,
} from '../../components/ui'
import {
  clearAdditionalSourcesRun,
  type AdditionalSourcesRunPayload,
  isAdditionalSourcesStageTerminal,
  readAdditionalSourcesRun,
} from './additionalSourcesRun'
import { useTranslation } from 'react-i18next'

/** Jobs not updated in this long are considered stale (e.g. server was killed) and not shown as in-progress. */
const STALE_JOB_MS = 10 * 60 * 1000
const SOURCES_PROGRESS_SNAPSHOT_TTL_MS = 10 * 60 * 1000 // 10 minutes
const SOURCES_PROGRESS_SNAPSHOT_STORAGE_PREFIX = 'dashboard.sources.progress.snapshot.'
const SOURCES_PROGRESS_HIDDEN_RUN_STORAGE_PREFIX = 'dashboard.sources.progress.hidden_run.'

function isJobStale(job: { updated_at?: string | null }): boolean {
  const updated = job.updated_at ? new Date(job.updated_at).getTime() : 0
  return Date.now() - updated > STALE_JOB_MS
}

function formatRelativeTime(iso: string): string {
  const d = new Date(iso)
  const diff = Date.now() - d.getTime()
  const s = Math.floor(diff / 1000)
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  const days = Math.floor(h / 24)
  if (days > 0) return `${days} day${days === 1 ? '' : 's'} ago`
  if (h > 0) return `${h} hour${h === 1 ? '' : 's'} ago`
  if (m > 0) return `${m} min ago`
  if (s > 10) return `${s} sec ago`
  return 'Just now'
}

const SOURCES_JOB_TERMINAL_STAGES = new Set(['done', 'complete', 'error', 'failed', 'cancelled'])
const SOURCES_JOB_POST_CRAWL_STAGES = new Set(['import_submitted', 'prompt_queued', 'prompt_generating'])

const BOOKING_LINK_JOB_TERMINAL_STATUS = new Set(['done', 'failed', 'error'])

type ReservationPlatformConfigItem = {
  id: string
  widget_key: string
  domain_key: string
  label: string
  url_placeholder?: string
}

type JobPipelineWorkflowConfig = {
  workflowId: string
  default: string[]
  platformOverrides: Record<string, string[]>
}

function normalizeJobIdList(raw: unknown): string[] {
  if (!Array.isArray(raw)) return []
  const cleaned = raw
    .map((stepId) => String(stepId || '').trim())
    .filter(Boolean)
  return Array.from(new Set(cleaned))
}

function resolveWidgetPlatformId(
  widgetConfig: Record<string, unknown> | null | undefined,
  platforms: ReservationPlatformConfigItem[],
): string {
  if (!widgetConfig || typeof widgetConfig !== 'object') return ''

  const explicitPlatform = String(widgetConfig.reservationPlatform || '').trim().toLowerCase()
  if (explicitPlatform) return explicitPlatform

  for (const linksKey of ['reservation_links', 'reservationLinks']) {
    const rawLinks = widgetConfig[linksKey]
    if (!rawLinks || typeof rawLinks !== 'object' || Array.isArray(rawLinks)) continue
    for (const [platformId, rawUrl] of Object.entries(rawLinks as Record<string, unknown>)) {
      if (!String(rawUrl || '').trim()) continue
      const normalizedPlatformId = String(platformId || '').trim().toLowerCase()
      if (normalizedPlatformId) return normalizedPlatformId
    }
  }

  for (const platform of platforms) {
    const widgetKey = String(platform.widget_key || '').trim()
    if (!widgetKey) continue
    const rawUrl = String(widgetConfig[widgetKey] || '').trim()
    if (!rawUrl) continue
    const normalizedPlatformId = String(platform.id || '').trim().toLowerCase()
    if (normalizedPlatformId) return normalizedPlatformId
  }

  return ''
}

function bookingStatusLabel(status?: string): string {
  const s = (status || '').toLowerCase()
  if (s === 'done') return '✓ Booking links ready'
  if (s === 'failed' || s === 'error') return 'Booking link extraction failed'
  if (s === 'running' || s === 'queued') return 'Booking links in progress…'
  return 'Booking links…'
}

const AVAILABILITY_TERMINAL_STATUS = new Set(['done', 'failed', 'error'])

type BookingLinkEntry = {
  url: string
  confidence?: number
  reasons?: string[]
  sources?: string[]
  snippets?: string[]
}

type SourcesTrainingStatusSnapshot = {
  stage?: string
  pages_crawled?: number
  docs_count?: number
  last_error?: string
  updated_at?: string
  hostname?: string
}

function sameSourcesTrainingStatus(
  a: SourcesTrainingStatusSnapshot | null,
  b: SourcesTrainingStatusSnapshot | null
): boolean {
  if (a === b) return true
  if (!a || !b) return false
  return (
    (a.stage || '') === (b.stage || '') &&
    Number(a.pages_crawled || 0) === Number(b.pages_crawled || 0) &&
    Number(a.docs_count || 0) === Number(b.docs_count || 0) &&
    (a.last_error || '') === (b.last_error || '') &&
    (a.updated_at || '') === (b.updated_at || '') &&
    (a.hostname || '') === (b.hostname || '')
  )
}

type AdditionalSourcesJobStatusSnapshot = {
  job_id: string
  stage?: string
  hostname?: string
  last_error?: string
  updated_at?: string
}

function sameAdditionalSourcesJobStatuses(
  a: AdditionalSourcesJobStatusSnapshot[] | null,
  b: AdditionalSourcesJobStatusSnapshot[] | null
): boolean {
  if (a === b) return true
  if (!a || !b) return false
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    const lhs = a[i]
    const rhs = b[i]
    if (!lhs || !rhs) return false
    if (lhs.job_id !== rhs.job_id) return false
    if ((lhs.stage || '') !== (rhs.stage || '')) return false
    if ((lhs.hostname || '') !== (rhs.hostname || '')) return false
    if ((lhs.last_error || '') !== (rhs.last_error || '')) return false
    if ((lhs.updated_at || '') !== (rhs.updated_at || '')) return false
  }
  return true
}

type SourcesProgressSnapshotPayload = {
  saved_at: number
  progress: UnifiedSourcesProgress
}

export default function BotKnowledgeTab() {
  const { t } = useTranslation()
  const { botId } = useParams()
  const {
    selectedBot,
    jobs,
    sources,
    loading,
    loadJobs,
    loadSources,
    deleteSource,
    queueCrawlUrls,
    cancelIndexJob,
    discoverUrls,
    getJobStatus,
    syncSource,
    updateSourceSyncSettings,
    listBookingLinkJobs,
    getBookingLinkJob,
    getLatestJobPipeline,
    resumeJobPipeline,
    startAvailabilityJob,
    getAvailabilityJob,
    saveWidgetConfig,
    selectedBotWidgetConfig,
    fetchPlatformConfig,
  } = useDashboardData()

  const allowKnowledgeDiscovery = !(
    selectedBotWidgetConfig &&
    typeof selectedBotWidgetConfig === 'object' &&
    (selectedBotWidgetConfig as Record<string, unknown>).contentHosting === 'shared'
  )

  const [deletingSourceId, setDeletingSourceId] = useState<string | null>(null)
  const [sourcesSelected, setSourcesSelected] = useState<Set<string>>(new Set())
  const [deletingSelectedSources, setDeletingSelectedSources] = useState(false)
  const [stoppingTraining, setStoppingTraining] = useState(false)

  // Sync state
  const [syncingSourceIds, setSyncingSourceIds] = useState<Set<string>>(new Set())
  const [syncSettingsSourceId, setSyncSettingsSourceId] = useState<string | null>(null)
  const [syncSettingsFrequency, setSyncSettingsFrequency] = useState('daily')
  const [syncSettingsHour, setSyncSettingsHour] = useState('09')
  const [syncSettingsMinute, setSyncSettingsMinute] = useState('00')
  const [savingSyncSettings, setSavingSyncSettings] = useState(false)
  const [syncingSelected, setSyncingSelected] = useState(false)
  const syncPopupRef = useRef<HTMLDivElement | null>(null)

  // Discovery UI (Add more pages)
  const [discoverInputUrl, setDiscoverInputUrl] = useState('')
  const discoveryMethod: 'auto' = 'auto'
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedDiscovered, setSelectedDiscovered] = useState<Set<string>>(new Set())
  const [trainingDiscovered, setTrainingDiscovered] = useState(false)
  const [discoveryError, setDiscoveryError] = useState<string | null>(null)
  const [discoveryErrorType, setDiscoveryErrorType] = useState<'error' | 'warning' | null>(null)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())
  const [discoverTrainingJobId, setDiscoverTrainingJobId] = useState<string | null>(null)
  const [discoverTrainingSuccess, setDiscoverTrainingSuccess] = useState(false)

  const [sourcesTrainingJobId, setSourcesTrainingJobId] = useState<string | null>(null)
  const [sourcesTrainingStatus, setSourcesTrainingStatus] = useState<SourcesTrainingStatusSnapshot | null>(null)
  const [additionalSourcesRun, setAdditionalSourcesRun] = useState<AdditionalSourcesRunPayload | null>(null)
  const [additionalSourcesStatuses, setAdditionalSourcesStatuses] = useState<AdditionalSourcesJobStatusSnapshot[] | null>(null)

  const [bookingLinkJob, setBookingLinkJob] = useState<BookingLinkJobRecord | null>(null)
  const [jobPipelineRun, setJobPipelineRun] = useState<JobPipelineRunRecord | null>(null)
  const [jobPipelineHydrated, setJobPipelineHydrated] = useState(false)
  const [hiddenProgressRunKey, setHiddenProgressRunKey] = useState<string | null>(null)
  const [progressDisplayHydrated, setProgressDisplayHydrated] = useState(false)
  const [persistedSourcesProgress, setPersistedSourcesProgress] = useState<UnifiedSourcesProgress | null>(null)
  const [persistedSourcesProgressSavedAt, setPersistedSourcesProgressSavedAt] = useState(0)

  const [allowRealtimeAvailability, setAllowRealtimeAvailability] = useState(false)
  const [bookingTestUrl, setBookingTestUrl] = useState('')

  const [platforms, setPlatforms] = useState<ReservationPlatformConfigItem[]>([])
  const [jobPipelineWorkflowConfig, setJobPipelineWorkflowConfig] = useState<JobPipelineWorkflowConfig>({
    workflowId: 'default',
    default: [],
    platformOverrides: {},
  })

  const selectedBotId = selectedBot?.bot_id || null
  const getJobStatusRef = useRef(getJobStatus)
  const loadJobsRef = useRef(loadJobs)
  const loadSourcesRef = useRef(loadSources)
  const listBookingLinkJobsRef = useRef(listBookingLinkJobs)
  const getBookingLinkJobRef = useRef(getBookingLinkJob)
  const getLatestJobPipelineRef = useRef(getLatestJobPipeline)
  const getAvailabilityJobRef = useRef(getAvailabilityJob)
  const fetchPlatformConfigRef = useRef(fetchPlatformConfig)

  useEffect(() => { getJobStatusRef.current = getJobStatus }, [getJobStatus])
  useEffect(() => { loadJobsRef.current = loadJobs }, [loadJobs])
  useEffect(() => { loadSourcesRef.current = loadSources }, [loadSources])
  useEffect(() => { listBookingLinkJobsRef.current = listBookingLinkJobs }, [listBookingLinkJobs])
  useEffect(() => { getBookingLinkJobRef.current = getBookingLinkJob }, [getBookingLinkJob])
  useEffect(() => { getLatestJobPipelineRef.current = getLatestJobPipeline }, [getLatestJobPipeline])
  useEffect(() => { getAvailabilityJobRef.current = getAvailabilityJob }, [getAvailabilityJob])
  useEffect(() => { fetchPlatformConfigRef.current = fetchPlatformConfig }, [fetchPlatformConfig])

  useEffect(() => {
    setJobPipelineHydrated(false)
  }, [selectedBotId])

  useEffect(() => {
    if (!selectedBotId) {
      setAdditionalSourcesRun(null)
      setAdditionalSourcesStatuses(null)
      return
    }
    const run = readAdditionalSourcesRun(selectedBotId)
    setAdditionalSourcesRun(run)
    if (!run) setAdditionalSourcesStatuses(null)
  }, [selectedBotId])

  useEffect(() => {
    setProgressDisplayHydrated(false)
    setPersistedSourcesProgress(null)
    setPersistedSourcesProgressSavedAt(0)
    setHiddenProgressRunKey(null)
    if (!selectedBotId) {
      setProgressDisplayHydrated(true)
      return
    }

    try {
      const hiddenKey = `${SOURCES_PROGRESS_HIDDEN_RUN_STORAGE_PREFIX}${selectedBotId}`
      const hidden = window.localStorage.getItem(hiddenKey)
      setHiddenProgressRunKey(hidden || null)
    } catch {
      setHiddenProgressRunKey(null)
    }

    try {
      const snapshotKey = `${SOURCES_PROGRESS_SNAPSHOT_STORAGE_PREFIX}${selectedBotId}`
      const raw = window.localStorage.getItem(snapshotKey)
      if (!raw) return
      const parsed = JSON.parse(raw) as Partial<SourcesProgressSnapshotPayload> | null
      const savedAt = Number(parsed?.saved_at || 0)
      const progress = parsed?.progress as UnifiedSourcesProgress | undefined
      if (!savedAt || !progress || typeof progress !== 'object') return
      setPersistedSourcesProgress(progress)
      setPersistedSourcesProgressSavedAt(savedAt)
    } catch {
      setPersistedSourcesProgress(null)
      setPersistedSourcesProgressSavedAt(0)
    }
    setProgressDisplayHydrated(true)
  }, [selectedBotId])

  useEffect(() => {
    let cancelled = false
    const loadPlatformConfig = async () => {
      const result = await fetchPlatformConfigRef.current('en')
      if (cancelled) return
      setPlatforms(result.platforms)
      setJobPipelineWorkflowConfig({
        workflowId: String(result.jobPipelineWorkflow?.workflowId || 'default').trim() || 'default',
        default: normalizeJobIdList(result.jobPipelineWorkflow?.default),
        platformOverrides: Object.fromEntries(
          Object.entries(result.jobPipelineWorkflow?.platformOverrides || {}).map(([platformId, rawSteps]) => [
            String(platformId || '').trim().toLowerCase(),
            normalizeJobIdList(rawSteps),
          ])
        ),
      })
    }
    void loadPlatformConfig()
    return () => {
      cancelled = true
    }
  }, [])
  const [availabilityTestJob, setAvailabilityTestJob] = useState<AvailabilityJobRecord | null>(null)
  const [availabilityTestError, setAvailabilityTestError] = useState<string | null>(null)
  const [availabilityTestRunning, setAvailabilityTestRunning] = useState(false)
  const availabilityTestPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const discoverSuccessTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const DISCOVER_TERMINAL_STAGES = new Set(['done', 'error', 'cancelled', 'import_submitted'])
  const DISCOVER_SUCCESS_STAGES = new Set(['done', 'import_submitted'])

  useEffect(() => {
    if (!allowKnowledgeDiscovery) return
    if (!selectedBotId || !discoverTrainingJobId) return
    let cancelled = false
    const poll = async () => {
      const status = await getJobStatusRef.current(selectedBotId, discoverTrainingJobId)
      if (cancelled || !status) return
      if (status.stage && DISCOVER_TERMINAL_STAGES.has(status.stage)) {
        void loadSourcesRef.current(selectedBotId)
        void loadJobsRef.current(selectedBotId)
        setDiscoverTrainingJobId(null)
        if (status.stage && DISCOVER_SUCCESS_STAGES.has(status.stage)) {
          setDiscoverTrainingSuccess(true)
          if (discoverSuccessTimeoutRef.current) window.clearTimeout(discoverSuccessTimeoutRef.current)
          discoverSuccessTimeoutRef.current = window.setTimeout(() => {
            setDiscoverTrainingSuccess(false)
            discoverSuccessTimeoutRef.current = null
          }, 3000)
        } else {
          setDiscoverTrainingSuccess(false)
        }
      }
    }
    void poll()
    const timer = window.setInterval(poll, 4000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
      if (discoverSuccessTimeoutRef.current) {
        window.clearTimeout(discoverSuccessTimeoutRef.current)
        discoverSuccessTimeoutRef.current = null
      }
    }
  }, [allowKnowledgeDiscovery, selectedBotId, discoverTrainingJobId])

  // Initialize booking settings from widget config
  useEffect(() => {
    const cfg = selectedBotWidgetConfig
    if (cfg && typeof cfg === 'object') {
      const allow = cfg.allowRealtimeAvailability
      const url = cfg.bookingTestUrl
      if (typeof allow === 'boolean') setAllowRealtimeAvailability(allow)
      if (typeof url === 'string' && url) setBookingTestUrl(url)
    }
  }, [selectedBotWidgetConfig])

  // Poll availability test job when running
  useEffect(() => {
    if (!selectedBotId || !availabilityTestJob) return
    if (AVAILABILITY_TERMINAL_STATUS.has(availabilityTestJob.status)) {
      setAvailabilityTestRunning(false)
      return
    }
    setAvailabilityTestRunning(true)
    availabilityTestPollRef.current = setInterval(async () => {
      const updated = await getAvailabilityJobRef.current(selectedBotId, availabilityTestJob.job_id)
      if (updated) {
        setAvailabilityTestJob(updated)
        if (AVAILABILITY_TERMINAL_STATUS.has(updated.status)) {
          if (availabilityTestPollRef.current) {
            clearInterval(availabilityTestPollRef.current)
            availabilityTestPollRef.current = null
          }
          setAvailabilityTestRunning(false)
        }
      }
    }, 4000)
    return () => {
      if (availabilityTestPollRef.current) {
        clearInterval(availabilityTestPollRef.current)
        availabilityTestPollRef.current = null
      }
    }
  }, [selectedBotId, availabilityTestJob?.job_id, availabilityTestJob?.status])

  // Sources section: detect in-progress index job and poll so we can show training progress bar (ignore stale jobs)
  const activeSourcesJob = useMemo(() => {
    if (!selectedBot || !jobs.length) return null
    const inProgress = jobs.filter(
      (j) => {
        const stage = (j.stage || '').toLowerCase()
        // Exclude terminal stages and post-crawl stages that don't need active polling
        if (SOURCES_JOB_TERMINAL_STAGES.has(stage)) return false
        if (['import_submitted', 'prompt_queued', 'prompt_generating'].includes(stage)) return false
        if (isJobStale(j)) return false
        return true
      }
    )
    if (inProgress.length === 0) return null
    inProgress.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    return inProgress[0] ?? null
  }, [selectedBot, jobs])

  const latestFailedSourcesJob = useMemo(() => {
    if (!jobs.length) return null
    const failed = jobs
      .filter((j) => ['error', 'failed'].includes((j.stage || '').toLowerCase()))
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    return failed[0] ?? null
  }, [jobs])

  useEffect(() => {
    if (!selectedBotId) {
      setSourcesTrainingJobId(null)
      setSourcesTrainingStatus(null)
      return
    }
    if (!activeSourcesJob) {
      // Prevent progress flicker on reload while jobs are still hydrating from API.
      const clearTimer = window.setTimeout(() => {
        setSourcesTrainingJobId(null)
        setSourcesTrainingStatus(null)
      }, 6000)
      return () => window.clearTimeout(clearTimer)
    }
    const jobId = activeSourcesJob.job_id
    setSourcesTrainingJobId(jobId)
    const initialStatus: SourcesTrainingStatusSnapshot = {
      stage: activeSourcesJob.stage,
      pages_crawled: activeSourcesJob.pages_crawled,
      docs_count: activeSourcesJob.docs_count,
      last_error: activeSourcesJob.last_error,
      updated_at: activeSourcesJob.updated_at,
      hostname: activeSourcesJob.hostname,
    }
    setSourcesTrainingStatus((prev) => (sameSourcesTrainingStatus(prev, initialStatus) ? prev : initialStatus))
    let cancelled = false
    const poll = async () => {
      const status = await getJobStatusRef.current(selectedBotId, jobId)
      if (cancelled || !status) return
      const nextStatus: SourcesTrainingStatusSnapshot = {
        stage: status.stage,
        pages_crawled: status.pages_crawled,
        docs_count: status.docs_count,
        last_error: status.last_error,
        updated_at: status.updated_at,
        hostname: status.hostname,
      }
      setSourcesTrainingStatus((prev) => (sameSourcesTrainingStatus(prev, nextStatus) ? prev : nextStatus))
      if (status.stage && SOURCES_JOB_TERMINAL_STAGES.has(status.stage.toLowerCase())) {
        setSourcesTrainingJobId(null)
        setSourcesTrainingStatus(null)
        void loadJobsRef.current(selectedBotId)
        void loadSourcesRef.current(selectedBotId)
      }
    }
    const timer = setInterval(poll, 4000)
    void poll()
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [selectedBotId, activeSourcesJob?.job_id])

  useEffect(() => {
    if (!selectedBotId || !additionalSourcesRun) {
      setAdditionalSourcesStatuses(null)
      return
    }
    if (additionalSourcesRun.bot_id !== selectedBotId) {
      clearAdditionalSourcesRun(additionalSourcesRun.bot_id)
      setAdditionalSourcesRun(null)
      setAdditionalSourcesStatuses(null)
      return
    }
    const jobIds = Array.from(
      new Set(
        (additionalSourcesRun.job_ids || [])
          .map((jobId) => String(jobId || '').trim())
          .filter(Boolean)
      )
    )
    if (jobIds.length === 0) {
      clearAdditionalSourcesRun(selectedBotId)
      setAdditionalSourcesRun(null)
      setAdditionalSourcesStatuses(null)
      return
    }

    let cancelled = false
    const poll = async () => {
      const statusResults = await Promise.all(
        jobIds.map((jobId) => getJobStatusRef.current(selectedBotId, jobId))
      )
      if (cancelled) return
      const nextStatuses = jobIds.map((jobId, index) => {
        const status = statusResults[index]
        return {
          job_id: jobId,
          stage: String(status?.stage || '').trim() || undefined,
          hostname: String(status?.hostname || '').trim() || undefined,
          last_error: String(status?.last_error || '').trim() || undefined,
          updated_at: String(status?.updated_at || '').trim() || undefined,
        }
      })
      setAdditionalSourcesStatuses((prev) =>
        sameAdditionalSourcesJobStatuses(prev, nextStatuses) ? prev : nextStatuses
      )
    }

    const timer = setInterval(() => {
      void poll()
    }, 4000)
    void poll()
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [selectedBotId, additionalSourcesRun?.run_id])

  const additionalRunTerminal = useMemo(() => {
    if (!additionalSourcesRun || !additionalSourcesStatuses) return false
    const statusById = new Map(additionalSourcesStatuses.map((item) => [item.job_id, item]))
    const trackedIds = Array.from(
      new Set(
        (additionalSourcesRun.job_ids || [])
          .map((jobId) => String(jobId || '').trim())
          .filter(Boolean)
      )
    )
    if (trackedIds.length === 0) return false
    return trackedIds.every((jobId) => {
      const status = statusById.get(jobId)
      if (!status) return false
      return isAdditionalSourcesStageTerminal(status.stage)
    })
  }, [additionalSourcesRun, additionalSourcesStatuses])

  useEffect(() => {
    if (!selectedBotId || !additionalSourcesRun || !additionalRunTerminal) return
    const currentRunId = additionalSourcesRun.run_id
    const timer = window.setTimeout(() => {
      const persisted = readAdditionalSourcesRun(selectedBotId)
      if (!persisted || persisted.run_id !== currentRunId) return
      clearAdditionalSourcesRun(selectedBotId)
      setAdditionalSourcesRun(null)
      setAdditionalSourcesStatuses(null)
    }, 12000)
    return () => window.clearTimeout(timer)
  }, [selectedBotId, additionalRunTerminal, additionalSourcesRun?.run_id])

  const additionalSourcesAnchorJobId = useMemo(() => {
    if (!additionalSourcesRun) return null
    const trackedIds = Array.from(
      new Set(
        (additionalSourcesRun.job_ids || [])
          .map((jobId) => String(jobId || '').trim())
          .filter(Boolean)
      )
    )
    if (trackedIds.length === 0) return null

    const statusById = new Map(additionalSourcesStatuses?.map((item) => [item.job_id, item]) || [])
    const jobById = new Map(jobs.map((job) => [job.job_id, job]))

    for (const jobId of trackedIds) {
      const stage = String(statusById.get(jobId)?.stage || '').trim().toLowerCase()
      const hostname = String(statusById.get(jobId)?.hostname || jobById.get(jobId)?.hostname || '').trim().toLowerCase()
      const inProgress = stage && !SOURCES_JOB_TERMINAL_STAGES.has(stage) && !SOURCES_JOB_POST_CRAWL_STAGES.has(stage)
      const isCrawlLike = hostname && hostname !== 'text.local' && hostname !== 'pdf.local' && hostname !== 'docs.local'
      if (inProgress && isCrawlLike) return jobId
    }

    for (const jobId of trackedIds) {
      const hostname = String(statusById.get(jobId)?.hostname || jobById.get(jobId)?.hostname || '').trim().toLowerCase()
      const isCrawlLike = hostname && hostname !== 'text.local' && hostname !== 'pdf.local' && hostname !== 'docs.local'
      if (isCrawlLike) return jobId
    }

    return null
  }, [additionalSourcesRun, additionalSourcesStatuses, jobs])

  const pipelineMatchJobId = additionalSourcesRun ? additionalSourcesAnchorJobId : sourcesTrainingJobId


  // Booking link extraction (RAG) progress
  useEffect(() => {
    if (!selectedBotId) {
      setBookingLinkJob(null)
      return
    }
    let cancelled = false
    let pollTimer: ReturnType<typeof setInterval> | null = null

    const load = async () => {
      const jobs = await listBookingLinkJobsRef.current(selectedBotId)
      if (cancelled) return
      if (!jobs.length) {
        setBookingLinkJob(null)
        return
      }
      const latest = jobs[0]
      setBookingLinkJob(latest)
      const isTerminal = BOOKING_LINK_JOB_TERMINAL_STATUS.has((latest.status || '').toLowerCase())
      const stale = isJobStale({ updated_at: latest.updated_at })
      if (isTerminal || stale) return

      pollTimer = setInterval(async () => {
        const current = await getBookingLinkJobRef.current(selectedBotId, latest.job_id)
        if (cancelled || !current) return
        setBookingLinkJob(current)
        const terminalNow = BOOKING_LINK_JOB_TERMINAL_STATUS.has((current.status || '').toLowerCase())
        if (terminalNow && pollTimer) {
          clearInterval(pollTimer)
          pollTimer = null
        }
      }, 4000)
    }

    void load()
    return () => {
      cancelled = true
      if (pollTimer) clearInterval(pollTimer)
    }
  }, [selectedBotId])

  // Config-first job pipeline progress
  useEffect(() => {
    if (!selectedBotId) {
      setJobPipelineRun(null)
      setJobPipelineHydrated(false)
      return
    }
    let cancelled = false
    let pollTimer: ReturnType<typeof setInterval> | null = null

    const normalizeStatus = (run: JobPipelineRunRecord | null): string =>
      String(run?.status || '').trim().toLowerCase()

    const activePipelineJobId = String(pipelineMatchJobId || '').trim()
    const hasAdditionalSourcesContext = Boolean(additionalSourcesRun)

    const isTerminalStatus = (run: JobPipelineRunRecord | null): boolean => {
      const status = normalizeStatus(run)
      return status === 'done' || status === 'error' || status === 'paused' || status === 'cancelled'
    }

    const runMatchesActiveCrawl = (run: JobPipelineRunRecord | null): boolean => {
      if (!run || !activePipelineJobId) return Boolean(run)
      const context = run.context && typeof run.context === 'object' ? run.context : {}
      const runIndexJobId = String((context as Record<string, unknown>).index_job_id || '').trim()
      if (!runIndexJobId) return false
      return runIndexJobId === activePipelineJobId
    }

    const shouldKeepPolling = (run: JobPipelineRunRecord | null): boolean => {
      if (activePipelineJobId) {
        // During crawl/import, keep polling until the pipeline run for this crawl appears and reaches a terminal state.
        if (!run) return true
        if (!runMatchesActiveCrawl(run)) return true
        return !isTerminalStatus(run)
      }
      if (hasAdditionalSourcesContext) return false
      if (!run) return false
      return !isTerminalStatus(run)
    }

    const pollLatest = async () => {
      const run = await getLatestJobPipelineRef.current(selectedBotId)
      if (cancelled) return
      setJobPipelineHydrated(true)
      if (activePipelineJobId) {
        setJobPipelineRun(runMatchesActiveCrawl(run) ? run : null)
      } else if (hasAdditionalSourcesContext) {
        setJobPipelineRun(null)
      } else {
        setJobPipelineRun(run)
      }
      if (!shouldKeepPolling(run) && pollTimer) {
        clearInterval(pollTimer)
        pollTimer = null
      }
    }

    pollTimer = setInterval(() => {
      void pollLatest()
    }, 4000)
    void pollLatest()

    return () => {
      cancelled = true
      if (pollTimer) clearInterval(pollTimer)
    }
  }, [selectedBotId, pipelineMatchJobId, additionalSourcesRun?.run_id])

  const handleResumeJobPipeline = useCallback(async () => {
    if (!selectedBot || !jobPipelineRun || (jobPipelineRun.status || '').toLowerCase() !== 'paused') return
    const resumed = await resumeJobPipeline(selectedBot.bot_id, jobPipelineRun.run_id)
    if (resumed) setJobPipelineRun(resumed)
  }, [selectedBot, jobPipelineRun, resumeJobPipeline])

  const plannedPipelineJobIds = useMemo(() => {
    const fallbackSteps = normalizeJobIdList(jobPipelineWorkflowConfig.default)
    const platformId = resolveWidgetPlatformId(selectedBotWidgetConfig, platforms)
    if (!platformId) return fallbackSteps
    const overrideSteps = normalizeJobIdList(jobPipelineWorkflowConfig.platformOverrides[platformId])
    return overrideSteps.length > 0 ? overrideSteps : fallbackSteps
  }, [jobPipelineWorkflowConfig.default, jobPipelineWorkflowConfig.platformOverrides, selectedBotWidgetConfig, platforms])

  const additionalSourcesProgressRun = useMemo<AdditionalSourcesProgressRun | null>(() => {
    if (!additionalSourcesRun) return null
    const runId = String(additionalSourcesRun.run_id || '').trim()
    const jobIds = Array.from(
      new Set(
        (additionalSourcesRun.job_ids || [])
          .map((jobId) => String(jobId || '').trim())
          .filter(Boolean)
      )
    )
    if (!runId || jobIds.length === 0) return null
    const totalSourcesRaw = Number(additionalSourcesRun.total_sources || 0)
    const totalSources = Number.isFinite(totalSourcesRaw) && totalSourcesRaw > 0
      ? Math.round(totalSourcesRaw)
      : jobIds.length
    return {
      runId,
      jobIds,
      totalSources: Math.max(1, totalSources),
    }
  }, [additionalSourcesRun])

  const additionalSourcesProgressStatuses = useMemo<AdditionalSourcesJobStatus[] | null>(() => {
    if (!additionalSourcesStatuses || additionalSourcesStatuses.length === 0) return null
    return additionalSourcesStatuses.map((status) => ({
      job_id: status.job_id,
      stage: status.stage,
      hostname: status.hostname,
      last_error: status.last_error,
      updated_at: status.updated_at,
    }))
  }, [additionalSourcesStatuses])

  const liveUnifiedSourcesProgress = useMemo(
    () =>
      buildUnifiedSourcesProgress({
        crawlJobId: pipelineMatchJobId,
        crawlStatus: sourcesTrainingStatus,
        pipelineRun: jobPipelineRun,
        plannedPipelineJobIds,
        additionalSourcesRun: additionalSourcesProgressRun,
        additionalSourcesStatuses: additionalSourcesProgressStatuses,
      }),
    [
      pipelineMatchJobId,
      sourcesTrainingStatus,
      jobPipelineRun,
      plannedPipelineJobIds,
      additionalSourcesProgressRun,
      additionalSourcesProgressStatuses,
    ]
  )

  useEffect(() => {
    if (!selectedBotId || !liveUnifiedSourcesProgress) return
    const savedAt = Date.now()
    setPersistedSourcesProgress(liveUnifiedSourcesProgress)
    setPersistedSourcesProgressSavedAt(savedAt)
    try {
      const snapshotKey = `${SOURCES_PROGRESS_SNAPSHOT_STORAGE_PREFIX}${selectedBotId}`
      const payload: SourcesProgressSnapshotPayload = {
        saved_at: savedAt,
        progress: liveUnifiedSourcesProgress,
      }
      window.localStorage.setItem(snapshotKey, JSON.stringify(payload))
    } catch {
      // Best effort only.
    }
  }, [selectedBotId, liveUnifiedSourcesProgress])

  const fallbackUnifiedSourcesProgress = useMemo(() => {
    if (liveUnifiedSourcesProgress) return null
    if (!persistedSourcesProgress || !persistedSourcesProgressSavedAt) return null
    // Never resurrect completed progress that the user already saw/auto-hidden.
    if (hiddenProgressRunKey) return null
    const ageMs = Date.now() - persistedSourcesProgressSavedAt
    if (ageMs > SOURCES_PROGRESS_SNAPSHOT_TTL_MS) return null
    // Crawl-only terminal snapshots are stale and cause misleading reload flashes.
    const isCrawlOnlyTerminal =
      persistedSourcesProgress.isTerminal &&
      persistedSourcesProgress.steps.length === 1 &&
      persistedSourcesProgress.steps[0]?.source === 'crawl'
    if (isCrawlOnlyTerminal) return null
    return persistedSourcesProgress
  }, [liveUnifiedSourcesProgress, persistedSourcesProgress, persistedSourcesProgressSavedAt, hiddenProgressRunKey])

  const unifiedSourcesProgress = liveUnifiedSourcesProgress || fallbackUnifiedSourcesProgress
  const liveUnifiedProgressRunKey = liveUnifiedSourcesProgress?.runKey || null
  const unifiedProgressRunKey = unifiedSourcesProgress?.runKey || null
  const shouldGateTerminalProgressVisibility = Boolean(
    unifiedSourcesProgress?.isTerminal && !progressDisplayHydrated
  )
  // Show progress immediately only when a crawl is actively running/queued (real-time training).
  // Otherwise, wait for pipeline data to load so the runKey is stable and matches the saved hidden key.
  const crawlActivelyRunning = Boolean(
    liveUnifiedSourcesProgress?.steps.some(
      (s) => s.source === 'crawl' && (s.status === 'queued' || s.status === 'running')
    )
  )
  const unifiedProgressVisible = Boolean(
    unifiedSourcesProgress &&
      !shouldGateTerminalProgressVisibility &&
      (crawlActivelyRunning || jobPipelineHydrated) &&
      hiddenProgressRunKey !== unifiedProgressRunKey
  )
  const unifiedProgressInFlight = Boolean(liveUnifiedSourcesProgress && !liveUnifiedSourcesProgress.isTerminal)
  const activeUnifiedStep =
    liveUnifiedSourcesProgress?.steps[liveUnifiedSourcesProgress.activeStepIndex] || null
  const displayUnifiedStep =
    unifiedSourcesProgress?.steps[unifiedSourcesProgress.activeStepIndex] || null
  const activeAdditionalSourcesStats = useMemo(() => {
    if (!displayUnifiedStep) return null
    if (String(displayUnifiedStep.id || '').trim().toLowerCase() !== 'additional_sources') return null
    const details = displayUnifiedStep.details
    if (!details || typeof details !== 'object') return null
    const doneRaw = Number((details as Record<string, unknown>).done_sources || 0)
    const totalRaw = Number((details as Record<string, unknown>).total_sources || 0)
    const done = Number.isFinite(doneRaw) ? Math.max(0, Math.round(doneRaw)) : 0
    const total = Number.isFinite(totalRaw) ? Math.max(0, Math.round(totalRaw)) : 0
    if (total <= 0) return null
    return {
      done: Math.min(done, total),
      total,
    }
  }, [displayUnifiedStep])
  const activeMenuExtractionStats = useMemo(() => {
    if (!displayUnifiedStep) return null
    if (String(displayUnifiedStep.id || '').trim().toLowerCase() !== 'menu_extraction') return null
    const details = displayUnifiedStep.details
    if (!details || typeof details !== 'object') return null
    const foundRaw = Number((details as Record<string, unknown>).assets_discovered || 0)
    const downloadedRaw = Number((details as Record<string, unknown>).assets_downloaded || 0)
    const savedRaw = Number((details as Record<string, unknown>).assets_created || 0)
    const found = Number.isFinite(foundRaw) ? Math.max(0, Math.round(foundRaw)) : 0
    const downloaded = Number.isFinite(downloadedRaw) ? Math.max(0, Math.round(downloadedRaw)) : 0
    const saved = Number.isFinite(savedRaw) ? Math.max(0, Math.round(savedRaw)) : 0
    const prepared = Math.max(saved, downloaded)
    if (found <= 0 && prepared <= 0) return null
    return { found, prepared }
  }, [displayUnifiedStep])
  const unifiedHasPipelineStep = Boolean(
    unifiedSourcesProgress?.steps.some((step) => step.source === 'pipeline')
  )
  const unifiedOnlyCrawlStep = Boolean(
    unifiedSourcesProgress &&
      unifiedSourcesProgress.steps.length === 1 &&
      unifiedSourcesProgress.steps[0]?.source === 'crawl'
  )
  const unifiedCounterReady = Boolean(
    unifiedSourcesProgress && (!unifiedOnlyCrawlStep || jobPipelineHydrated || unifiedHasPipelineStep)
  )
  const crawlStepInFlight = Boolean(
    activeUnifiedStep &&
      activeUnifiedStep.source === 'crawl' &&
      (activeUnifiedStep.status === 'queued' || activeUnifiedStep.status === 'running')
  )
  const progressControlState = liveUnifiedSourcesProgress

  useEffect(() => {
    if (!liveUnifiedProgressRunKey) return
    // Wait for pipeline data before clearing a saved hidden key — the runKey format
    // changes from "crawl:xxx" to "pipeline:xxx" once the async pipeline fetch completes.
    // Clearing before hydration causes the saved key to be destroyed by a transient mismatch.
    if (!jobPipelineHydrated) return
    if (hiddenProgressRunKey && hiddenProgressRunKey !== liveUnifiedProgressRunKey) {
      setHiddenProgressRunKey(null)
      if (selectedBotId) {
        try {
          const hiddenKey = `${SOURCES_PROGRESS_HIDDEN_RUN_STORAGE_PREFIX}${selectedBotId}`
          window.localStorage.removeItem(hiddenKey)
        } catch {
          // ignore storage failures
        }
      }
    }
  }, [liveUnifiedProgressRunKey, hiddenProgressRunKey, selectedBotId, jobPipelineHydrated])

  useEffect(() => {
    if (!liveUnifiedSourcesProgress || !liveUnifiedProgressRunKey) return
    if (hiddenProgressRunKey === liveUnifiedProgressRunKey) return
    if (!liveUnifiedSourcesProgress.isTerminal || liveUnifiedSourcesProgress.isError) return
    if (!jobPipelineHydrated && unifiedOnlyCrawlStep) return
    const timer = window.setTimeout(() => {
      setHiddenProgressRunKey(liveUnifiedProgressRunKey)
      if (selectedBotId) {
        try {
          const hiddenKey = `${SOURCES_PROGRESS_HIDDEN_RUN_STORAGE_PREFIX}${selectedBotId}`
          window.localStorage.setItem(hiddenKey, liveUnifiedProgressRunKey)
        } catch {
          // ignore storage failures
        }
      }
    }, 2400)
    return () => window.clearTimeout(timer)
  }, [
    liveUnifiedSourcesProgress,
    liveUnifiedProgressRunKey,
    hiddenProgressRunKey,
    jobPipelineHydrated,
    unifiedOnlyCrawlStep,
    selectedBotId,
  ])

  const normalizedDiscoverUrl = (discoverInputUrl || '').trim().replace(/\/+$/, '') || undefined
  const urlCategories = useMemo(() => {
    if (!discoveredUrls.length || !normalizedDiscoverUrl) return null
    return categorizeUrls(discoveredUrls, normalizedDiscoverUrl)
  }, [discoveredUrls, normalizedDiscoverUrl])

  useEffect(() => {
    if (!urlCategories) {
      setExpandedCategories(new Set())
      return
    }
    setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
  }, [urlCategories])

  // Refetch jobs/sources when Knowledge tab is shown for a bot so we pick up data from create-bot (queueCrawlUrls may have completed after initial load).
  const lastRefetchedBotIdRef = useRef<string | null>(null)
  useEffect(() => {
    if (!botId || !selectedBot || selectedBot.bot_id !== botId) return
    if (lastRefetchedBotIdRef.current === botId) return
    lastRefetchedBotIdRef.current = botId
    void loadJobsRef.current(botId)
    void loadSourcesRef.current(botId)
  }, [botId, selectedBot?.bot_id])
  useEffect(() => {
    if (!botId) {
      lastRefetchedBotIdRef.current = null
      setEmptyPollCount(0)
    }
  }, [botId])

  // When we have no sources and no jobs (e.g. just came from create-bot and batch is still being created), poll a few times.
  const [emptyPollCount, setEmptyPollCount] = useState(0)
  useEffect(() => {
    if (!botId || !selectedBot || selectedBot.bot_id !== botId) return
    if (sources.length > 0 || jobs.length > 0) return
    if (emptyPollCount >= 5) return
    const t = setTimeout(() => {
      void loadJobsRef.current(botId)
      void loadSourcesRef.current(botId)
      setEmptyPollCount((c) => c + 1)
    }, 2000)
    return () => clearTimeout(t)
  }, [botId, selectedBot?.bot_id, sources.length, jobs.length, emptyPollCount])

  const handleDiscover = useCallback(async () => {
    if (!allowKnowledgeDiscovery) return
    if (!discoverInputUrl.trim() || isDiscovering) return
    setIsDiscovering(true)
    setDiscoveredUrls([])
    setSelectedDiscovered(new Set())
    setDiscoveryError(null)
    setDiscoveryErrorType(null)
    try {
      const result = await discoverUrls(discoverInputUrl.trim(), discoveryMethod, (evt) => {
        if (evt.type === 'discovered' && typeof evt.url === 'string') {
          const url = evt.url
          setDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
          // Clear stale "no results yet" warning once URLs start arriving.
          setDiscoveryError(null)
          setDiscoveryErrorType(null)
        }
        if (evt.type === 'error' && typeof evt.message === 'string') {
          const reason = (evt as { failure_reason?: string }).failure_reason
          if (reason === 'robots_blocked') {
            setDiscoveryError('🚫 This site blocks crawlers via robots.txt. Try adding specific URLs manually.')
            setDiscoveryErrorType('error')
          } else if (reason === 'sitemap_empty') {
            setDiscoveryError("No sitemap found. Discovery uses automatic method.")
            setDiscoveryErrorType('warning')
          } else {
            setDiscoveryError(evt.message)
            setDiscoveryErrorType('error')
          }
        }
        if (evt.type === 'warning' && typeof evt.message === 'string') {
          setDiscoveryError(evt.message)
          setDiscoveryErrorType('warning')
        }
      }, undefined, { max_duration_sec: 90 })
      setDiscoveredUrls(result.urls || [])
      if (result.failureReason === 'no_results') {
        setDiscoveryError('⚠️ We could not discover real pages from this site. Add sources manually or upload files.')
        setDiscoveryErrorType('warning')
      } else if (result.urls?.length === 0) {
        const reason = result.failureReason
        if (reason === 'robots_blocked') {
          setDiscoveryError('🚫 All discovered URLs are blocked by robots.txt')
          setDiscoveryErrorType('error')
        } else if (reason === 'no_results') {
          setDiscoveryError('⚠️ No pages found. Site may be blocking crawlers or have no discoverable links.')
          setDiscoveryErrorType('warning')
        } else if (!discoveryError) {
          setDiscoveryError('No URLs found for this site.')
          setDiscoveryErrorType('warning')
        }
      }
    } catch {
      setDiscoveredUrls([])
      setDiscoveryError('Discovery failed')
      setDiscoveryErrorType('error')
    } finally {
      setIsDiscovering(false)
    }
  }, [allowKnowledgeDiscovery, discoverInputUrl, discoveryMethod, isDiscovering, discoverUrls, discoveryError])

  const toggleDiscovered = useCallback((url: string) => {
    setSelectedDiscovered((prev) => {
      const next = new Set(prev)
      if (next.has(url)) next.delete(url)
      else next.add(url)
      return next
    })
  }, [])

  const allDiscoveredSelected = discoveredUrls.length > 0 && selectedDiscovered.size === discoveredUrls.length
  const toggleAllDiscovered = useCallback(() => {
    if (allDiscoveredSelected) setSelectedDiscovered(new Set())
    else setSelectedDiscovered(new Set(discoveredUrls))
  }, [allDiscoveredSelected, discoveredUrls])

  const expandAllCategories = useCallback(() => {
    if (urlCategories) setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
  }, [urlCategories])
  const collapseAllCategories = useCallback(() => setExpandedCategories(new Set()), [])

  const toggleCategoryExpand = useCallback((path: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const isCategorySelected = useCallback(
    (category: UrlCategory): boolean => {
      const categoryUrls = getAllUrlsFromCategory(category)
      return categoryUrls.length > 0 && categoryUrls.every((url) => selectedDiscovered.has(url))
    },
    [selectedDiscovered]
  )
  const isCategoryPartiallySelected = useCallback(
    (category: UrlCategory): boolean => {
      const categoryUrls = getAllUrlsFromCategory(category)
      const count = categoryUrls.filter((url) => selectedDiscovered.has(url)).length
      return count > 0 && count < categoryUrls.length
    },
    [selectedDiscovered]
  )
  const toggleCategory = useCallback((_path: string, categoryUrls: string[]) => {
    setSelectedDiscovered((prev) => {
      const next = new Set(prev)
      const allSelected = categoryUrls.length > 0 && categoryUrls.every((u) => next.has(u))
      if (allSelected) categoryUrls.forEach((u) => next.delete(u))
      else categoryUrls.forEach((u) => next.add(u))
      return next
    })
  }, [])

  const renderDiscoverCategory = useCallback(
    (category: UrlCategory): React.ReactNode => {
      const categoryUrls = getAllUrlsFromCategory(category)
      const urlCount = getCategoryUrlCount(category)
      const isExpanded = expandedCategories.has(category.path)
      const isSelected = isCategorySelected(category)
      const isPartial = isCategoryPartiallySelected(category)
      const hasChildCategories = category.children.size > 0
      const hasExpandableContent = hasChildCategories || category.urls.length > 0

      return (
        <div key={category.path || 'root'} style={{ marginLeft: `${category.level * 20}px` }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              padding: '8px 0',
              cursor: 'pointer',
              userSelect: 'none',
            }}
          >
            {hasExpandableContent ? (
              <span
                onClick={(e) => {
                  e.stopPropagation()
                  toggleCategoryExpand(category.path)
                }}
                style={{
                  marginRight: '8px',
                  width: '14px',
                  height: '14px',
                  transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: '#64748b',
                }}
                aria-hidden
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 18l6-6-6-6" />
                </svg>
              </span>
            ) : (
              <span style={{ marginRight: '16px', width: '12px' }} />
            )}
            <input
              type="checkbox"
              checked={isSelected}
              onChange={(e) => {
                e.stopPropagation()
                toggleCategory(category.path, categoryUrls)
              }}
              ref={(input) => {
                if (input) input.indeterminate = isPartial
              }}
              style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
            />
            <span
              onClick={() => hasExpandableContent && toggleCategoryExpand(category.path)}
              style={{ flex: 1, cursor: hasExpandableContent ? 'pointer' : 'default' }}
            >
              {getCategoryDisplayPath(category)}
            </span>
            <span
              style={{
                marginLeft: '8px',
                padding: '2px 8px',
                borderRadius: '999px',
                background: 'rgba(246, 180, 109, 0.2)',
                color: '#d97706',
                fontSize: '13px',
                fontWeight: 500,
              }}
            >
              {urlCount} {urlCount === 1 ? 'page' : 'pages'}
            </span>
          </div>
          {hasExpandableContent && isExpanded && (
            <div>
              {Array.from(category.children.values())
                .sort((a, b) => {
                  const countA = getCategoryUrlCount(a)
                  const countB = getCategoryUrlCount(b)
                  if (countA !== countB) return countB - countA
                  return a.name.localeCompare(b.name)
                })
                .map((child) => renderDiscoverCategory(child))}
              {category.urls.length > 0 && (
                <div style={{ marginLeft: '20px', paddingLeft: '20px' }}>
                  {category.urls.map((url) => (
                    <label
                      key={url}
                      className="url-list-item"
                      style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedDiscovered.has(url)}
                        onChange={() => toggleDiscovered(url)}
                        style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
                      />
                      <span style={{ fontSize: '16px', color: '#334155' }}>{url}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )
    },
    [
      expandedCategories,
      isCategorySelected,
      isCategoryPartiallySelected,
      toggleCategoryExpand,
      toggleCategory,
      toggleDiscovered,
      selectedDiscovered,
    ]
  )

  const handleTrainDiscovered = useCallback(async () => {
    if (!allowKnowledgeDiscovery) return
    if (!selectedBot || selectedDiscovered.size === 0 || trainingDiscovered) return
    const urls = Array.from(selectedDiscovered)
    setTrainingDiscovered(true)
    try {
      const jobId = await queueCrawlUrls(selectedBot.bot_id, urls)
      if (jobId) {
        setDiscoverTrainingJobId(jobId)
        setSelectedDiscovered(new Set())
        setDiscoveredUrls([])
        setDiscoverInputUrl('')
      } else {
        await loadJobs(selectedBot.bot_id)
      }
    } finally {
      setTrainingDiscovered(false)
    }
  }, [allowKnowledgeDiscovery, selectedBot, selectedDiscovered, queueCrawlUrls, loadJobs, trainingDiscovered])

  const handleSaveAvailabilitySettings = useCallback(async () => {
    if (!selectedBot) return
    const existing = selectedBotWidgetConfig && typeof selectedBotWidgetConfig === 'object' ? selectedBotWidgetConfig : {}
    const merged = {
      ...existing,
      allowRealtimeAvailability,
      bookingTestUrl: bookingTestUrl.trim() || undefined,
    }
    await saveWidgetConfig(selectedBot.bot_id, merged)
  }, [selectedBot, selectedBotWidgetConfig, allowRealtimeAvailability, bookingTestUrl, saveWidgetConfig])

  const handleRunAvailabilityTest = useCallback(async () => {
    if (!selectedBot || !bookingTestUrl.trim() || availabilityTestRunning) return
    setAvailabilityTestError(null)
    try {
      const existing = selectedBotWidgetConfig && typeof selectedBotWidgetConfig === 'object' ? selectedBotWidgetConfig : {}
      const merged = { ...existing, allowRealtimeAvailability, bookingTestUrl: bookingTestUrl.trim() }
      await saveWidgetConfig(selectedBot.bot_id, merged)
    } catch {
      // Non-blocking; continue with test
    }
    const created = await startAvailabilityJob(selectedBot.bot_id, {
      url: bookingTestUrl.trim(),
      max_seconds: 60,
      question: 'Summarize availability and pricing from the page.',
    })
    if (!created) {
      setAvailabilityTestError('Failed to start availability test')
      return
    }
    setAvailabilityTestJob(created)
  }, [
    selectedBot,
    bookingTestUrl,
    availabilityTestRunning,
    allowRealtimeAvailability,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    startAvailabilityJob,
  ])

  const handleDeleteSource = useCallback(
    async (sourceId: string) => {
      if (!selectedBot || deletingSourceId) return
      setDeletingSourceId(sourceId)
      try {
        await deleteSource(selectedBot.bot_id, sourceId)
        setSourcesSelected((prev) => {
          const next = new Set(prev)
          next.delete(sourceId)
          return next
        })
      } finally {
        setDeletingSourceId(null)
      }
    },
    [selectedBot, deleteSource, deletingSourceId]
  )

  const toggleSourcesSelected = useCallback((sourceId: string) => {
    setSourcesSelected((prev) => {
      const next = new Set(prev)
      if (next.has(sourceId)) next.delete(sourceId)
      else next.add(sourceId)
      return next
    })
  }, [])

  const selectAllSources = useCallback(() => {
    setSourcesSelected(new Set(sources.map((s) => s.source_id)))
  }, [sources])

  const deselectAllSources = useCallback(() => {
    setSourcesSelected(new Set())
  }, [])

  const handleDeleteSelectedSources = useCallback(async () => {
    if (!selectedBot || sourcesSelected.size === 0 || deletingSelectedSources) return
    setDeletingSelectedSources(true)
    try {
      for (const sourceId of sourcesSelected) {
        await deleteSource(selectedBot.bot_id, sourceId)
      }
      setSourcesSelected(new Set())
      void loadSources(selectedBot.bot_id)
      void loadJobs(selectedBot.bot_id)
    } finally {
      setDeletingSelectedSources(false)
    }
  }, [selectedBot, sourcesSelected, deletingSelectedSources, deleteSource, loadSources, loadJobs])

  const handleStopTraining = useCallback(async () => {
    if (!selectedBot || !activeSourcesJob || stoppingTraining) return
    setStoppingTraining(true)
    try {
      const cancelUrl =
        (activeSourcesJob.hostname || '').toLowerCase() === 'batch'
          ? 'https://batch/'
          : (activeSourcesJob.url || `https://${activeSourcesJob.hostname || 'batch'}/`)
      await cancelIndexJob(selectedBot.bot_id, cancelUrl)
    } finally {
      setStoppingTraining(false)
    }
  }, [selectedBot, activeSourcesJob, stoppingTraining, cancelIndexJob])

  const handleSyncSource = useCallback(async (sourceId: string) => {
    if (!selectedBot || syncingSourceIds.has(sourceId)) return
    setSyncingSourceIds((prev) => new Set(prev).add(sourceId))
    try {
      await syncSource(selectedBot.bot_id, sourceId)
    } finally {
      setSyncingSourceIds((prev) => { const next = new Set(prev); next.delete(sourceId); return next })
    }
  }, [selectedBot, syncingSourceIds, syncSource])

  const handleSyncSelected = useCallback(async () => {
    if (!selectedBot || syncingSelected) return
    const urlSourceIds = sources
      .filter((s) => s.type.toLowerCase() === 'url' && sourcesSelected.has(s.source_id))
      .map((s) => s.source_id)
    if (urlSourceIds.length === 0) return
    setSyncingSelected(true)
    try {
      for (const sid of urlSourceIds) {
        setSyncingSourceIds((prev) => new Set(prev).add(sid))
        try {
          await syncSource(selectedBot.bot_id, sid)
        } finally {
          setSyncingSourceIds((prev) => { const next = new Set(prev); next.delete(sid); return next })
        }
      }
    } finally {
      setSyncingSelected(false)
    }
  }, [selectedBot, syncingSelected, sources, sourcesSelected, syncSource])

  const handleOpenSyncSettings = useCallback((source: { source_id: string; sync_frequency?: string; sync_time_utc?: string }) => {
    setSyncSettingsSourceId(source.source_id)
    setSyncSettingsFrequency(source.sync_frequency || 'daily')
    const timeParts = (source.sync_time_utc || '00:00').split(':')
    setSyncSettingsHour(timeParts[0] || '00')
    setSyncSettingsMinute(timeParts[1] || '00')
  }, [])

  const handleSaveSyncSettings = useCallback(async (sourceId: string, enabled: boolean) => {
    if (!selectedBot || savingSyncSettings) return
    setSavingSyncSettings(true)
    try {
      const localOffset = new Date().getTimezoneOffset()
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
      const localH = parseInt(syncSettingsHour, 10)
      const localM = parseInt(syncSettingsMinute, 10)
      const totalMinutes = localH * 60 + localM + localOffset
      const utcMinutes = ((totalMinutes % 1440) + 1440) % 1440
      const utcH = String(Math.floor(utcMinutes / 60)).padStart(2, '0')
      const utcM = String(utcMinutes % 60).padStart(2, '0')
      await updateSourceSyncSettings(selectedBot.bot_id, sourceId, {
        sync_enabled: enabled,
        sync_frequency: syncSettingsFrequency,
        sync_time_utc: `${utcH}:${utcM}`,
        sync_timezone: tz,
      })
      setSyncSettingsSourceId(null)
    } finally {
      setSavingSyncSettings(false)
    }
  }, [selectedBot, savingSyncSettings, syncSettingsFrequency, syncSettingsHour, syncSettingsMinute, updateSourceSyncSettings])

  const handleToggleAutoSync = useCallback(async (source: { source_id: string; sync_enabled?: boolean; sync_frequency?: string; sync_time_utc?: string; sync_timezone?: string }) => {
    if (!selectedBot) return
    const newEnabled = !source.sync_enabled
    if (newEnabled) {
      handleOpenSyncSettings(source)
    } else {
      await updateSourceSyncSettings(selectedBot.bot_id, source.source_id, {
        sync_enabled: false,
        sync_frequency: source.sync_frequency || 'daily',
        sync_time_utc: source.sync_time_utc || '00:00',
        sync_timezone: source.sync_timezone || 'UTC',
      })
    }
  }, [selectedBot, handleOpenSyncSettings, updateSourceSyncSettings])

  // Close sync popup on outside click
  useEffect(() => {
    if (!syncSettingsSourceId) return
    const handler = (e: MouseEvent) => {
      if (syncPopupRef.current && !syncPopupRef.current.contains(e.target as Node)) {
        setSyncSettingsSourceId(null)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [syncSettingsSourceId])

  /** True if at least one selected source is a URL type. */
  const hasSelectedUrlSources = useMemo(() => {
    return sources.some((s) => s.type.toLowerCase() === 'url' && sourcesSelected.has(s.source_id))
  }, [sources, sourcesSelected])

  // Source grouping: URL sources nested by meaningful path, non-URL sources stay flat.
  const urlSources = useMemo(() => sources.filter((s) => s.type.toLowerCase() === 'url'), [sources])
  const nonUrlSources = useMemo(() => sources.filter((s) => s.type.toLowerCase() !== 'url'), [sources])

  const sourceDisplayOrder = useMemo(() => {
    const order = new Map<string, number>()
    sources.forEach((s, idx) => order.set(s.source_id, idx))
    return order
  }, [sources])

  const sourceUrlMap = useMemo(() => {
    const map = new Map<string, typeof sources>()
    for (const s of urlSources) {
      const url =
        typeof (s.config as Record<string, unknown>)?.url === 'string'
          ? ((s.config as Record<string, unknown>).url as string)
          : ''
      if (!url) continue
      const key = getNormalizedUrlKey(url)
      const prev = map.get(key)
      if (prev) prev.push(s)
      else map.set(key, [s])
    }
    return map
  }, [urlSources])

  const sourceCategories = useMemo(() => {
    const urls = urlSources
      .map((s) =>
        typeof (s.config as Record<string, unknown>)?.url === 'string'
          ? ((s.config as Record<string, unknown>).url as string)
          : ''
      )
      .filter(Boolean)
    if (!urls.length) return null
    return categorizeUrls(urls, '')
  }, [urlSources])

  const [expandedSourceGroups, setExpandedSourceGroups] = useState<Set<string>>(new Set())
  useEffect(() => {
    if (!sourceCategories) {
      setExpandedSourceGroups(new Set())
      return
    }
    setExpandedSourceGroups(new Set(getAllExpandablePaths(sourceCategories)))
  }, [sourceCategories])

  const getSourcesForCategoryUrls = useCallback(
    (urls: string[]) => {
      const grouped = urls.flatMap((url) => sourceUrlMap.get(getNormalizedUrlKey(url)) || [])
      return grouped.sort((a, b) => {
        const orderA = sourceDisplayOrder.get(a.source_id) ?? 0
        const orderB = sourceDisplayOrder.get(b.source_id) ?? 0
        return orderA - orderB
      })
    },
    [sourceDisplayOrder, sourceUrlMap]
  )

  /** Convert UTC HH:MM to local time display string. */
  function formatSyncTimeLocal(utcTime: string, _tz?: string): string {
    const [h, m] = (utcTime || '00:00').split(':').map(Number)
    const now = new Date()
    now.setUTCHours(h, m, 0, 0)
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true })
  }

  /** For Source column: URL or config summary (not display name). */
  function sourceUrlOrConfig(source: { type: string; config: Record<string, unknown> }): string {
    if (source.type === 'url' && typeof source.config?.url === 'string') return source.config.url
    if (source.type === 'pdf' && typeof source.config?.filename === 'string') return `PDF: ${source.config.filename}`
    if (source.type === 'drive' && typeof source.config?.folder_id === 'string') return `Drive folder: ${source.config.folder_id}`
    if (source.type === 'docs' && typeof source.config?.doc_id === 'string') return `Doc: ${source.config.doc_id}`
    return source.type || '—'
  }

  function sourceClickableUrl(source: { type: string; config: Record<string, unknown> }): string | null {
    if (source.type !== 'url' || typeof source.config?.url !== 'string') return null
    const candidate = source.config.url.trim()
    if (!candidate) return null
    try {
      const parsed = new URL(candidate)
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null
      return parsed.toString()
    } catch {
      return null
    }
  }

  /** For Name column: display_name or fallback from URL (pathname/hostname) for URL sources. */
  function sourceDisplayName(source: { type: string; config: Record<string, unknown>; display_name?: string | null }): string {
    if (source.display_name?.trim()) return source.display_name.trim()
    if (source.type === 'pdf' && typeof source.config?.filename === 'string') return source.config.filename
    if (source.type === 'url' && typeof source.config?.url === 'string') {
      try {
        const u = new URL(source.config.url)
        const path = u.pathname.replace(/\/+$/, '') || '/'
        if (path !== '/') return path.length > 80 ? path.slice(0, 77) + '...' : path
        return u.hostname || '—'
      } catch {
        return source.config.url.length > 60 ? source.config.url.slice(0, 57) + '...' : source.config.url
      }
    }
    return '—'
  }

  function sourceTypeLabel(type: string): string {
    const t = (type || '').toLowerCase()
    if (t === 'url') return 'URL'
    if (t === 'pdf') return 'PDF'
    if (t === 'drive') return 'Drive'
    if (t === 'docs') return 'Doc'
    return type || '—'
  }

  /** True if this source has an index job currently in progress (queued, crawling, uploading, importing). Stale jobs are ignored. */
  function isSourceTraining(sourceId: string, sourceType?: string): boolean {
    const hasInProgressBatchJob = jobs.some(
      (j) =>
        (j.hostname || '').toLowerCase() === 'batch' &&
        !SOURCES_JOB_TERMINAL_STAGES.has((j.stage || '').toLowerCase()) &&
        !SOURCES_JOB_POST_CRAWL_STAGES.has((j.stage || '').toLowerCase()) &&
        !isJobStale(j)
    )
    if (hasInProgressBatchJob && (sourceType || 'url').toLowerCase() === 'url') {
      return true
    }
    return jobs.some(
      (j) =>
        j.source_id === sourceId &&
        !SOURCES_JOB_TERMINAL_STAGES.has((j.stage || '').toLowerCase()) &&
        !SOURCES_JOB_POST_CRAWL_STAGES.has((j.stage || '').toLowerCase()) &&
        !isJobStale(j)
    )
  }

  if (!selectedBot) {
    return <div className="empty-panel">{t('botKnowledge.selectBotToManage', 'Select a bot to manage knowledge.')}</div>
  }

  return (
    <AnimatedPage className="card-grid knowledge-redesign">



      {/* Sources: main table — one row per source (URL, Drive, Docs, etc.) */}
      <GlassCard style={{ gridColumn: '1 / -1' }}>
        <div className="card-title">{t('botKnowledge.title', 'Sources')} ({sources.length})</div>
        {unifiedProgressVisible && unifiedSourcesProgress ? (
          <div style={{ marginBottom: '1rem', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
              {!unifiedSourcesProgress.isTerminal && (
                <span className="discovery-loading-dots sources-progress-loader" style={{ color: '#ec4899' }} aria-hidden>
                  <span /><span /><span />
                </span>
              )}
              <strong style={{ color: 'var(--ui-flow-accent)' }}>
                {unifiedCounterReady
                  ? t('botKnowledge.progressStepCounter', 'Step {{current}} of {{total}}', {
                    current: unifiedSourcesProgress.activeStepIndex + 1,
                    total: unifiedSourcesProgress.steps.length,
                  })
                  : t('botKnowledge.progressStepCounterLoading', 'Preparing training steps...')}
              </strong>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.55rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {displayUnifiedStep
                  ? t(displayUnifiedStep.labelKey, displayUnifiedStep.labelFallback)
                  : t('botKnowledge.progressStepCounterLoading', 'Preparing training steps...')}
              </span>
              <span style={{ fontSize: '0.9rem', color: 'var(--ui-flow-accent)', fontWeight: 600 }}>
                {Math.max(0, Math.min(100, Math.round(unifiedSourcesProgress.activeStepProgressPct || 0)))}%
              </span>
            </div>

            <div className="progress-track" style={{ height: '8px', borderRadius: '4px', overflow: 'hidden', background: 'var(--ui-flow-border)' }}>
              <div
                className="progress-fill"
                style={{
                  height: '100%',
                  width: `${unifiedSourcesProgress.activeStepProgressPct}%`,
                  borderRadius: '4px',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>

            {unifiedSourcesProgress.isError && (unifiedSourcesProgress.headlineMessage || unifiedSourcesProgress.headlineMessageKey) && (
              <div className="muted" style={{ fontSize: '0.84rem', marginTop: '0.45rem', color: '#dc2626' }}>
                {unifiedSourcesProgress.headlineMessage ||
                  (unifiedSourcesProgress.headlineMessageKey
                    ? t(
                      unifiedSourcesProgress.headlineMessageKey,
                      unifiedSourcesProgress.headlineMessageFallback || ''
                    )
                    : '')}
              </div>
            )}

            {unifiedSourcesProgress.activeCrawlStats &&
              (unifiedSourcesProgress.activeCrawlStats.pagesCrawled > 0 || unifiedSourcesProgress.activeCrawlStats.docsCount > 0) && (
                <div className="muted" style={{ fontSize: '0.84rem', marginTop: '0.4rem' }}>
                  {unifiedSourcesProgress.activeCrawlStats.pagesCrawled > 0
                    ? t('botKnowledge.learningPagesCount', 'Already read {{pages}} pages', {
                      pages: unifiedSourcesProgress.activeCrawlStats.pagesCrawled,
                    })
                    : t('botKnowledge.learningContentProcessing', 'Processing website content')}
                </div>
              )}
            {!unifiedSourcesProgress.activeCrawlStats && activeAdditionalSourcesStats && (
              <div className="muted" style={{ fontSize: '0.84rem', marginTop: '0.4rem' }}>
                {t('botKnowledge.progressAdditionalSourcesMessage', '{{done}}/{{total}} sources', {
                  done: activeAdditionalSourcesStats.done,
                  total: activeAdditionalSourcesStats.total,
                })}
              </div>
            )}
            {!unifiedSourcesProgress.activeCrawlStats && activeMenuExtractionStats && (
              <div className="muted" style={{ fontSize: '0.84rem', marginTop: '0.4rem' }}>
                {t('botKnowledge.menuExtractionStats', 'Menu items: found {{found}} · prepared {{prepared}}', {
                  found: activeMenuExtractionStats.found,
                  prepared: activeMenuExtractionStats.prepared,
                })}
              </div>
            )}

            {(progressControlState?.showStop || progressControlState?.showResume) && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.65rem', flexWrap: 'wrap' }}>
                {progressControlState?.showStop && (
                  <button
                    type="button"
                    className="primary"
                    onClick={handleStopTraining}
                    disabled={stoppingTraining}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
                  >
                    <span aria-hidden style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: 'currentColor', borderRadius: 2 }} />
                    {stoppingTraining ? t('botKnowledge.stopping', 'Stopping...') : t('botKnowledge.stopTraining', 'Stop training')}
                  </button>
                )}
                {progressControlState?.showResume && (
                  <button type="button" className="secondary" onClick={() => void handleResumeJobPipeline()}>
                    {t('botKnowledge.resumePipeline', 'Resume pipeline')}
                  </button>
                )}
              </div>
            )}
          </div>
        ) : (
          <p className="card-subtitle" style={{ marginTop: 0, marginBottom: '1rem' }}>
            {t('botKnowledge.subtitle', 'Every source (URL, PDF, Drive, Docs, etc.) this bot learns from. Add a URL or PDF (Drive/Docs coming soon). Training runs in the background.')}
          </p>
        )}
        {!unifiedProgressInFlight && latestFailedSourcesJob?.last_error && (
          <div className="alert error" style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>
            {latestFailedSourcesJob.last_error}
          </div>
        )}
        <div className="knowledge-toolbar" style={{ marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          {crawlStepInFlight ? (
            <span className="primary" style={{ opacity: 0.6, cursor: 'not-allowed', display: 'inline-flex', alignItems: 'center', padding: '0.6rem 1.1rem', borderRadius: '10px', border: '1px solid transparent', fontWeight: 500, fontSize: '1rem' }} aria-disabled>
              {t('botKnowledge.addSource', '+ Add source')}
            </span>
          ) : (
            <Link to={botId ? `/bots/${botId}/sources/new` : '#'} className="primary">
              {t('botKnowledge.addSource', '+ Add source')}
            </Link>
          )}
          {sources.length > 0 && (
            <>
              <button
                type="button"
                className={sourcesSelected.size === sources.length ? 'ghost' : 'secondary'}
                onClick={selectAllSources}
                disabled={crawlStepInFlight}
              >
                {sourcesSelected.size === sources.length ? t('botKnowledge.deselectAll', 'Deselect all') : t('botKnowledge.selectAll', 'Select all')}
              </button>
              {sourcesSelected.size > 0 && (
                <button
                  type="button"
                  className="secondary"
                  onClick={handleDeleteSelectedSources}
                  disabled={deletingSelectedSources || crawlStepInFlight}
                  style={{ color: '#dc2626' }}
                >
                  {deletingSelectedSources ? t('botKnowledge.deleting', 'Deleting...') : t('botKnowledge.deleteSelected', 'Delete selected ({{count}})', { count: sourcesSelected.size })}
                </button>
              )}
              {sourcesSelected.size > 0 && hasSelectedUrlSources && (
                <button
                  type="button"
                  className="secondary"
                  onClick={handleSyncSelected}
                  disabled={syncingSelected || crawlStepInFlight}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: syncingSelected ? 'spin 1s linear infinite' : 'none' }}>
                    <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0 1 15-6.7L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
                  </svg>
                  {syncingSelected ? t('botKnowledge.syncing', 'Syncing...') : t('botKnowledge.syncNow', 'Sync now')}
                </button>
              )}
              {sourcesSelected.size > 0 && <span className="muted">{t('botKnowledge.selectedCount', '{{count}} selected', { count: sourcesSelected.size })}</span>}
            </>
          )}
        </div>
        {sources.length > 0 ? (
          <div className="knowledge-table-wrap knowledge-table-wrap-scroll">
            <table className="knowledge-table">
              <thead>
                <tr>
                  <th style={{ width: '44px' }} aria-label={t('botKnowledge.selectAll', 'Select all')}>
                    <input
                      type="checkbox"
                      checked={sources.length > 0 && sourcesSelected.size === sources.length}
                      ref={(el) => { if (el) el.indeterminate = sourcesSelected.size > 0 && sourcesSelected.size < sources.length }}
                      onChange={() => sourcesSelected.size === sources.length ? deselectAllSources() : selectAllSources()}
                      disabled={crawlStepInFlight}
                      aria-label={t('botKnowledge.selectAll', 'Select all')}
                      style={{ cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
                    />
                  </th>
                  <th style={{ width: '120px' }}>{t('botKnowledge.type', 'Type')}</th>
                  <th style={{ width: '160px' }}>{t('botKnowledge.name', 'Name')}</th>
                  <th>{t('botKnowledge.source', 'Source')}</th>
                  <th style={{ width: '100px' }}>{t('botKnowledge.statusColumn', 'Status')}</th>
                  <th style={{ width: '220px' }}>{t('botKnowledge.sync', 'Sync')}</th>
                  <th>{t('botKnowledge.added', 'Added')}</th>
                  <th style={{ width: '80px' }}></th>
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const renderSourceRow = (s: typeof sources[0], indentLevel: number) => {
                    const training = isSourceTraining(s.source_id, s.type)
                    return (
                      <tr key={s.source_id} style={indentLevel > 0 ? { background: 'transparent' } : {}}>
                        <td style={indentLevel > 0 ? { paddingLeft: `${(0.75 + indentLevel * 1.1).toFixed(2)}rem` } : {}}>
                          <input
                            type="checkbox"
                            checked={sourcesSelected.has(s.source_id)}
                            onChange={() => toggleSourcesSelected(s.source_id)}
                            disabled={crawlStepInFlight}
                            aria-label={`Select ${sourceDisplayName(s)}`}
                            style={{ cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
                          />
                        </td>
                        <td>
                          <span className="source-type-badge" data-type={s.type.toLowerCase()}>
                            {sourceTypeLabel(s.type)}
                          </span>
                        </td>
                        <td className="knowledge-name">{sourceDisplayName(s)}</td>
                        <td className="knowledge-name" style={{ wordBreak: 'break-all' }}>
                          {(() => {
                            const url = sourceClickableUrl(s)
                            if (!url) return sourceUrlOrConfig(s)
                            return (
                              <a
                                href={url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="knowledge-source-link"
                                title={url}
                              >
                                {url}
                              </a>
                            )
                          })()}
                        </td>
                        <td>
                          <span
                            className="pill"
                            style={{
                              padding: '0.25rem 0.5rem',
                              borderRadius: '9999px',
                              fontSize: '0.8125rem',
                              fontWeight: 500,
                              ...(training
                                ? { background: '#e2e8f0', color: '#64748b' }
                                : { background: 'rgba(246, 180, 109, 0.2)', color: '#d97706' }),
                            }}
                          >
                            {training ? t('botKnowledge.training', 'Training') : t('botKnowledge.trained', 'Trained')}
                          </span>
                        </td>
                        <td>
                          {s.type.toLowerCase() === 'url' ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8125rem', position: 'relative' }}>
                              <button
                                type="button"
                                onClick={() => handleSyncSource(s.source_id)}
                                disabled={training || syncingSourceIds.has(s.source_id)}
                                title={t('botKnowledge.syncNow', 'Sync now')}
                                style={{ width: 30, height: 30, borderRadius: '50%', border: '1px solid var(--ui-flow-border)', background: 'var(--ui-flow-bg, #fff)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: training ? 'not-allowed' : 'pointer', opacity: training ? 0.4 : 1, flexShrink: 0, padding: 0 }}
                              >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: syncingSourceIds.has(s.source_id) ? 'spin 1s linear infinite' : 'none' }}>
                                  <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0 1 15-6.7L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
                                </svg>
                              </button>
                              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '0.8125rem', flexShrink: 0 }}>
                                <input
                                  type="checkbox"
                                  checked={!!s.sync_enabled}
                                  onChange={() => handleToggleAutoSync(s)}
                                  style={{ cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
                                />
                                {t('botKnowledge.autoSync', 'Auto')}
                              </label>
                              {s.sync_enabled && (
                                <span className="muted" style={{ fontSize: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '3px', whiteSpace: 'nowrap' }}>
                                  {t('botKnowledge.syncSchedule', '{{frequency}} at {{time}}', {
                                    frequency: t(`botKnowledge.${s.sync_frequency || 'daily'}`, s.sync_frequency || 'daily'),
                                    time: formatSyncTimeLocal(s.sync_time_utc || '00:00', s.sync_timezone),
                                  })}
                                  <button type="button" onClick={() => handleOpenSyncSettings(s)} title={t('botKnowledge.edit', 'edit')} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px', display: 'inline-flex', color: 'var(--ui-flow-muted)' }}>
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                      <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                                    </svg>
                                  </button>
                                </span>
                              )}
                              {s.last_synced_at && (
                                <span className="muted" style={{ fontSize: '0.7rem', whiteSpace: 'nowrap' }}>
                                  {formatRelativeTime(s.last_synced_at)}
                                </span>
                              )}
                              {syncSettingsSourceId === s.source_id && (
                                <div
                                  ref={syncPopupRef}
                                  className="knowledge-sync-popup"
                                  style={{
                                    position: 'absolute', top: '100%', left: 0, zIndex: 100,
                                    background: '#fff', border: '1px solid var(--ui-flow-border)', borderRadius: '10px',
                                    boxShadow: '0 8px 24px rgba(0,0,0,0.12)', padding: '14px 16px',
                                    display: 'flex', flexDirection: 'column', gap: '10px', minWidth: '240px', marginTop: '4px',
                                  }}
                                >
                                  <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{t('botKnowledge.autoSync', 'Auto sync')}</div>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <select className="knowledge-sync-popup-select" value={syncSettingsFrequency} onChange={(e) => setSyncSettingsFrequency(e.target.value)} style={{ fontSize: '0.8125rem', padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--ui-flow-border)', flex: 1 }}>
                                      <option value="daily">{t('botKnowledge.daily', 'Daily')}</option>
                                      <option value="weekly">{t('botKnowledge.weekly', 'Weekly')}</option>
                                      <option value="monthly">{t('botKnowledge.monthly', 'Monthly')}</option>
                                    </select>
                                    <span style={{ fontSize: '0.8125rem' }}>{t('botKnowledge.at', 'at')}</span>
                                    <select className="knowledge-sync-popup-select" value={syncSettingsHour} onChange={(e) => setSyncSettingsHour(e.target.value)} style={{ fontSize: '0.8125rem', padding: '4px 6px', borderRadius: '6px', border: '1px solid var(--ui-flow-border)', width: '52px' }}>
                                      {Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0')).map((h) => (
                                        <option key={h} value={h}>{h}</option>
                                      ))}
                                    </select>
                                    <span style={{ fontSize: '0.8125rem' }}>:</span>
                                    <select className="knowledge-sync-popup-select" value={syncSettingsMinute} onChange={(e) => setSyncSettingsMinute(e.target.value)} style={{ fontSize: '0.8125rem', padding: '4px 6px', borderRadius: '6px', border: '1px solid var(--ui-flow-border)', width: '52px' }}>
                                      <option value="00">00</option>
                                      <option value="15">15</option>
                                      <option value="30">30</option>
                                      <option value="45">45</option>
                                    </select>
                                  </div>
                                  <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                                    <button type="button" className="ghost" onClick={() => setSyncSettingsSourceId(null)} style={{ fontSize: '0.8125rem', padding: '4px 12px', borderRadius: '6px' }}>
                                      {t('botKnowledge.cancel', 'Cancel')}
                                    </button>
                                    <button type="button" className="primary" onClick={() => handleSaveSyncSettings(s.source_id, true)} disabled={savingSyncSettings} style={{ fontSize: '0.8125rem', padding: '4px 14px', borderRadius: '6px' }}>
                                      {savingSyncSettings ? '...' : t('botKnowledge.save', 'Save')}
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="muted" style={{ fontSize: '0.8125rem' }}>{t('botKnowledge.notApplicable', 'N/A')}</span>
                          )}
                        </td>
                        <td className="muted">{formatRelativeTime(s.updated_at)}</td>
                        <td>
                          <button
                            type="button"
                            className="delete-btn"
                            onClick={() => handleDeleteSource(s.source_id)}
                            disabled={deletingSourceId === s.source_id}
                            aria-label={`Delete ${sourceDisplayName(s)}`}
                          >
                            {deletingSourceId === s.source_id ? (
                              <span style={{ fontSize: '0.875rem' }}>...</span>
                            ) : (
                              <Trash2 size={18} aria-hidden />
                            )}
                          </button>
                        </td>
                      </tr>
                    )
                  }

                  const sortSourceCategories = (category: UrlCategory): UrlCategory[] => {
                    return Array.from(category.children.values()).sort((a, b) => {
                      const countA = getCategoryUrlCount(a)
                      const countB = getCategoryUrlCount(b)
                      if (countA !== countB) return countB - countA
                      return a.name.localeCompare(b.name)
                    })
                  }

                  const toggleSourceGroup = (path: string) => {
                    setExpandedSourceGroups((prev) => {
                      const next = new Set(prev)
                      if (next.has(path)) next.delete(path)
                      else next.add(path)
                      return next
                    })
                  }

                  const renderSourceCategory = (category: UrlCategory): React.ReactNode => {
                    const isExpanded = expandedSourceGroups.has(category.path)
                    const directSources = getSourcesForCategoryUrls(category.urls)
                    const totalCount = getCategoryUrlCount(category)
                    const childCategories = sortSourceCategories(category)

                    return (
                      <React.Fragment key={category.path}>
                        <tr
                          className="source-group-header"
                          onClick={() => toggleSourceGroup(category.path)}
                          style={{ cursor: 'pointer', background: 'var(--ui-flow-surface, rgba(228,88,122,0.03))', userSelect: 'none' }}
                        >
                          <td colSpan={8} style={{ padding: '0.5rem 0.75rem' }}>
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontWeight: 600,
                                fontSize: '0.85rem',
                                color: 'var(--ui-flow-text)',
                                paddingLeft: `${Math.max(0, category.level - 1) * 1.25}rem`,
                              }}
                            >
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ transition: 'transform 0.15s', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', color: 'var(--ui-flow-muted)', flexShrink: 0 }}>
                                <path d="M9 18l6-6-6-6" />
                              </svg>
                              {getCategoryDisplayPath(category)}
                              <span style={{ background: 'rgba(228,88,122,0.1)', color: 'var(--ui-flow-accent, #e4587a)', borderRadius: '999px', padding: '1px 8px', fontSize: '0.73rem', fontWeight: 700 }}>
                                {totalCount}
                              </span>
                            </span>
                          </td>
                        </tr>
                        {isExpanded && (
                          <>
                            {directSources.map((source) => renderSourceRow(source, Math.max(1, category.level)))}
                            {childCategories.map((child) => renderSourceCategory(child))}
                          </>
                        )}
                      </React.Fragment>
                    )
                  }

                  if (!sourceCategories) {
                    return <>{sources.map((source) => renderSourceRow(source, 0))}</>
                  }

                  return (
                    <>
                      {sourceCategories.urls.length > 0 &&
                        getSourcesForCategoryUrls(sourceCategories.urls).map((source) => renderSourceRow(source, 0))}
                      {sortSourceCategories(sourceCategories).map((category) => renderSourceCategory(category))}
                      {nonUrlSources.map((source) => renderSourceRow(source, 0))}
                    </>
                  )
                })()}
              </tbody>
            </table>
          </div>
        ) : unifiedProgressInFlight ? (
          <div className="empty muted" style={{ padding: '1.5rem' }}>
            {t('botKnowledge.trainingSelectedUrls', 'Training your selected URLs... Check progress above.')}
          </div>
        ) : (
          <div className="empty muted" style={{ padding: '1.5rem' }}>
            {t('botKnowledge.noSourcesYet', 'No sources yet. Add a URL or PDF to train this bot.')}
          </div>
        )}
      </GlassCard>

      {/* Booking links - hotel bots only */}
      {selectedBotWidgetConfig?.businessType === 'hotel' && (
        <GlassCard style={{ gridColumn: '1 / -1' }}>
          <div className="card-title">{t('botKnowledge.bookingLinksTitle', 'Booking links')}</div>
          {bookingLinkJob ? (
            (() => {
              const status = (bookingLinkJob.status || '').toLowerCase()
              const isError = status === 'failed' || status === 'error'
              const isDone = status === 'done'
              const links = (bookingLinkJob.links || []) as BookingLinkEntry[]
              return (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                    {!isDone && !isError && (
                      <span className="discovery-loading-dots" aria-hidden>
                        <span /><span /><span />
                      </span>
                    )}
                    <span style={{ color: isError ? '#dc2626' : '#0f766e', fontWeight: 600 }}>
                      {bookingStatusLabel(bookingLinkJob.status)}
                    </span>
                    {bookingLinkJob.updated_at && (
                      <span className="muted" style={{ fontSize: '0.875rem' }}>
                        · Updated {formatRelativeTime(bookingLinkJob.updated_at)}
                      </span>
                    )}
                  </div>
                  {bookingLinkJob.error && (
                    <div className="alert error" style={{ marginBottom: '0.75rem' }}>
                      {bookingLinkJob.error}
                    </div>
                  )}
                  {links.length > 0 ? (
                    <div className="url-list knowledge-table-wrap-scroll" style={{ maxHeight: '320px', overflowY: 'auto', border: '1px solid #e0e0e0', borderRadius: '6px', padding: '12px' }}>
                      {links.map((link) => {
                        const url = link.url || ''
                        const rawConfidence = typeof link.confidence === 'number' ? link.confidence : Number(link.confidence || 0)
                        const confidence = Number.isFinite(rawConfidence) ? rawConfidence : 0
                        const confidencePct = Math.round(confidence * 100)
                        const reason = Array.isArray(link.reasons) && link.reasons.length > 0 ? link.reasons[0] : ''
                        return (
                          <div key={url} className="url-list-item" style={{ marginBottom: '0.6rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                              <a href={url} target="_blank" rel="noreferrer" style={{ color: '#2563eb', wordBreak: 'break-all' }}>
                                {url}
                              </a>
                              <span className="muted" style={{ fontSize: '0.85rem' }}>
                                · {t('botKnowledge.confidence', 'Confidence {{percent}}%', { percent: confidencePct })}
                              </span>
                            </div>
                            {reason && (
                              <div className="muted" style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
                                {reason}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="muted">{t('botKnowledge.noBookingLinksYet', 'No booking links found yet.')}</div>
                  )}
                </>
              )
            })()
          ) : (
            <p className="card-subtitle" style={{ marginTop: 0 }}>
              {t('botKnowledge.bookingLinksSubtitle', 'Booking links are extracted from your trained knowledge after import completes.')}
            </p>
          )}
        </GlassCard>
      )}

      {/* Realtime availability (hotel bots only) */}
      {selectedBotWidgetConfig?.businessType === 'hotel' && (
        <GlassCard style={{ gridColumn: '1 / -1' }}>
          <div className="card-title">{t('botKnowledge.realtimeAvailabilityTitle', 'Realtime availability')}</div>
          <p className="card-subtitle" style={{ marginTop: 0 }}>
            {t('botKnowledge.realtimeAvailabilitySubtitle', 'Optional: allow the agent to check real-time room availability/pricing using a booking URL pattern.')}
          </p>

          <div className="design-form stack" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              <label
                className="url-list-item"
                style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', gap: '0.5rem' }}
              >
                <input
                  type="checkbox"
                  checked={allowRealtimeAvailability}
                  onChange={(e) => setAllowRealtimeAvailability(e.target.checked)}
                  style={{ accentColor: 'var(--ui-flow-accent-secondary)' }}
                />
                <span>{t('botKnowledge.allowRealtimeAvailability', 'Allow agent to check real-time room availability and answer user queries')}</span>
              </label>
              <button type="button" className="secondary" onClick={() => void handleSaveAvailabilitySettings()}>
                {t('botKnowledge.save', 'Save')}
              </button>
            </div>

            {allowRealtimeAvailability && (
              <div style={{ marginTop: '0.75rem' }}>
                <div className="testing-field">
                  <label className="testing-label">{t('botKnowledge.bookingTestUrl', 'Booking test URL')}</label>
                  <input
                    type="url"
                    className="design-form-input"
                    value={bookingTestUrl}
                    onChange={(e) => setBookingTestUrl(e.target.value)}
                    placeholder={t('botKnowledge.bookingTestUrlPlaceholder', 'https://www.booking.com/hotel/... or Agoda, Expedia, etc.')}
                    style={{ width: '100%', maxWidth: '700px' }}
                  />
                  <p className="muted" style={{ fontSize: '0.875rem', marginTop: '0.35rem' }}>
                    {t('botKnowledge.bookingTestUrlHint', 'Paste a booking URL (Agoda, Expedia, Booking.com, hotel site) with your dates and guests selected. The agent will learn the URL pattern for future checks.')}
                  </p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    className="primary"
                    onClick={() => void handleRunAvailabilityTest()}
                    disabled={!bookingTestUrl.trim() || availabilityTestRunning}
                  >
                    {availabilityTestRunning ? t('botKnowledge.agentTesting', 'Agent testing...') : t('botKnowledge.runAvailabilityTest', 'Run availability test')}
                  </button>
                </div>

                {availabilityTestError && (
                  <div className="alert error" style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                    {availabilityTestError}
                  </div>
                )}

                {availabilityTestJob && (
                  <div
                    className="progress-card"
                    style={{ marginTop: '0.75rem', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '0.75rem' }}
                  >
                    {availabilityTestRunning ? (
                      <div className="muted" style={{ fontSize: '0.875rem' }}>
                        {t('botKnowledge.agentTesting', 'Agent testing...')} {t('botKnowledge.statusLabel', 'Status: ')}<strong>{availabilityTestJob.status}</strong>
                      </div>
                    ) : (
                      <>
                        <div style={{ fontWeight: 500, marginBottom: '0.35rem' }}>{t('botKnowledge.agentTestResults', 'Agent test results')}</div>
                        <div className="muted" style={{ fontSize: '0.875rem' }}>
                          {t('botKnowledge.statusLabel', 'Status: ')}<strong>{availabilityTestJob.status}</strong>
                        </div>
                        {availabilityTestJob.summary && (
                          <div
                            className="alert info"
                            style={{ marginTop: '0.5rem', fontSize: '0.875rem', whiteSpace: 'pre-wrap' }}
                          >
                            {availabilityTestJob.summary}
                          </div>
                        )}
                        {availabilityTestJob.last_error && (
                          <div
                            className="alert error"
                            style={{ marginTop: '0.5rem', fontSize: '0.875rem', whiteSpace: 'pre-wrap' }}
                          >
                            {availabilityTestJob.last_error}
                          </div>
                        )}
                        <Link
                          to={botId ? `/bots/${botId}/testing` : '#'}
                          className="secondary"
                          style={{ display: 'inline-block', marginTop: '0.5rem', fontSize: '0.875rem' }}
                        >
                          {t('botKnowledge.viewInTestingTab', 'View in Testing tab')}
                        </Link>
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </GlassCard>
      )}

      {/* Add more pages — own-website bots only */}
      {allowKnowledgeDiscovery && (
        <GlassCard style={{ gridColumn: '1 / -1' }}>
          <div className="card-title">{t('botKnowledge.addMorePagesTitle', 'Add more pages')}</div>
          <p className="card-subtitle" style={{ marginTop: 0 }}>
            {isDiscovering ? (
              <span className="discovery-loading" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="discovery-loading-dots" aria-hidden>
                  <span />
                  <span />
                  <span />
                </span>
                <span style={{ color: 'var(--ui-flow-accent-secondary)', fontWeight: 500 }}>
                  {t('botKnowledge.discoveringPages', 'Discovering pages... {{count}} found so far', { count: discoveredUrls.length })}
                </span>
              </span>
            ) : (
              <>
                {t('botKnowledge.addMorePagesSubtitle', 'Enter a website URL to discover pages. Choose the ones your bot should learn from.')}
                {discoveredUrls.length > 0 && (
                  <span style={{ marginLeft: '8px', color: 'var(--ui-flow-accent-secondary)', fontWeight: 500 }}>
                    {t('botKnowledge.pagesFound', '{{count}} page(s) found.', { count: discoveredUrls.length })}
                  </span>
                )}
              </>
            )}
          </p>
          <div className="design-form stack" style={{ marginBottom: '1rem' }}>
            <div className="row" style={{ flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
              <input
                type="url"
                value={discoverInputUrl}
                onChange={(e) => setDiscoverInputUrl(e.target.value)}
                placeholder={t('botKnowledge.discoverUrlPlaceholder', 'https://example.com')}
                className="design-form-input"
                style={{ flex: 1, minWidth: '200px' }}
                disabled={crawlStepInFlight}
              />
              <button
                type="button"
                className="primary"
                onClick={handleDiscover}
                disabled={!discoverInputUrl.trim() || loading || isDiscovering || crawlStepInFlight}
              >
                {isDiscovering ? t('botKnowledge.discovering', 'Discovering...') : t('botKnowledge.discover', 'Discover')}
              </button>
            </div>
            {discoveryError && (
              <div className={`alert ${discoveryErrorType || 'error'}`} style={{ marginTop: '0.75rem', fontSize: '0.875rem' }}>
                {discoveryError}
              </div>
            )}
          </div>

          {(discoverTrainingJobId || discoverTrainingSuccess) && (
            <div className="muted" style={{ fontSize: '0.875rem', marginBottom: '0.75rem' }}>
              {t('botKnowledge.progressShownAbove', 'Progress is shown above.')}
            </div>
          )}

          {discoveredUrls.length > 0 && !isDiscovering && !discoverTrainingJobId && !discoverTrainingSuccess && (
            <>
              <div className="flow-toolbar" style={{ marginBottom: '0.75rem' }}>
                <button
                  type="button"
                  className={allDiscoveredSelected ? 'ghost' : 'secondary'}
                  onClick={toggleAllDiscovered}
                  disabled={crawlStepInFlight}
                >
                  {allDiscoveredSelected ? t('botKnowledge.deselectAll', 'Deselect all') : t('botKnowledge.selectAll', 'Select all')}
                </button>
                <button
                  type="button"
                  className={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
                  onClick={expandedCategories.size > 0 ? collapseAllCategories : expandAllCategories}
                  disabled={crawlStepInFlight}
                >
                  {expandedCategories.size > 0 ? t('botKnowledge.collapseAll', 'Collapse all') : t('botKnowledge.expandAll', 'Expand all')}
                </button>
                <div className="muted">{t('botKnowledge.selectedCount', '{{count}} selected', { count: selectedDiscovered.size })}</div>
              </div>
              <div className="url-list" style={{ maxHeight: '500px', overflowY: 'auto', border: '1px solid #e0e0e0', borderRadius: '4px', padding: '12px' }}>
                {urlCategories ? (
                  <div>
                    {Array.from(urlCategories.children.values())
                      .sort((a, b) => {
                        const countA = getCategoryUrlCount(a)
                        const countB = getCategoryUrlCount(b)
                        if (countA !== countB) return countB - countA
                        return a.name.localeCompare(b.name)
                      })
                      .map((category) => renderDiscoverCategory(category))}
                    {urlCategories.urls.length > 0 && (
                      <div style={{ marginLeft: 0 }}>
                        {urlCategories.urls.map((url) => (
                          <label
                            key={url}
                            className="url-list-item"
                            style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                          >
                            <input
                              type="checkbox"
                              checked={selectedDiscovered.has(url)}
                              onChange={() => toggleDiscovered(url)}
                              style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
                            />
                            <span style={{ fontSize: '16px', color: '#334155' }}>{url}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="muted">{t('botKnowledge.loadingCategories', 'Loading categories...')}</div>
                )}
              </div>
              <div className="flow-actions" style={{ marginTop: '1rem' }}>
                <button
                  type="button"
                  className="primary"
                  onClick={handleTrainDiscovered}
                  disabled={selectedDiscovered.size === 0 || loading || trainingDiscovered || crawlStepInFlight}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
                >
                  {trainingDiscovered ? t('botKnowledge.starting', 'Starting...') : t('botKnowledge.startTraining', '▷ Start training')}
                </button>
              </div>
            </>
          )}
        </GlassCard>
      )}

    </AnimatedPage>
  )
}
