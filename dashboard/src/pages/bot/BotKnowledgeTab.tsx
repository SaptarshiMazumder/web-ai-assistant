import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Trash2 } from 'lucide-react'
import { useDashboardData, type AvailabilityJobRecord, type BookingLinkJobRecord } from '../../hooks/useDashboardData'
import {
  categorizeUrls,
  getAllExpandablePaths,
  getAllUrlsFromCategory,
  getCategoryDisplayPath,
  getCategoryUrlCount,
  type UrlCategory,
} from '../createBot/urlCategorizer'
import {
  AnimatedPage,
  GlassCard,
} from '../../components/ui'

/** Jobs not updated in this long are considered stale (e.g. server was killed) and not shown as in-progress. */
const STALE_JOB_MS = 10 * 60 * 1000

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

function statusLabel(stage: string): string {
  const s = (stage || '').toLowerCase()
  if (s === 'complete' || s === 'done') return '✓ Trained'
  if (s === 'crawling' || s === 'running' || s === 'pending' || s === 'queued') return 'Training'
  if (s === 'uploading' || s === 'importing' || s === 'import_submitted') return 'Importing'
  if (s === 'failed' || s === 'error') return 'Failed'
  if (s === 'cancelled') return 'Cancelled'
  return stage || '—'
}

/** Rough progress 0–100 for training stage (for progress bar). */
function trainingProgressPercent(stage: string | undefined): number {
  const s = (stage || '').toLowerCase()
  if (s === 'done' || s === 'complete' || s === 'error' || s === 'failed' || s === 'cancelled' || s === 'import_submitted') return 100
  if (s === 'uploading' || s === 'importing') return 75
  if (s === 'crawling' || s === 'running' || s === 'pending') return 45
  if (s === 'queued') return 15
  return 10
}

/** Rough progress 0–100 for training stage (for progress bar). */
function trainingProgressDisplayPercent(
  stage: string | undefined,
  pagesCrawled: number | undefined,
  totalUrls: number
): number {
  const s = (stage || '').toLowerCase()
  if (s === 'done' || s === 'complete' || s === 'error' || s === 'failed' || s === 'cancelled' || s === 'import_submitted') return 100
  if (s === 'uploading' || s === 'importing') return 75
  if (s === 'crawling' || s === 'running' || s === 'pending') {
    if (totalUrls > 0 && pagesCrawled != null && pagesCrawled >= 0) {
      const pct = Math.round((pagesCrawled / totalUrls) * 100)
      return Math.min(99, Math.max(0, pct))
    }
    return 45
  }
  if (s === 'queued') return 15
  return 10
}

/** Human-readable label for training progress (Sources section). */
function trainingProgressLabel(stage: string | undefined): string {
  const s = (stage || '').toLowerCase()
  if (s === 'queued' || s === 'crawling' || s === 'running' || s === 'pending' || s === 'uploading' || s === 'importing' || s === 'import_submitted') {
    return 'Training in progress…'
  }
  return 'Training…'
}

const SOURCES_JOB_TERMINAL_STAGES = new Set(['done', 'complete', 'error', 'failed', 'cancelled', 'import_submitted'])

const BOOKING_LINK_JOB_TERMINAL_STATUS = new Set(['done', 'failed', 'error'])

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

export default function BotKnowledgeTab() {
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
    listBookingLinkJobs,
    getBookingLinkJob,
    startAvailabilityJob,
    getAvailabilityJob,
    saveWidgetConfig,
    selectedBotWidgetConfig,
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
  const [discoverTrainingStatus, setDiscoverTrainingStatus] = useState<{
    stage?: string
    docs_count?: number
    last_error?: string
  } | null>(null)
  const [discoverTrainingUrlCount, setDiscoverTrainingUrlCount] = useState(0)
  const [discoverTrainingSuccess, setDiscoverTrainingSuccess] = useState(false)

  const [sourcesTrainingJobId, setSourcesTrainingJobId] = useState<string | null>(null)
  const [sourcesTrainingStatus, setSourcesTrainingStatus] = useState<{
    stage?: string
    pages_crawled?: number
    docs_count?: number
    last_error?: string
  } | null>(null)

  const [bookingLinkJob, setBookingLinkJob] = useState<BookingLinkJobRecord | null>(null)

  const [allowRealtimeAvailability, setAllowRealtimeAvailability] = useState(false)
  const [bookingTestUrl, setBookingTestUrl] = useState('')
  const [availabilityTestJob, setAvailabilityTestJob] = useState<AvailabilityJobRecord | null>(null)
  const [availabilityTestError, setAvailabilityTestError] = useState<string | null>(null)
  const [availabilityTestRunning, setAvailabilityTestRunning] = useState(false)
  const availabilityTestPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const discoverSuccessTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const DISCOVER_TERMINAL_STAGES = new Set(['done', 'error', 'cancelled', 'import_submitted'])
  const DISCOVER_SUCCESS_STAGES = new Set(['done', 'import_submitted'])

  useEffect(() => {
    if (!allowKnowledgeDiscovery) return
    if (!selectedBot || !discoverTrainingJobId) return
    let cancelled = false
    const poll = async () => {
      const status = await getJobStatus(selectedBot.bot_id, discoverTrainingJobId!)
      if (cancelled || !status) return
      setDiscoverTrainingStatus({ stage: status.stage, docs_count: status.docs_count, last_error: status.last_error })
      if (status.stage && DISCOVER_TERMINAL_STAGES.has(status.stage)) {
        void loadSources(selectedBot.bot_id)
        void loadJobs(selectedBot.bot_id)
        setDiscoverTrainingJobId(null)
        if (status.stage && DISCOVER_SUCCESS_STAGES.has(status.stage)) {
          setDiscoverTrainingSuccess(true)
          if (discoverSuccessTimeoutRef.current) window.clearTimeout(discoverSuccessTimeoutRef.current)
          discoverSuccessTimeoutRef.current = window.setTimeout(() => {
            setDiscoverTrainingStatus(null)
            setDiscoverTrainingUrlCount(0)
            setDiscoverTrainingSuccess(false)
            discoverSuccessTimeoutRef.current = null
          }, 3000)
        } else {
          setDiscoverTrainingStatus(null)
          setDiscoverTrainingUrlCount(0)
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
  }, [allowKnowledgeDiscovery, selectedBot, discoverTrainingJobId, getJobStatus, loadJobs, loadSources])

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
    if (!selectedBot || !availabilityTestJob) return
    if (AVAILABILITY_TERMINAL_STATUS.has(availabilityTestJob.status)) {
      setAvailabilityTestRunning(false)
      return
    }
    setAvailabilityTestRunning(true)
    availabilityTestPollRef.current = setInterval(async () => {
      const updated = await getAvailabilityJob(selectedBot.bot_id, availabilityTestJob.job_id)
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
  }, [selectedBot, availabilityTestJob?.job_id, availabilityTestJob?.status, getAvailabilityJob])

  // Sources section: detect in-progress index job and poll so we can show training progress bar (ignore stale jobs)
  const activeSourcesJob = useMemo(() => {
    if (!selectedBot || !jobs.length) return null
    const inProgress = jobs.filter(
      (j) =>
        !SOURCES_JOB_TERMINAL_STAGES.has((j.stage || '').toLowerCase()) && !isJobStale(j)
    )
    if (inProgress.length === 0) return null
    inProgress.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    return inProgress[0] ?? null
  }, [selectedBot, jobs])

  useEffect(() => {
    if (!selectedBot || !activeSourcesJob) {
      setSourcesTrainingJobId(null)
      setSourcesTrainingStatus(null)
      return
    }
    const jobId = activeSourcesJob.job_id
    setSourcesTrainingJobId(jobId)
    setSourcesTrainingStatus({
      stage: activeSourcesJob.stage,
      pages_crawled: activeSourcesJob.pages_crawled,
      docs_count: activeSourcesJob.docs_count,
      last_error: activeSourcesJob.last_error,
    })
    let cancelled = false
    const poll = async () => {
      const status = await getJobStatus(selectedBot.bot_id, jobId)
      if (cancelled || !status) return
      setSourcesTrainingStatus({
        stage: status.stage,
        pages_crawled: status.pages_crawled,
        docs_count: status.docs_count,
        last_error: status.last_error,
      })
      if (status.stage && SOURCES_JOB_TERMINAL_STAGES.has(status.stage.toLowerCase())) {
        setSourcesTrainingJobId(null)
        setSourcesTrainingStatus(null)
        void loadJobs(selectedBot.bot_id)
        void loadSources(selectedBot.bot_id)
      }
    }
    const timer = setInterval(poll, 4000)
    void poll()
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [selectedBot, activeSourcesJob?.job_id, getJobStatus, loadJobs, loadSources])


  // Booking link extraction (RAG) progress
  useEffect(() => {
    if (!selectedBot) {
      setBookingLinkJob(null)
      return
    }
    let cancelled = false
    let pollTimer: ReturnType<typeof setInterval> | null = null

    const load = async () => {
      const jobs = await listBookingLinkJobs(selectedBot.bot_id)
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
        const current = await getBookingLinkJob(selectedBot.bot_id, latest.job_id)
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
  }, [selectedBot, listBookingLinkJobs, getBookingLinkJob])

  const normalizedDiscoverUrl = (discoverInputUrl || '').trim().replace(/\/+$/, '') || undefined
  const urlCategories = useMemo(() => {
    if (!discoveredUrls.length || !normalizedDiscoverUrl) return null
    return categorizeUrls(discoveredUrls, normalizedDiscoverUrl)
  }, [discoveredUrls, normalizedDiscoverUrl])

  const hasExpandedDefault = useRef(false)
  useEffect(() => {
    if (!discoveredUrls.length) {
      hasExpandedDefault.current = false
      return
    }
    if (urlCategories && !hasExpandedDefault.current) {
      setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
      hasExpandedDefault.current = true
    }
  }, [urlCategories, discoveredUrls.length])

  // Refetch jobs/sources when Knowledge tab is shown for a bot so we pick up data from create-bot (queueCrawlUrls may have completed after initial load).
  const lastRefetchedBotIdRef = useRef<string | null>(null)
  useEffect(() => {
    if (!botId || !selectedBot || selectedBot.bot_id !== botId) return
    if (lastRefetchedBotIdRef.current === botId) return
    lastRefetchedBotIdRef.current = botId
    void loadJobs(botId)
    void loadSources(botId)
  }, [botId, selectedBot?.bot_id, loadJobs, loadSources])
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
      void loadJobs(botId)
      void loadSources(botId)
      setEmptyPollCount((c) => c + 1)
    }, 2000)
    return () => clearTimeout(t)
  }, [botId, selectedBot?.bot_id, sources.length, jobs.length, emptyPollCount, loadJobs, loadSources])

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
        setDiscoveryError('⚠️ We could not discover real pages from this site. Try PDF upload for key pages.')
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
        setDiscoverTrainingUrlCount(urls.length)
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

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to manage knowledge.</div>
  }

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
    const trainingInProgress = !!(sourcesTrainingJobId && sourcesTrainingStatus)
    if (trainingInProgress) return
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
  }, [selectedBot, sourcesSelected, deletingSelectedSources, sourcesTrainingJobId, sourcesTrainingStatus, deleteSource, loadSources, loadJobs])

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

  /** For Source column: URL or config summary (not display name). */
  function sourceUrlOrConfig(source: { type: string; config: Record<string, unknown> }): string {
    if (source.type === 'url' && typeof source.config?.url === 'string') return source.config.url
    if (source.type === 'pdf' && typeof source.config?.filename === 'string') return `PDF: ${source.config.filename}`
    if (source.type === 'drive' && typeof source.config?.folder_id === 'string') return `Drive folder: ${source.config.folder_id}`
    if (source.type === 'docs' && typeof source.config?.doc_id === 'string') return `Doc: ${source.config.doc_id}`
    return source.type || '—'
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
        !isJobStale(j)
    )
    if (hasInProgressBatchJob && (sourceType || 'url').toLowerCase() === 'url') {
      return true
    }
    return jobs.some(
      (j) =>
        j.source_id === sourceId &&
        !SOURCES_JOB_TERMINAL_STAGES.has((j.stage || '').toLowerCase()) &&
        !isJobStale(j)
    )
  }

  return (
    <AnimatedPage className="card-grid knowledge-redesign">



      {/* Sources: main table — one row per source (URL, Drive, Docs, etc.) */}
      <GlassCard style={{ gridColumn: '1 / -1' }}>
        <div className="card-title">Sources ({sources.length})</div>
        {sourcesTrainingJobId && sourcesTrainingStatus ? (
          (() => {
            const totalUrls = sources.length
            const pct = trainingProgressDisplayPercent(
              sourcesTrainingStatus.stage,
              sourcesTrainingStatus.pages_crawled,
              totalUrls
            )
            return (
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                  <span className="discovery-loading-dots" aria-hidden>
                    <span /><span /><span />
                  </span>
                  <span style={{ color: 'var(--ui-flow-accent)', fontWeight: 600 }}>
                    {trainingProgressLabel(sourcesTrainingStatus.stage)} {pct}%
                  </span>
                  {sourcesTrainingStatus.pages_crawled != null && sourcesTrainingStatus.pages_crawled > 0 && (
                    <span className="muted" style={{ fontSize: '0.875rem' }}>
                      · {sourcesTrainingStatus.pages_crawled} pages · {sourcesTrainingStatus.docs_count ?? 0} docs
                    </span>
                  )}
                  <button
                    type="button"
                    className="primary"
                    onClick={handleStopTraining}
                    disabled={stoppingTraining}
                    style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
                  >
                    <span aria-hidden style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: 'currentColor', borderRadius: 2 }} />
                    {stoppingTraining ? 'Stopping…' : 'Stop training'}
                  </button>
                </div>
                <div className="progress-track" style={{ height: '8px', borderRadius: '4px', overflow: 'hidden', background: 'var(--ui-flow-border)' }}>
                  <div
                    className="progress-fill"
                    style={{
                      height: '100%',
                      width: `${pct}%`,
                      borderRadius: '4px',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
                {sourcesTrainingStatus.last_error && (
                  <div className="alert error" style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                    {sourcesTrainingStatus.last_error}
                  </div>
                )}
              </div>
            )
          })()
        ) : (
          <p className="card-subtitle" style={{ marginTop: 0, marginBottom: '1rem' }}>
            Every source (URL, PDF, Drive, Docs, etc.) this bot learns from. Add a URL or PDF (Drive/Docs coming soon). Training runs in the background.
          </p>
        )}
        <div className="knowledge-toolbar" style={{ marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          {sourcesTrainingJobId && sourcesTrainingStatus ? (
            <span className="primary" style={{ opacity: 0.6, cursor: 'not-allowed', display: 'inline-flex', alignItems: 'center', padding: '0.6rem 1.1rem', borderRadius: '10px', border: '1px solid transparent', fontWeight: 500, fontSize: '1rem' }} aria-disabled>
              + Add source
            </span>
          ) : (
            <Link to={botId ? `/bots/${botId}/sources/new` : '#'} className="primary">
              + Add source
            </Link>
          )}
          {sources.length > 0 && (
            <>
              <button
                type="button"
                className={sourcesSelected.size === sources.length ? 'ghost' : 'secondary'}
                onClick={selectAllSources}
                disabled={!!(sourcesTrainingJobId && sourcesTrainingStatus)}
              >
                {sourcesSelected.size === sources.length ? 'Deselect all' : 'Select all'}
              </button>
              {sourcesSelected.size > 0 && (
                <button
                  type="button"
                  className="secondary"
                  onClick={handleDeleteSelectedSources}
                  disabled={deletingSelectedSources || !!(sourcesTrainingJobId && sourcesTrainingStatus)}
                  style={{ color: '#dc2626' }}
                >
                  {deletingSelectedSources ? 'Deleting…' : `Delete selected (${sourcesSelected.size})`}
                </button>
              )}
              {sourcesSelected.size > 0 && <span className="muted">{sourcesSelected.size} selected</span>}
            </>
          )}
        </div>
        {sources.length > 0 ? (
          <div className="knowledge-table-wrap knowledge-table-wrap-scroll">
            <table className="knowledge-table">
              <thead>
                <tr>
                  <th style={{ width: '44px' }} aria-label="Select">
                    <input
                      type="checkbox"
                      checked={sources.length > 0 && sourcesSelected.size === sources.length}
                      ref={(el) => { if (el) el.indeterminate = sourcesSelected.size > 0 && sourcesSelected.size < sources.length }}
                      onChange={() => sourcesSelected.size === sources.length ? deselectAllSources() : selectAllSources()}
                      disabled={!!(sourcesTrainingJobId && sourcesTrainingStatus)}
                      aria-label="Select all sources"
                      style={{ cursor: 'pointer', accentColor: 'var(--ui-flow-accent-secondary)' }}
                    />
                  </th>
                  <th style={{ width: '120px' }}>Type</th>
                  <th style={{ width: '160px' }}>Name</th>
                  <th>Source</th>
                  <th style={{ width: '100px' }}>Status</th>
                  <th>Added</th>
                  <th style={{ width: '80px' }}></th>
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => {
                  const training = isSourceTraining(s.source_id, s.type)
                  return (
                    <tr key={s.source_id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={sourcesSelected.has(s.source_id)}
                          onChange={() => toggleSourcesSelected(s.source_id)}
                          disabled={!!(sourcesTrainingJobId && sourcesTrainingStatus)}
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
                        {sourceUrlOrConfig(s)}
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
                          {training ? 'Training' : 'Trained'}
                        </span>
                      </td>
                      <td className="muted">{formatRelativeTime(s.updated_at)}</td>
                      <td>
                        <button
                          type="button"
                          onClick={() => handleDeleteSource(s.source_id)}
                          disabled={deletingSourceId === s.source_id}
                          aria-label={`Delete ${sourceDisplayName(s)}`}
                          style={{
                            padding: '0.25rem',
                            background: 'none',
                            border: 'none',
                            cursor: deletingSourceId === s.source_id ? 'not-allowed' : 'pointer',
                            color: deletingSourceId === s.source_id ? '#94a3b8' : '#dc2626',
                            opacity: deletingSourceId === s.source_id ? 0.7 : 1,
                          }}
                        >
                          {deletingSourceId === s.source_id ? (
                            <span style={{ fontSize: '0.875rem' }}>…</span>
                          ) : (
                            <Trash2 size={18} aria-hidden />
                          )}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : sourcesTrainingJobId && sourcesTrainingStatus ? (
          <div className="empty muted" style={{ padding: '1.5rem' }}>
            Training your selected URLs… Check progress above.
          </div>
        ) : (
          <div className="empty muted" style={{ padding: '1.5rem' }}>
            No sources yet. Add a URL or PDF to train this bot.
          </div>
        )}
      </GlassCard>

      {/* Booking links - hotel bots only */}
      {selectedBotWidgetConfig?.businessType === 'hotel' && (
        <GlassCard style={{ gridColumn: '1 / -1' }}>
          <div className="card-title">Booking links</div>
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
                                · Confidence {confidencePct}%
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
                    <div className="muted">No booking links found yet.</div>
                  )}
                </>
              )
            })()
          ) : (
            <p className="card-subtitle" style={{ marginTop: 0 }}>
              Booking links are extracted from your trained knowledge after import completes.
            </p>
          )}
        </GlassCard>
      )}

      {/* Realtime availability (hotel bots only) */}
      {selectedBotWidgetConfig?.businessType === 'hotel' && (
        <GlassCard style={{ gridColumn: '1 / -1' }}>
          <div className="card-title">Realtime availability</div>
          <p className="card-subtitle" style={{ marginTop: 0 }}>
            Optional: allow the agent to check real-time room availability/pricing using a booking URL pattern.
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
                <span>Allow agent to check real-time room availability and answer user queries</span>
              </label>
              <button type="button" className="secondary" onClick={() => void handleSaveAvailabilitySettings()}>
                Save
              </button>
            </div>

            {allowRealtimeAvailability && (
              <div style={{ marginTop: '0.75rem' }}>
                <div className="testing-field">
                  <label className="testing-label">Booking test URL</label>
                  <input
                    type="url"
                    className="design-form-input"
                    value={bookingTestUrl}
                    onChange={(e) => setBookingTestUrl(e.target.value)}
                    placeholder="https://www.booking.com/hotel/... or Agoda, Expedia, etc."
                    style={{ width: '100%', maxWidth: '700px' }}
                  />
                  <p className="muted" style={{ fontSize: '0.875rem', marginTop: '0.35rem' }}>
                    Paste a booking URL (Agoda, Expedia, Booking.com, hotel site) with your dates and guests selected.
                    The agent will learn the URL pattern for future checks.
                  </p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    className="primary"
                    onClick={() => void handleRunAvailabilityTest()}
                    disabled={!bookingTestUrl.trim() || availabilityTestRunning}
                  >
                    {availabilityTestRunning ? 'Agent testing…' : 'Run availability test'}
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
                        Agent testing… Status: <strong>{availabilityTestJob.status}</strong>
                      </div>
                    ) : (
                      <>
                        <div style={{ fontWeight: 500, marginBottom: '0.35rem' }}>Agent test results</div>
                        <div className="muted" style={{ fontSize: '0.875rem' }}>
                          Status: <strong>{availabilityTestJob.status}</strong>
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
                          View in Testing tab
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
          <div className="card-title">Add more pages</div>
          <p className="card-subtitle" style={{ marginTop: 0 }}>
            {isDiscovering ? (
              <span className="discovery-loading" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="discovery-loading-dots" aria-hidden>
                  <span />
                  <span />
                  <span />
                </span>
                <span style={{ color: 'var(--ui-flow-accent-secondary)', fontWeight: 500 }}>
                  Discovering pages… {discoveredUrls.length} found so far
                </span>
              </span>
            ) : (
              <>
                Enter a website URL to discover pages. Choose the ones your bot should learn from.
                {discoveredUrls.length > 0 && (
                  <span style={{ marginLeft: '8px', color: 'var(--ui-flow-accent-secondary)', fontWeight: 500 }}>
                    {discoveredUrls.length} page{discoveredUrls.length === 1 ? '' : 's'} found.
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
                placeholder="https://example.com"
                className="design-form-input"
                style={{ flex: 1, minWidth: '200px' }}
                disabled={!!(sourcesTrainingJobId && sourcesTrainingStatus)}
              />
              <button
                type="button"
                className="primary"
                onClick={handleDiscover}
                disabled={!discoverInputUrl.trim() || loading || isDiscovering || !!(sourcesTrainingJobId && sourcesTrainingStatus)}
              >
                {isDiscovering ? 'Discovering…' : 'Discover'}
              </button>
            </div>
            {discoveryError && (
              <div className={`alert ${discoveryErrorType || 'error'}`} style={{ marginTop: '0.75rem', fontSize: '0.875rem' }}>
                {discoveryError}
              </div>
            )}
          </div>

          {(discoverTrainingJobId || discoverTrainingSuccess) && (
            <div className="progress-card" style={{ marginBottom: '1rem', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
              {discoverTrainingSuccess ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#059669', fontWeight: 500 }}>
                  <span aria-hidden style={{ fontSize: '1.25rem' }}>✓</span>
                  <span>Added to bot knowledge</span>
                </div>
              ) : (
                <>
                  <div className="progress-label" style={{ color: '#334155' }}>
                    Training in progress… {discoverTrainingUrlCount} page{discoverTrainingUrlCount === 1 ? '' : 's'}
                  </div>
                  <div className="progress-track" style={{ marginTop: '0.5rem' }}>
                    <div
                      className="progress-fill"
                      style={{ width: `${trainingProgressPercent(discoverTrainingStatus?.stage)}%` }}
                    />
                  </div>
                  <div className="muted" style={{ fontSize: '0.875rem', marginTop: '0.35rem' }}>
                    {discoverTrainingStatus?.stage ? statusLabel(discoverTrainingStatus.stage) : 'Starting…'}
                    {discoverTrainingStatus?.docs_count != null && discoverTrainingStatus.docs_count > 0 && (
                      <> · {discoverTrainingStatus.docs_count} doc{discoverTrainingStatus.docs_count === 1 ? '' : 's'}</>
                    )}
                  </div>
                  {discoverTrainingStatus?.last_error && (
                    <div className="alert error" style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                      {discoverTrainingStatus.last_error}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {discoveredUrls.length > 0 && !isDiscovering && !discoverTrainingJobId && !discoverTrainingSuccess && (
            <>
              <div className="flow-toolbar" style={{ marginBottom: '0.75rem' }}>
                <button
                  type="button"
                  className={allDiscoveredSelected ? 'ghost' : 'secondary'}
                  onClick={toggleAllDiscovered}
                  disabled={!!(sourcesTrainingJobId && sourcesTrainingStatus)}
                >
                  {allDiscoveredSelected ? 'Deselect all' : 'Select all'}
                </button>
                <button
                  type="button"
                  className={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
                  onClick={expandedCategories.size > 0 ? collapseAllCategories : expandAllCategories}
                  disabled={!!(sourcesTrainingJobId && sourcesTrainingStatus)}
                >
                  {expandedCategories.size > 0 ? 'Collapse all' : 'Expand all'}
                </button>
                <div className="muted">{selectedDiscovered.size} selected</div>
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
                  <div className="muted">Loading categories…</div>
                )}
              </div>
              <div className="flow-actions" style={{ marginTop: '1rem' }}>
                <button
                  type="button"
                  className="primary"
                  onClick={handleTrainDiscovered}
                  disabled={selectedDiscovered.size === 0 || loading || trainingDiscovered || !!(sourcesTrainingJobId && sourcesTrainingStatus)}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
                >
                  {trainingDiscovered ? 'Starting…' : '▷ Start training'}
                </button>
              </div>
            </>
          )}
        </GlassCard>
      )}

    </AnimatedPage>
  )
}

