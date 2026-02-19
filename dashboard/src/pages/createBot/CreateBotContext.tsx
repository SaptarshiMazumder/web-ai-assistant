import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import { DEFAULT_WIDGET_DESIGN_STATE, type SuggestedMessageConfig } from '../../components/WidgetDesignForm'
import { CREATE_BOT_FIRST_PATH, getCreateBotNextPath, getCreateBotPrevPath, getCreateBotSteps } from './flowConfig'

type TrainingStage = 'idle' | 'training' | 'complete'
type ContentHosting = 'own' | 'shared'

/** Step 1: Bot details. */
export type CreateBotStep1Slice = {
  botName: string
  setBotName: (value: string) => void
  businessType: '' | 'hotel' | 'other'
  setBusinessType: (value: '' | 'hotel' | 'other') => void
  localError: string | null
  setLocalError: (value: string | null) => void
}

export type SharedUrlRow = {
  url: string
  label: string
}

export type CustomTextEntry = {
  id: string
  title: string
  content: string
}

/** Step 2: Hosting + sources (Step 2 + Step 3 pages consume different parts of this slice). */
export type CreateBotStep2Slice = {
  contentHosting: ContentHosting | null
  setContentHosting: (value: ContentHosting) => void
  websiteUrl: string
  setWebsiteUrl: (value: string) => void
  discoveryMethod: string
  setDiscoveryMethod: (value: string) => void
  discoveredUrls: string[]
  selectedUrls: string[]
  normalizedWebsiteUrl: string
  sharedUrlRows: SharedUrlRow[]
  setSharedUrlRows: (rows: SharedUrlRow[]) => void
  trainingUrls: string[]
  setTrainingUrls: (urls: string[]) => void
  sharedUrls: string[]
  pdfFiles: File[]
  setPdfFiles: (files: File[]) => void
  textDocFiles: File[]
  setTextDocFiles: (files: File[]) => void
  plainTextContent: string
  setPlainTextContent: (value: string) => void
  customTextEntries: CustomTextEntry[]
  setCustomTextEntries: (entries: CustomTextEntry[]) => void
  discoveryDurationMs: number | null
  discoveryTimedOutMessage: string | null
  isDiscovering: boolean
  isStartingTraining: boolean
  localError: string | null
  localErrorType: 'error' | 'warning' | null
  setLocalError: (value: string | null, type?: 'error' | 'warning' | null) => void
  continueWithoutSources: () => Promise<string | null>
  discoverUrls: () => Promise<boolean>
  toggleUrl: (url: string) => void
  toggleCategory: (categoryPath: string, categoryUrls: string[]) => void
  selectAll: () => void
  deselectAll: () => void
  startTraining: (selectedDiscoveredUrls?: string[]) => Promise<string | null>
  stopDiscovery: () => void
}

/** Step 3: Training progress. Change only this slice when editing the third step. */
export type CreateBotStep3Slice = {
  trainingStage: TrainingStage
  trainingProgress: number
  trainingPagesCrawled: number
  trainingDocsCount: number
  trainingStageName: string
  botId: string | null
  jobId: string | null
  pdfJobIds: string[]
  pdfJobs: Array<{
    job_id: string
    stage: string
    pages_crawled?: number
    docs_count?: number
    last_error?: string
  }>
  extraJobIds: string[]
  extraJobs: Array<{
    job_id: string
    stage: string
    pages_crawled?: number
    docs_count?: number
    last_error?: string
  }>
  localError: string | null
  setLocalError: (value: string | null) => void
  resetFlow: () => void
}

/** Step 4: Design widget. Change only this slice when editing the fourth step. */
export type CreateBotStep4Slice = {
  widgetPosition: 'bottom-right' | 'bottom-left'
  setWidgetPosition: (value: 'bottom-right' | 'bottom-left') => void
  widgetPrimaryColor: string
  setWidgetPrimaryColor: (value: string) => void
  widgetTitle: string
  setWidgetTitle: (value: string) => void
  widgetSize: 'small' | 'medium' | 'large'
  setWidgetSize: (value: 'small' | 'medium' | 'large') => void
  welcomeMessage: string
  setWelcomeMessage: (value: string) => void
  placeholder: string
  setPlaceholder: (value: string) => void
  footerMessage: string
  setFooterMessage: (value: string) => void
  theme: 'light' | 'dark'
  setTheme: (value: 'light' | 'dark') => void
  textColor: string
  setTextColor: (value: string) => void
  launcherIconUrl: string
  setLauncherIconUrl: (value: string) => void
  launcherText: string
  setLauncherText: (value: string) => void
  headerIconUrl: string
  setHeaderIconUrl: (value: string) => void
  shareIconUrl: string
  setShareIconUrl: (value: string) => void
  maxHeight: number
  setMaxHeight: (value: number) => void
  fontSize: 'small' | 'medium' | 'large'
  setFontSize: (value: 'small' | 'medium' | 'large') => void
  headerSize: 'small' | 'medium' | 'large'
  setHeaderSize: (value: 'small' | 'medium' | 'large') => void
  autoPopupWelcome: 'off' | '1s' | '2s' | '3s'
  setAutoPopupWelcome: (value: 'off' | '1s' | '2s' | '3s') => void
  autoScrollNewMessages: boolean
  setAutoScrollNewMessages: (value: boolean) => void
  displaySourcesInMessages: boolean
  setDisplaySourcesInMessages: (value: boolean) => void
  sourcesLabel: string
  setSourcesLabel: (value: string) => void
  suggestedMessages: SuggestedMessageConfig[]
  setSuggestedMessages: (value: SuggestedMessageConfig[]) => void
}

/** Flow navigation. Derived from flowConfig; add/remove steps there. */
export type CreateBotFlowSlice = {
  nextPath: string | null
  prevPath: string | null
  firstPath: string
}

export type CreateBotContextValue = {
  step1: CreateBotStep1Slice
  step2: CreateBotStep2Slice
  step3: CreateBotStep3Slice
  step4: CreateBotStep4Slice
  flow: CreateBotFlowSlice
  resetFlow: () => void
}

const CreateBotContext = createContext<CreateBotContextValue | undefined>(undefined)

/**
 * When adding a new step: 1) Add step to flowConfig.ts (path, label, description).
 * 2) Add Route in App.tsx. 3) Define StepNSlice type and add stepN to value below.
 * 4) Add step state and include it in resetFlow().
 */
function normalizeUrl(value: string) {
  const trimmed = value.trim()
  if (!trimmed) return ''
  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
  const parsed = new URL(withProtocol)
  return parsed.href
}

export function CreateBotProvider({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const {
    createBot,
    discoverUrls: discoverUrlsFromHook,
    queueCrawlUrls,
    saveWidgetConfig,
    getJobStatus,
    uploadPdfSources,
    uploadTextSources,
    uploadDocsSources,
    setSelectedBotId,
    orgs,
    activeOrgId,
    isSuperAdmin,
    generateSuggestedMessages,
  } = useDashboardData()
  const [botName, setBotName] = useState('')
  const [websiteUrl, setWebsiteUrl] = useState('')
  const [contentHosting, setContentHosting] = useState<ContentHosting | null>('shared')
  const [businessType, setBusinessType] = useState<'' | 'hotel' | 'other'>('')
  const [discoveryMethod, setDiscoveryMethod] = useState('auto') // 'auto' (crawl4ai) or 'sitemap'
  const [normalizedWebsiteUrl, setNormalizedWebsiteUrl] = useState('')
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedUrls, setSelectedUrls] = useState<string[]>([])
  const [sharedUrlRows, setSharedUrlRows] = useState<SharedUrlRow[]>([{ url: '', label: '' }])
  const [trainingUrls, setTrainingUrls] = useState<string[]>([])
  const [pdfFiles, setPdfFiles] = useState<File[]>([])
  const [textDocFiles, setTextDocFiles] = useState<File[]>([])
  const [plainTextContent, setPlainTextContent] = useState('')
  const [customTextEntries, setCustomTextEntries] = useState<CustomTextEntry[]>([{ id: '1', title: '', content: '' }])
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [isStartingTraining, setIsStartingTraining] = useState(false)
  const [discoveryDurationMs, setDiscoveryDurationMs] = useState<number | null>(null)
  const [discoveryTimedOutMessage, setDiscoveryTimedOutMessage] = useState<string | null>(null)
  const selectionTouchedRef = useRef(false)
  const discoveryStartTimeRef = useRef<number | null>(null)
  const discoveryAbortRef = useRef<AbortController | null>(null)
  const discovery60sTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const discoveryTimedOutByTimerRef = useRef(false)
  const suggestionsGenTriggeredRef = useRef(false)
  const [trainingStage, setTrainingStage] = useState<TrainingStage>('idle')
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingPagesCrawled, setTrainingPagesCrawled] = useState(0)
  const [trainingDocsCount, setTrainingDocsCount] = useState(0)
  const [trainingStageName, setTrainingStageName] = useState('')
  const [botId, setBotId] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [pdfJobIds, setPdfJobIds] = useState<string[]>([])
  const [pdfJobStatusById, setPdfJobStatusById] = useState<Record<string, any>>({})
  const [extraJobIds, setExtraJobIds] = useState<string[]>([])
  const [extraJobStatusById, setExtraJobStatusById] = useState<Record<string, any>>({})
  const [localError, setLocalError] = useState<string | null>(null)
  const [localErrorType, setLocalErrorType] = useState<'error' | 'warning' | null>(null)
  const [widgetPosition, setWidgetPosition] = useState<'bottom-right' | 'bottom-left'>('bottom-right')
  const [widgetPrimaryColor, setWidgetPrimaryColor] = useState('#e4587a')
  const [widgetTitle, setWidgetTitle] = useState('Chat')
  const [widgetSize, setWidgetSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [welcomeMessage, setWelcomeMessage] = useState('Welcome! How can I help you today?')
  const [placeholder, setPlaceholder] = useState('Ask a question...')
  const [footerMessage, setFooterMessage] = useState('Powered by WebAI')
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [textColor, setTextColor] = useState('#ffffff')
  const [launcherIconUrl, setLauncherIconUrl] = useState('')
  const [launcherText, setLauncherText] = useState('Help')
  const [headerIconUrl, setHeaderIconUrl] = useState('')
  const [shareIconUrl, setShareIconUrl] = useState('')
  const [maxHeight, setMaxHeight] = useState(560)
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [headerSize, setHeaderSize] = useState<'small' | 'medium' | 'large'>('small')
  const [autoPopupWelcome, setAutoPopupWelcome] = useState<'off' | '1s' | '2s' | '3s'>('off')
  const [autoScrollNewMessages, setAutoScrollNewMessages] = useState(true)
  const [displaySourcesInMessages, setDisplaySourcesInMessages] = useState(false)
  const [sourcesLabel, setSourcesLabel] = useState('Sources')
  const [suggestedMessages, setSuggestedMessages] = useState<SuggestedMessageConfig[]>(
    DEFAULT_WIDGET_DESIGN_STATE.suggestedMessages
  )
  const resetFlow = useCallback(() => {
    setBotName('')
    setWebsiteUrl('')
    setContentHosting('shared')
    setBusinessType('')
    setDiscoveryMethod('auto')
    setNormalizedWebsiteUrl('')
    setDiscoveredUrls([])
    setSelectedUrls([])
    setSharedUrlRows([{ url: '', label: '' }])
    setTrainingUrls([])
    setPdfFiles([])
    setTextDocFiles([])
    setPlainTextContent('')
    setCustomTextEntries([{ id: '1', title: '', content: '' }])
    setIsDiscovering(false)
    setIsStartingTraining(false)
    setDiscoveryDurationMs(null)
    setDiscoveryTimedOutMessage(null)
    setTrainingStage('idle')
    setTrainingProgress(0)
    setTrainingPagesCrawled(0)
    setTrainingDocsCount(0)
    setTrainingStageName('')
    setBotId(null)
    setJobId(null)
    setPdfJobIds([])
    setPdfJobStatusById({})
    setExtraJobIds([])
    setExtraJobStatusById({})
    setLocalError(null)
    setWidgetPosition('bottom-right')
    setWidgetPrimaryColor('#e4587a')
    setWidgetTitle('Chat')
    setWidgetSize('medium')
    setWelcomeMessage('Welcome! How can I help you today?')
    setPlaceholder('Ask a question...')
    setFooterMessage('Powered by WebAI')
    setTheme('light')
    setTextColor('#ffffff')
    setLauncherIconUrl('')
    setLauncherText('Help')
    setHeaderIconUrl('')
    setShareIconUrl('')
    setMaxHeight(560)
    setFontSize('medium')
    setHeaderSize('small')
    setAutoPopupWelcome('off')
    setAutoScrollNewMessages(true)
    setDisplaySourcesInMessages(false)
    setSourcesLabel('Sources')
    setSuggestedMessages(DEFAULT_WIDGET_DESIGN_STATE.suggestedMessages)
  }, [])

  const discoverUrls = useCallback(async () => {
    setLocalError(null)
    if (!botName.trim()) {
      setLocalError('Enter a bot name to continue.')
      return false
    }
    if (!websiteUrl.trim()) {
      setLocalError('Enter a website URL to continue.')
      return false
    }
    let normalized = ''
    try {
      normalized = normalizeUrl(websiteUrl)
    } catch {
      setLocalError('Enter a valid website URL.')
      return false
    }
    setIsDiscovering(true)
    setNormalizedWebsiteUrl(normalized)

    // Reset lists for streaming UI; keep selection auto-checked until user touches selection.
    setDiscoveredUrls([])
    setSelectedUrls([])
    setDiscoveryDurationMs(null)
    setDiscoveryTimedOutMessage(null)
    selectionTouchedRef.current = false
    discoveryStartTimeRef.current = Date.now()

    const controller = new AbortController()
    discoveryAbortRef.current = controller
    discoveryTimedOutByTimerRef.current = false

    // Client-side 90s cap: when user presses Discover, we stop reading the stream after 90s and use what we have.
    discovery60sTimerRef.current = setTimeout(() => {
      discovery60sTimerRef.current = null
      discoveryTimedOutByTimerRef.current = true
      controller.abort()
    }, 90_000)

    // Track discovered count locally to avoid stale state in finally block
    let localDiscoveredCount = 0
    let hasShownError = false

    // Fire-and-forget stream so UI can navigate immediately and update progressively.
    void (async () => {
      await discoverUrlsFromHook(normalized, discoveryMethod, (evt) => {
        if (evt.type === 'discovered' && typeof evt.url === 'string') {
          const url = evt.url
          localDiscoveredCount++
          setDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))

          if (!selectionTouchedRef.current) {
            setSelectedUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
          }
        }

        if (evt.type === 'error' && typeof evt.message === 'string') {
          hasShownError = true
          const reason = (evt as { failure_reason?: string }).failure_reason
          if (reason === 'robots_blocked') {
            setLocalError('🚫 This site blocks crawlers via robots.txt. Try adding specific URLs manually.')
            setLocalErrorType('error')
          } else if (reason === 'sitemap_empty') {
            setLocalError("No sitemap found. Switch to 'Automatic' discovery (recommended).")
            setLocalErrorType('warning')
          } else {
            setLocalError(evt.message)
            setLocalErrorType('error')
          }
        }

        if (evt.type === 'warning' && typeof evt.message === 'string') {
          hasShownError = true
          setLocalError(evt.message)
          setLocalErrorType('warning')
        }

        if (evt.type === 'done') {
          if (discovery60sTimerRef.current) {
            clearTimeout(discovery60sTimerRef.current)
            discovery60sTimerRef.current = null
          }
          const start = discoveryStartTimeRef.current
          if (start != null) setDiscoveryDurationMs(Date.now() - start)
          setIsDiscovering(false)
          if ((evt as { timed_out?: boolean }).timed_out === true) {
            setDiscoveryTimedOutMessage('Found main URLs. You can train on these now.')
          }
          const urls = (evt as { urls?: unknown[] }).urls || []
          const reason = (evt as { failure_reason?: string }).failure_reason
          if (reason === 'no_results') {
            hasShownError = true
            setLocalError('⚠️ We could not discover real pages from this site. Please use the PDF upload steps below.')
            setLocalErrorType('warning')
          } else if (Array.isArray(urls) && urls.length === 0) {
            hasShownError = true
            if (reason === 'robots_blocked') {
              setLocalError('🚫 All discovered URLs are blocked by robots.txt')
              setLocalErrorType('error')
            } else if (reason === 'sitemap_empty') {
              setLocalError("No sitemap found. Switch to 'Automatic' discovery.")
              setLocalErrorType('warning')
            } else if (reason === 'no_results') {
              setLocalError('⚠️ No pages found. Site may be blocking crawlers or have no discoverable links.')
              setLocalErrorType('warning')
            } else {
              setLocalError(
                discoveryMethod === 'sitemap'
                  ? "Could not discover via sitemap. Switch to 'Automatic' (recommended)."
                  : 'No URLs found for this site.'
              )
              setLocalErrorType('warning')
            }
          }
        }
      }, controller.signal, { max_duration_sec: 90 })
        .then((final) => {
          // If we got ≤1 URL, treat as failure
          const urlCount = final?.urls?.length ?? 0
          localDiscoveredCount = Math.max(localDiscoveredCount, urlCount)
          if (localDiscoveredCount <= 1 || final?.failureReason === 'no_results') {
            hasShownError = true
            setLocalError('⚠️ We could not discover real pages from this site. Please use the PDF upload steps below.')
            setLocalErrorType('warning')
          } else if (final && !final.urls?.length && final.error) {
            hasShownError = true
            setLocalError(final.error)
            setLocalErrorType('error')
          }
          const start = discoveryStartTimeRef.current
          if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
        })
        .catch((err: Error & { name?: string }) => {
          if (err.name === 'AbortError') {
            const start = discoveryStartTimeRef.current
            if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
            if (discoveryTimedOutByTimerRef.current) {
              setDiscoveryTimedOutMessage('Found main URLs. You can train on these now.')
            }
          }
        })
        .finally(() => {
          if (discovery60sTimerRef.current) {
            clearTimeout(discovery60sTimerRef.current)
            discovery60sTimerRef.current = null
          }
          setIsDiscovering(false)
          discoveryAbortRef.current = null

          // CRITICAL SAFETY: If ≤1 URL discovered and no error shown, FORCE show error.
          // Use local count to avoid stale React state in closure.
          if (localDiscoveredCount <= 1 && !hasShownError) {
            setLocalError('⚠️ Discovery completed but found no usable pages. Please use PDF upload instead.')
            setLocalErrorType('warning')
          }
        })
    })()

    // Return true so the UI can move to the URLs page immediately.
    return true
  }, [botName, websiteUrl, discoveryMethod, discoverUrlsFromHook])

  const continueWithoutSources = useCallback(async () => {
    setLocalError(null)
    if (!botName.trim()) {
      setLocalError('Enter a bot name to continue.')
      return null
    }
    setIsStartingTraining(true)
    const orgOverride =
      isSuperAdmin && (!activeOrgId || activeOrgId === '__all__') && orgs.length > 0 ? orgs[0].org_id : undefined
    const created = await createBot(botName.trim(), orgOverride)
    if (!created) {
      setIsStartingTraining(false)
      setLocalError('Failed to create bot. Select an organization above if you are an admin.')
      return null
    }
    setBotId(created.bot_id)
    setSelectedBotId(created.bot_id)

    const urlBankForSave = (() => {
      const normalize = (entry: string): string => {
        const raw = (entry || '').trim()
        if (!raw) return ''
        try {
          const u = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`)
          if (u.protocol !== 'http:' && u.protocol !== 'https:') return ''
          return u.toString()
        } catch {
          return ''
        }
      }
      const m = new Map<string, { label: string; url: string }>()
      for (const row of sharedUrlRows) {
        const url = normalize(row.url)
        if (!url) continue
        const rawLabel = (row.label || '').trim()
        const label = rawLabel || 'Other'
        const prev = m.get(url)
        if (!prev || ((prev.label === 'Other' || prev.label === 'Link') && rawLabel)) {
          m.set(url, { url, label })
        }
      }
      return Array.from(m.values())
    })()

    void saveWidgetConfig(created.bot_id, {
      contentHosting: 'shared',
      businessType: businessType || undefined,
      urlBank: urlBankForSave,
    }).catch(() => { })
    setTrainingStage('complete')
    setTrainingProgress(100)
    setTrainingPagesCrawled(0)
    setTrainingDocsCount(0)
    setTrainingStageName('skipped')
    setJobId(null)
    setIsStartingTraining(false)
    return created.bot_id
  }, [botName, createBot, setSelectedBotId, orgs, activeOrgId, isSuperAdmin, saveWidgetConfig, contentHosting, businessType, sharedUrlRows])

  const normalizeOneUrl = useCallback((entry: string): string => {
    const raw = (entry || '').trim()
    if (!raw) return ''
    try {
      const u = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`)
      if (u.protocol !== 'http:' && u.protocol !== 'https:') return ''
      return u.toString()
    } catch {
      return ''
    }
  }, [])

  const sharedUrls = useMemo(() => {
    const out: string[] = []
    for (const row of sharedUrlRows) {
      const u = normalizeOneUrl(row.url)
      if (u && !out.includes(u)) out.push(u)
    }
    return out
  }, [sharedUrlRows, normalizeOneUrl])

  type UrlBankEntry = { label: string; url: string }
  const urlBank: UrlBankEntry[] = useMemo(() => {
    const m = new Map<string, UrlBankEntry>()
    for (const row of sharedUrlRows) {
      const url = normalizeOneUrl(row.url)
      if (!url) continue
      const rawLabel = (row.label || '').trim()
      const label = rawLabel || 'Other'
      const prev = m.get(url)
      if (!prev || ((prev.label === 'Other' || prev.label === 'Link') && rawLabel)) {
        m.set(url, { url, label })
      }
    }
    return Array.from(m.values())
  }, [sharedUrlRows, normalizeOneUrl])

  const toggleUrl = useCallback((url: string) => {
    selectionTouchedRef.current = true
    setSelectedUrls((prev) => (prev.includes(url) ? prev.filter((item) => item !== url) : [...prev, url]))
  }, [])

  const toggleCategory = useCallback((_categoryPath: string, categoryUrls: string[]) => {
    selectionTouchedRef.current = true
    setSelectedUrls((prev) => {
      const allSelected = categoryUrls.every(url => prev.includes(url))
      if (allSelected) {
        // Deselect all URLs in category
        return prev.filter(url => !categoryUrls.includes(url))
      } else {
        // Select all URLs in category
        const newSelected = [...prev]
        for (const url of categoryUrls) {
          if (!newSelected.includes(url)) {
            newSelected.push(url)
          }
        }
        return newSelected
      }
    })
  }, [])

  const selectAll = useCallback(() => {
    selectionTouchedRef.current = true
    setSelectedUrls(discoveredUrls)
  }, [discoveredUrls])

  const deselectAll = useCallback(() => {
    selectionTouchedRef.current = true
    setSelectedUrls([])
  }, [])

  const stopDiscovery = useCallback(() => {
    if (discovery60sTimerRef.current) {
      clearTimeout(discovery60sTimerRef.current)
      discovery60sTimerRef.current = null
    }
    discoveryAbortRef.current?.abort()
  }, [])

  const startTraining = useCallback(async (selectedDiscoveredUrls?: string[]) => {
    setLocalError(null)
    if (!botName.trim()) {
      setLocalError('Enter a bot name to continue.')
      return null
    }
    const overrideUrls = (selectedDiscoveredUrls || [])
      .map((u) => normalizeOneUrl(u))
      .filter(Boolean)
    const finalUrls = overrideUrls.length > 0
      ? overrideUrls
      : (contentHosting === 'own' ? selectedUrls : trainingUrls)
    const hasPdfs = pdfFiles.length > 0
    const hasDocs = textDocFiles.length > 0
    const hasPlainText = plainTextContent.trim().length > 0
    const hasCustom = customTextEntries.some((e) => e.content.trim())
    setIsStartingTraining(true)
    const orgOverride =
      isSuperAdmin && (!activeOrgId || activeOrgId === '__all__') && orgs.length > 0 ? orgs[0].org_id : undefined
    const created = await createBot(botName.trim(), orgOverride)
    if (!created) {
      setIsStartingTraining(false)
      setLocalError('Failed to start training. Select an organization above if you are an admin.')
      return null
    }
    setBotId(created.bot_id)
    setSelectedBotId(created.bot_id)
    void saveWidgetConfig(created.bot_id, {
      contentHosting: 'shared',
      businessType: businessType || undefined,
      urlBank,
    }).catch(() => { })
    if (!finalUrls.length && !hasPdfs && !hasDocs && !hasPlainText && !hasCustom) {
      setTrainingStage('complete')
      setTrainingProgress(100)
      setTrainingPagesCrawled(0)
      setTrainingDocsCount(0)
      setTrainingStageName('skipped')
      setJobId(null)
      setPdfJobIds([])
      setPdfJobStatusById({})
      setExtraJobIds([])
      setExtraJobStatusById({})
      setIsStartingTraining(false)
      return created.bot_id
    }

    setTrainingStage('training')
    setTrainingProgress(0)
    setTrainingPagesCrawled(0)
    setTrainingDocsCount(0)
    setTrainingStageName('crawling')
    setJobId(null)
    setPdfJobIds([])
    setPdfJobStatusById({})
    setExtraJobIds([])
    setExtraJobStatusById({})

    const starters: Promise<unknown>[] = []
    if (finalUrls.length > 0) {
      starters.push(
        queueCrawlUrls(created.bot_id, finalUrls)
          .then((jobIdResult) => {
            if (jobIdResult) setJobId(jobIdResult)
            else setLocalError('Could not start. Please try again.')
          })
          .catch(() => { })
      )
    }
    if (hasPdfs) {
      const startPdfUpload = uploadPdfSources(created.bot_id, pdfFiles, null)
        .then((resp) => {
          const ids = (resp?.items || []).map((it) => it.job_id).filter(Boolean)
          setPdfJobIds(ids)
          return ids
        })
        .catch(() => {
          return []
        })

      // If this is a PDF-only training run, wait until we have job ids before navigating,
      // otherwise the progress page can't poll and will sit at 0%.
      if (finalUrls.length === 0) {
        const ids = await startPdfUpload
        if (!ids.length) {
          setLocalError('Could not start. Please try again.')
        }
      } else {
        starters.push(startPdfUpload)
      }
    }
    // Docs (.txt/.md/.docx/.doc) upload
    if (hasDocs) {
      starters.push(
        uploadDocsSources(created.bot_id, textDocFiles)
          .then((resp) => {
            const ids = (resp?.items || []).map((it) => it.job_id).filter(Boolean)
            setExtraJobIds((prev) => [...prev, ...ids])
          })
          .catch(() => { })
      )
    }

    // Plain text upload
    if (hasPlainText) {
      starters.push(
        uploadTextSources(created.bot_id, [{ content: plainTextContent }])
          .then((resp) => {
            const ids = (resp?.items || []).map((it) => it.job_id).filter(Boolean)
            setExtraJobIds((prev) => [...prev, ...ids])
          })
          .catch(() => { })
      )
    }

    // Custom text entries upload
    if (hasCustom) {
      const entries = customTextEntries
        .filter((e) => e.content.trim())
        .map((e) => ({ title: e.title.trim() || undefined, content: e.content.trim() }))
      starters.push(
        uploadTextSources(created.bot_id, entries)
          .then((resp) => {
            const ids = (resp?.items || []).map((it) => it.job_id).filter(Boolean)
            setExtraJobIds((prev) => [...prev, ...ids])
          })
          .catch(() => { })
      )
    }

    Promise.allSettled(starters).finally(() => setIsStartingTraining(false))
    // Background discovery disabled for now.
    return created.bot_id
  }, [botName, createBot, queueCrawlUrls, saveWidgetConfig, contentHosting, selectedUrls, trainingUrls, setSelectedBotId, orgs, activeOrgId, isSuperAdmin, normalizedWebsiteUrl, websiteUrl, discoveryMethod, businessType, pdfFiles, uploadPdfSources, textDocFiles, plainTextContent, customTextEntries, uploadTextSources, uploadDocsSources, urlBank, normalizeOneUrl])

  useEffect(() => {
    if (trainingStage !== 'training' || !botId) return
    if (!jobId && pdfJobIds.length === 0 && extraJobIds.length === 0) return

    function stagePercent(stage: string): number {
      const st = (stage || '').toLowerCase()
      if (st === 'queued') return 5
      if (st === 'crawling') return 40
      if (st === 'uploading') return 70
      if (st === 'importing') return 85
      if (st === 'import_submitted') return 100
      if (st === 'done') return 100
      if (st === 'error') return 100
      return 15
    }

    const pollStatus = async () => {
      let urlStatus: any | null = null
      if (jobId) {
        urlStatus = await getJobStatus(botId, jobId)
      }
      const pdfStatuses: Record<string, any> = {}
      for (const id of pdfJobIds) {
        const st = await getJobStatus(botId, id)
        if (st) pdfStatuses[id] = st
      }
      setPdfJobStatusById(pdfStatuses)

      const extraStatuses: Record<string, any> = {}
      for (const id of extraJobIds) {
        const st = await getJobStatus(botId, id)
        if (st) extraStatuses[id] = st
      }
      setExtraJobStatusById(extraStatuses)

      const urlStage = (urlStatus?.stage || '').toLowerCase()
      const urlTerminal = urlStage === 'done' || urlStage === 'import_submitted' || urlStage === 'error' || urlStage === 'cancelled'

      const pdfStages = Object.values(pdfStatuses).map((s: any) => (s?.stage || '').toLowerCase())
      const pdfTerminal = pdfStages.length > 0 ? pdfStages.every((st) => ['done', 'import_submitted', 'error', 'cancelled'].includes(st)) : true

      const extraStages = Object.values(extraStatuses).map((s: any) => (s?.stage || '').toLowerCase())
      const extraTerminal = extraStages.length > 0 ? extraStages.every((st) => ['done', 'import_submitted', 'error', 'cancelled'].includes(st)) : true

      const allStatuses = [...Object.values(pdfStatuses), ...Object.values(extraStatuses)]
      const pages = (urlStatus?.pages_crawled || 0) + allStatuses.reduce((sum: number, s: any) => sum + (s?.pages_crawled || 0), 0)
      const docs = (urlStatus?.docs_count || 0) + allStatuses.reduce((sum: number, s: any) => sum + (s?.docs_count || 0), 0)
      setTrainingPagesCrawled(pages)
      setTrainingDocsCount(docs)

      const stageName = urlStatus?.stage || pdfStages[0] || extraStages[0] || 'crawling'
      setTrainingStageName(stageName)

      const totalUrls = contentHosting === 'own' ? selectedUrls.length : trainingUrls.length
      const urlProgress =
        jobId && urlStatus
          ? totalUrls > 0 && urlStatus.pages_crawled
            ? Math.min(Math.round((urlStatus.pages_crawled / totalUrls) * 100), 95)
            : stagePercent(urlStatus.stage || 'queued')
          : null

      const allJobProgresses = allStatuses.map((s: any) => stagePercent(s?.stage || 'queued'))
      const parts: number[] = []
      if (typeof urlProgress === 'number') parts.push(urlProgress)
      parts.push(...allJobProgresses)
      const combined = parts.length ? Math.min(Math.round(parts.reduce((a, b) => a + b, 0) / parts.length), 100) : 0
      setTrainingProgress(combined)

      if ((urlTerminal || !jobId) && pdfTerminal && extraTerminal) {
        setTrainingStage('complete')
        setTrainingProgress(100)
        // Auto-generate suggested messages from trained content (once)
        if (botId && !suggestionsGenTriggeredRef.current) {
          suggestionsGenTriggeredRef.current = true
          void generateSuggestedMessages(botId)
        }
        if (urlStage === 'error') setLocalError(urlStatus?.last_error || 'Training failed')
        const pdfError = Object.values(pdfStatuses).find((s: any) => (s?.stage || '').toLowerCase() === 'error')
        if (pdfError) setLocalError((pdfError as any).last_error || 'PDF processing failed')
        const extraError = Object.values(extraStatuses).find((s: any) => (s?.stage || '').toLowerCase() === 'error')
        if (extraError) setLocalError((extraError as any).last_error || 'Source processing failed')
      }
    }
    pollStatus()
    const timer = window.setInterval(pollStatus, 1500)
    return () => window.clearInterval(timer)
  }, [trainingStage, botId, jobId, pdfJobIds, extraJobIds, getJobStatus, contentHosting, selectedUrls.length, trainingUrls.length, generateSuggestedMessages])

  const steps = getCreateBotSteps()
  const nextPath = getCreateBotNextPath(location.pathname, steps)
  const prevPath = getCreateBotPrevPath(location.pathname, steps)

  const value = useMemo(
    () => ({
      step1: {
        botName,
        setBotName,
        businessType,
        setBusinessType,
        localError,
        localErrorType,
        setLocalError: (value: string | null, type: 'error' | 'warning' | null = 'error') => {
          setLocalError(value)
          setLocalErrorType(type)
        },
      },
      step2: {
        contentHosting,
        setContentHosting,
        websiteUrl,
        setWebsiteUrl,
        discoveryMethod,
        setDiscoveryMethod,
        discoveredUrls,
        selectedUrls,
        normalizedWebsiteUrl,
        sharedUrlRows,
        setSharedUrlRows,
        trainingUrls,
        setTrainingUrls,
        sharedUrls,
        pdfFiles,
        setPdfFiles,
        textDocFiles,
        setTextDocFiles,
        plainTextContent,
        setPlainTextContent,
        customTextEntries,
        setCustomTextEntries,
        discoveryDurationMs,
        discoveryTimedOutMessage,
        isDiscovering,
        isStartingTraining,
        localError,
        localErrorType,
        setLocalError: (value: string | null, type: 'error' | 'warning' | null = 'error') => {
          setLocalError(value)
          setLocalErrorType(type)
        },
        continueWithoutSources,
        discoverUrls,
        toggleUrl,
        toggleCategory,
        selectAll,
        deselectAll,
        startTraining,
        stopDiscovery,
      },
      step3: {
        trainingStage,
        trainingProgress,
        trainingPagesCrawled,
        trainingDocsCount,
        trainingStageName,
        botId,
        jobId,
        pdfJobIds,
        pdfJobs: pdfJobIds.map((id) => ({ job_id: id, ...(pdfJobStatusById[id] || {}) })),
        extraJobIds,
        extraJobs: extraJobIds.map((id) => ({ job_id: id, ...(extraJobStatusById[id] || {}) })),
        localError,
        setLocalError,
        resetFlow,
      },
      step4: {
        widgetPosition,
        setWidgetPosition,
        widgetPrimaryColor,
        setWidgetPrimaryColor,
        widgetTitle,
        setWidgetTitle,
        widgetSize,
        setWidgetSize,
        welcomeMessage,
        setWelcomeMessage,
        placeholder,
        setPlaceholder,
        footerMessage,
        setFooterMessage,
        theme,
        setTheme,
        textColor,
        setTextColor,
        launcherIconUrl,
        setLauncherIconUrl,
        launcherText,
        setLauncherText,
        headerIconUrl,
        setHeaderIconUrl,
        shareIconUrl,
        setShareIconUrl,
        maxHeight,
        setMaxHeight,
        fontSize,
        setFontSize,
        headerSize,
        setHeaderSize,
        autoPopupWelcome,
        setAutoPopupWelcome,
        autoScrollNewMessages,
        setAutoScrollNewMessages,
        displaySourcesInMessages,
        setDisplaySourcesInMessages,
        sourcesLabel,
        setSourcesLabel,
        suggestedMessages,
        setSuggestedMessages,
      },
      flow: {
        nextPath,
        prevPath,
        firstPath: CREATE_BOT_FIRST_PATH,
      },
      resetFlow,
    }),
    [
      botName,
      setBotName,
      websiteUrl,
      setWebsiteUrl,
      contentHosting,
      setContentHosting,
      businessType,
      setBusinessType,
      discoveryMethod,
      setDiscoveryMethod,
      normalizedWebsiteUrl,
      discoveredUrls,
      selectedUrls,
      sharedUrlRows,
      setSharedUrlRows,
      trainingUrls,
      setTrainingUrls,
      sharedUrls,
      pdfFiles,
      setPdfFiles,
      textDocFiles,
      setTextDocFiles,
      customTextEntries,
      setCustomTextEntries,
      isDiscovering,
      isStartingTraining,
      discoveryDurationMs,
      discoveryTimedOutMessage,
      trainingStage,
      trainingProgress,
      trainingPagesCrawled,
      trainingDocsCount,
      trainingStageName,
      botId,
      jobId,
      pdfJobIds,
      pdfJobStatusById,
      localError,
      widgetPosition,
      setWidgetPosition,
      widgetPrimaryColor,
      setWidgetPrimaryColor,
      widgetTitle,
      setWidgetTitle,
      widgetSize,
      setWidgetSize,
      welcomeMessage,
      setWelcomeMessage,
      placeholder,
      setPlaceholder,
      footerMessage,
      setFooterMessage,
      theme,
      setTheme,
      textColor,
      setTextColor,
      launcherIconUrl,
      setLauncherIconUrl,
      launcherText,
      setLauncherText,
      headerIconUrl,
      setHeaderIconUrl,
      shareIconUrl,
      setShareIconUrl,
      maxHeight,
      setMaxHeight,
      fontSize,
      setFontSize,
      headerSize,
      setHeaderSize,
      autoPopupWelcome,
      setAutoPopupWelcome,
      autoScrollNewMessages,
      setAutoScrollNewMessages,
      displaySourcesInMessages,
      setDisplaySourcesInMessages,
      sourcesLabel,
      setSourcesLabel,
      suggestedMessages,
      setSuggestedMessages,
      continueWithoutSources,
      discoverUrls,
      stopDiscovery,
      toggleUrl,
      toggleCategory,
      selectAll,
      deselectAll,
      startTraining,
      resetFlow,
      businessType,
      setBusinessType,
      nextPath,
      prevPath,
    ]
  )

  return <CreateBotContext.Provider value={value}>{children}</CreateBotContext.Provider>
}

export function useCreateBotFlow() {
  const context = useContext(CreateBotContext)
  if (!context) {
    throw new Error('useCreateBotFlow must be used within CreateBotProvider')
  }
  return context
}
