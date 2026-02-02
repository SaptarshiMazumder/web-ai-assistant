import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import { CREATE_BOT_FIRST_PATH, getCreateBotNextPath, getCreateBotPrevPath } from './flowConfig'

type TrainingStage = 'idle' | 'training' | 'complete'

/** Step 1: Name + Website. Change only this slice when editing the first step. */
export type CreateBotStep1Slice = {
  botName: string
  setBotName: (value: string) => void
  websiteUrl: string
  setWebsiteUrl: (value: string) => void
  discoveryMethod: string
  setDiscoveryMethod: (value: string) => void
  normalizedWebsiteUrl: string
  isDiscovering: boolean
  discoveryDurationMs: number | null
  discoveryTimedOutMessage: string | null
  localError: string | null
  setLocalError: (value: string | null) => void
  discoverUrls: () => Promise<boolean>
  stopDiscovery: () => void
}

/** Step 2: Select URLs. Change only this slice when editing the second step. */
export type CreateBotStep2Slice = {
  discoveredUrls: string[]
  selectedUrls: string[]
  normalizedWebsiteUrl: string
  discoveryDurationMs: number | null
  discoveryTimedOutMessage: string | null
  isDiscovering: boolean
  isStartingTraining: boolean
  localError: string | null
  setLocalError: (value: string | null) => void
  toggleUrl: (url: string) => void
  toggleCategory: (categoryPath: string, categoryUrls: string[]) => void
  selectAll: () => void
  deselectAll: () => void
  startTraining: () => Promise<string | null>
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
  return parsed.origin
}

export function CreateBotProvider({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const { createBot, discoverUrls: discoverUrlsFromHook, queueCrawlUrls, startBackgroundDiscovery, getJobStatus, setSelectedBotId, orgs, activeOrgId, isSuperAdmin } = useDashboardData()
  const [botName, setBotName] = useState('')
  const [websiteUrl, setWebsiteUrl] = useState('')
  const [discoveryMethod, setDiscoveryMethod] = useState('auto') // 'auto' (crawl4ai) or 'sitemap'
  const [normalizedWebsiteUrl, setNormalizedWebsiteUrl] = useState('')
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedUrls, setSelectedUrls] = useState<string[]>([])
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [isStartingTraining, setIsStartingTraining] = useState(false)
  const [discoveryDurationMs, setDiscoveryDurationMs] = useState<number | null>(null)
  const [discoveryTimedOutMessage, setDiscoveryTimedOutMessage] = useState<string | null>(null)
  const selectionTouchedRef = useRef(false)
  const discoveryStartTimeRef = useRef<number | null>(null)
  const discoveryAbortRef = useRef<AbortController | null>(null)
  const discovery60sTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const discoveryTimedOutByTimerRef = useRef(false)
  const [trainingStage, setTrainingStage] = useState<TrainingStage>('idle')
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingPagesCrawled, setTrainingPagesCrawled] = useState(0)
  const [trainingDocsCount, setTrainingDocsCount] = useState(0)
  const [trainingStageName, setTrainingStageName] = useState('')
  const [botId, setBotId] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
  const [widgetPosition, setWidgetPosition] = useState<'bottom-right' | 'bottom-left'>('bottom-right')
  const [widgetPrimaryColor, setWidgetPrimaryColor] = useState('#6366f1')
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
  const [maxHeight, setMaxHeight] = useState(720)
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [headerSize, setHeaderSize] = useState<'small' | 'medium' | 'large'>('small')
  const [autoPopupWelcome, setAutoPopupWelcome] = useState<'off' | '1s' | '2s' | '3s'>('off')
  const [autoScrollNewMessages, setAutoScrollNewMessages] = useState(true)
  const [displaySourcesInMessages, setDisplaySourcesInMessages] = useState(false)
  const [sourcesLabel, setSourcesLabel] = useState('Sources')
  const resetFlow = useCallback(() => {
    setBotName('')
    setWebsiteUrl('')
    setDiscoveryMethod('auto')
    setNormalizedWebsiteUrl('')
    setDiscoveredUrls([])
    setSelectedUrls([])
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
    setLocalError(null)
    setWidgetPosition('bottom-right')
    setWidgetPrimaryColor('#6366f1')
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
    setMaxHeight(720)
    setFontSize('medium')
    setHeaderSize('small')
    setAutoPopupWelcome('off')
    setAutoScrollNewMessages(true)
    setDisplaySourcesInMessages(false)
    setSourcesLabel('Sources')
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

    // Client-side 60s cap: when user presses Discover, we stop reading the stream after 60s and use what we have.
    discovery60sTimerRef.current = setTimeout(() => {
      discovery60sTimerRef.current = null
      discoveryTimedOutByTimerRef.current = true
      controller.abort()
    }, 60_000)

    // Fire-and-forget stream so UI can navigate immediately and update progressively.
    void (async () => {
      await discoverUrlsFromHook(normalized, discoveryMethod, (evt) => {
        if (evt.type === 'discovered' && typeof evt.url === 'string') {
          const url = evt.url
          setDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))

          if (!selectionTouchedRef.current) {
            setSelectedUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
          }
        }

        if (evt.type === 'error' && typeof evt.message === 'string') {
          setLocalError(evt.message)
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
            setDiscoveryTimedOutMessage("Found main URLs. You can train on these now; we'll discover more in the background.")
          }
          if (Array.isArray((evt as { urls?: unknown }).urls) && ((evt as { urls?: unknown[] }).urls || []).length === 0) {
            setLocalError(
              discoveryMethod === 'sitemap'
                ? "Could not discover via sitemap. Switch to 'Automatic' (recommended)."
                : 'No URLs found for this site.'
            )
          }
        }
      }, controller.signal, { max_duration_sec: 60 })
        .then((final) => {
          if (final && !final.urls?.length && final.error) setLocalError(final.error)
          const start = discoveryStartTimeRef.current
          if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
        })
        .catch((err: Error & { name?: string }) => {
          if (err.name === 'AbortError') {
            const start = discoveryStartTimeRef.current
            if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
            if (discoveryTimedOutByTimerRef.current) {
              setDiscoveryTimedOutMessage("Found main URLs. You can train on these now; we'll discover more in the background.")
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
        })
    })()

    // Return true so the UI can move to the URLs page immediately.
    return true
  }, [botName, websiteUrl, discoveryMethod, discoverUrlsFromHook])

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

  const startTraining = useCallback(async () => {
    setLocalError(null)
    if (!botName.trim()) {
      setLocalError('Enter a bot name to continue.')
      return null
    }
    if (!selectedUrls.length) {
      setLocalError('Select at least one URL to train on.')
      return null
    }
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
    setTrainingStage('training')
    setTrainingProgress(0)
    setTrainingPagesCrawled(0)
    setTrainingDocsCount(0)
    setTrainingStageName('crawling')
    void queueCrawlUrls(created.bot_id, selectedUrls)
      .then((jobIdResult) => {
        if (jobIdResult) setJobId(jobIdResult)
        else setLocalError('Failed to start crawl job')
      })
      .catch(() => {})
      .finally(() => setIsStartingTraining(false))
    void startBackgroundDiscovery(created.bot_id, normalizedWebsiteUrl, discoveryMethod)
    return created.bot_id
  }, [botName, createBot, queueCrawlUrls, startBackgroundDiscovery, selectedUrls, normalizedWebsiteUrl, discoveryMethod, setSelectedBotId, orgs, activeOrgId, isSuperAdmin])

  useEffect(() => {
    if (trainingStage !== 'training' || !botId || !jobId) return
    const pollStatus = async () => {
      const status = await getJobStatus(botId, jobId)
      if (!status) return
      setTrainingStageName(status.stage || 'crawling')
      setTrainingPagesCrawled(status.pages_crawled || 0)
      setTrainingDocsCount(status.docs_count || 0)
      if (status.stage === 'done' || status.stage === 'import_submitted') {
        setTrainingStage('complete')
        setTrainingProgress(100)
      } else if (status.stage === 'error') {
        setTrainingStage('complete')
        setLocalError(status.last_error || 'Training failed')
      } else {
        const totalUrls = selectedUrls.length
        if (totalUrls > 0 && status.pages_crawled) {
          const progress = Math.min(Math.round((status.pages_crawled / totalUrls) * 100), 95)
          setTrainingProgress(progress)
        }
      }
    }
    pollStatus()
    const timer = window.setInterval(pollStatus, 1500)
    return () => window.clearInterval(timer)
  }, [trainingStage, botId, jobId, getJobStatus, selectedUrls.length])

  const nextPath = getCreateBotNextPath(location.pathname)
  const prevPath = getCreateBotPrevPath(location.pathname)

  const value = useMemo(
    () => ({
      step1: {
        botName,
        setBotName,
        websiteUrl,
        setWebsiteUrl,
        discoveryMethod,
        setDiscoveryMethod,
        normalizedWebsiteUrl,
        isDiscovering,
        discoveryDurationMs,
        discoveryTimedOutMessage,
        localError,
        setLocalError,
        discoverUrls,
        stopDiscovery,
      },
      step2: {
        discoveredUrls,
        selectedUrls,
        normalizedWebsiteUrl,
        discoveryDurationMs,
        discoveryTimedOutMessage,
        isDiscovering,
        isStartingTraining,
        localError,
        setLocalError,
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
      websiteUrl,
      discoveryMethod,
      normalizedWebsiteUrl,
      discoveredUrls,
      selectedUrls,
      isDiscovering,
      isStartingTraining,
      discoveryDurationMs,
      trainingStage,
      trainingProgress,
      trainingPagesCrawled,
      trainingDocsCount,
      trainingStageName,
      botId,
      jobId,
      localError,
      widgetPosition,
      widgetPrimaryColor,
      widgetTitle,
      widgetSize,
      welcomeMessage,
      placeholder,
      footerMessage,
      theme,
      textColor,
      launcherIconUrl,
      launcherText,
      headerIconUrl,
      shareIconUrl,
      maxHeight,
      fontSize,
      headerSize,
      autoPopupWelcome,
      autoScrollNewMessages,
      displaySourcesInMessages,
      sourcesLabel,
      discoverUrls,
      stopDiscovery,
      toggleUrl,
      toggleCategory,
      selectAll,
      deselectAll,
      startTraining,
      resetFlow,
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
