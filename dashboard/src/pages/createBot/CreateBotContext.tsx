import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useDashboardData } from '../../hooks/useDashboardData'

type TrainingStage = 'idle' | 'training' | 'complete'

type CreateBotContextValue = {
  botName: string
  setBotName: (value: string) => void
  websiteUrl: string
  setWebsiteUrl: (value: string) => void
  discoveryMethod: string
  setDiscoveryMethod: (value: string) => void
  normalizedWebsiteUrl: string
  discoveredUrls: string[]
  selectedUrls: string[]
  isDiscovering: boolean
  isStartingTraining: boolean
  discoveryDurationMs: number | null
  trainingStage: TrainingStage
  trainingProgress: number
  trainingPagesCrawled: number
  trainingDocsCount: number
  trainingStageName: string
  botId: string | null
  jobId: string | null
  localError: string | null
  setLocalError: (value: string | null) => void
  discoverUrls: () => Promise<boolean>
  stopDiscovery: () => void
  toggleUrl: (url: string) => void
  toggleCategory: (categoryPath: string, categoryUrls: string[]) => void
  selectAll: () => void
  deselectAll: () => void
  startTraining: () => Promise<string | null>
  resetFlow: () => void
}

const CreateBotContext = createContext<CreateBotContextValue | undefined>(undefined)

function normalizeUrl(value: string) {
  const trimmed = value.trim()
  if (!trimmed) return ''
  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
  const parsed = new URL(withProtocol)
  return parsed.origin
}

export function CreateBotProvider({ children }: { children: React.ReactNode }) {
  const { createBot, discoverUrls: discoverUrlsFromHook, queueCrawlUrls, getJobStatus, setSelectedBotId, orgs, activeOrgId, isSuperAdmin } = useDashboardData()
  const [botName, setBotName] = useState('')
  const [websiteUrl, setWebsiteUrl] = useState('')
  const [discoveryMethod, setDiscoveryMethod] = useState('auto') // 'auto' (crawl4ai) or 'sitemap'
  const [normalizedWebsiteUrl, setNormalizedWebsiteUrl] = useState('')
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedUrls, setSelectedUrls] = useState<string[]>([])
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [isStartingTraining, setIsStartingTraining] = useState(false)
  const [discoveryDurationMs, setDiscoveryDurationMs] = useState<number | null>(null)
  const selectionTouchedRef = useRef(false)
  const discoveryStartTimeRef = useRef<number | null>(null)
  const discoveryAbortRef = useRef<AbortController | null>(null)
  const [trainingStage, setTrainingStage] = useState<TrainingStage>('idle')
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingPagesCrawled, setTrainingPagesCrawled] = useState(0)
  const [trainingDocsCount, setTrainingDocsCount] = useState(0)
  const [trainingStageName, setTrainingStageName] = useState('')
  const [botId, setBotId] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)

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
    setTrainingStage('idle')
    setTrainingProgress(0)
    setTrainingPagesCrawled(0)
    setTrainingDocsCount(0)
    setTrainingStageName('')
    setBotId(null)
    setJobId(null)
    setLocalError(null)
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
    selectionTouchedRef.current = false
    discoveryStartTimeRef.current = Date.now()

    const controller = new AbortController()
    discoveryAbortRef.current = controller

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
          const start = discoveryStartTimeRef.current
          if (start != null) setDiscoveryDurationMs(Date.now() - start)
          setIsDiscovering(false)
          if (Array.isArray((evt as { urls?: unknown }).urls) && ((evt as { urls?: unknown[] }).urls || []).length === 0) {
            setLocalError(
              discoveryMethod === 'sitemap'
                ? "Could not discover via sitemap. Switch to 'Automatic' (recommended)."
                : 'No URLs found for this site.'
            )
          }
        }
      }, controller.signal)
        .then((final) => {
          if (final && !final.urls?.length && final.error) setLocalError(final.error)
          const start = discoveryStartTimeRef.current
          if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
        })
        .catch((err: Error & { name?: string }) => {
          if (err.name === 'AbortError') {
            const start = discoveryStartTimeRef.current
            if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
          }
        })
        .finally(() => {
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
    return created.bot_id
  }, [botName, createBot, queueCrawlUrls, selectedUrls, setSelectedBotId, orgs, activeOrgId, isSuperAdmin])

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
    const timer = window.setInterval(pollStatus, 2000)
    return () => window.clearInterval(timer)
  }, [trainingStage, botId, jobId, getJobStatus, selectedUrls.length])

  const value = useMemo(
    () => ({
      botName,
      setBotName,
      websiteUrl,
      setWebsiteUrl,
      discoveryMethod,
      setDiscoveryMethod,
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
      setLocalError,
      discoverUrls,
      stopDiscovery,
      toggleUrl,
      toggleCategory,
      selectAll,
      deselectAll,
      startTraining,
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
      setLocalError,
      discoverUrls,
      stopDiscovery,
      toggleUrl,
      toggleCategory,
      selectAll,
      deselectAll,
      startTraining,
      resetFlow,
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
