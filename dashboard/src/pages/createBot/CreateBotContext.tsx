import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { useDashboardData } from '../../hooks/useDashboardData'

type TrainingStage = 'idle' | 'training' | 'complete'

type CreateBotContextValue = {
  botName: string
  setBotName: (value: string) => void
  websiteUrl: string
  setWebsiteUrl: (value: string) => void
  normalizedWebsiteUrl: string
  discoveredUrls: string[]
  selectedUrls: string[]
  isDiscovering: boolean
  trainingStage: TrainingStage
  trainingProgress: number
  trainingPagesCrawled: number
  trainingDocsCount: number
  trainingStageName: string
  botId: string | null
  jobId: string | null
  localError: string | null
  discoverUrls: () => Promise<boolean>
  toggleUrl: (url: string) => void
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
  const { createBot, discoverUrls: discoverUrlsFromHook, queueCrawlUrls, getJobStatus, setNewBotName, setSelectedBotId } = useDashboardData()
  const [botName, setBotName] = useState('')
  const [websiteUrl, setWebsiteUrl] = useState('')
  const [normalizedWebsiteUrl, setNormalizedWebsiteUrl] = useState('')
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedUrls, setSelectedUrls] = useState<string[]>([])
  const [isDiscovering, setIsDiscovering] = useState(false)
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
    setNormalizedWebsiteUrl('')
    setDiscoveredUrls([])
    setSelectedUrls([])
    setIsDiscovering(false)
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
    const urls = await discoverUrlsFromHook(normalized)
    if (!urls.length) {
      setLocalError('No URLs found for this site.')
      setDiscoveredUrls([])
      setSelectedUrls([])
      setIsDiscovering(false)
      return false
    }
    setDiscoveredUrls(urls)
    setSelectedUrls(urls)
    setIsDiscovering(false)
    return true
  }, [botName, websiteUrl, discoverUrlsFromHook])

  const toggleUrl = useCallback((url: string) => {
    setSelectedUrls((prev) => (prev.includes(url) ? prev.filter((item) => item !== url) : [...prev, url]))
  }, [])

  const selectAll = useCallback(() => {
    setSelectedUrls(discoveredUrls)
  }, [discoveredUrls])

  const deselectAll = useCallback(() => {
    setSelectedUrls([])
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
    setNewBotName(botName.trim())
    const created = await createBot()
    if (!created) {
      return null
    }
    setBotId(created.bot_id)
    setSelectedBotId(created.bot_id)
    setTrainingStage('training')
    setTrainingProgress(0)
    setTrainingPagesCrawled(0)
    setTrainingDocsCount(0)
    setTrainingStageName('crawling')
    const jobIdResult = await queueCrawlUrls(created.bot_id, selectedUrls)
    if (jobIdResult) {
      setJobId(jobIdResult)
    } else {
      setLocalError('Failed to start crawl job')
    }
    return created.bot_id
  }, [botName, createBot, queueCrawlUrls, selectedUrls, setNewBotName, setSelectedBotId])

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
    const timer = window.setInterval(pollStatus, 3000)
    return () => window.clearInterval(timer)
  }, [trainingStage, botId, jobId, getJobStatus, selectedUrls.length])

  const value = useMemo(
    () => ({
      botName,
      setBotName,
      websiteUrl,
      setWebsiteUrl,
      normalizedWebsiteUrl,
      discoveredUrls,
      selectedUrls,
      isDiscovering,
      trainingStage,
      trainingProgress,
      trainingPagesCrawled,
      trainingDocsCount,
      trainingStageName,
      botId,
      jobId,
      localError,
      discoverUrls,
      toggleUrl,
      selectAll,
      deselectAll,
      startTraining,
      resetFlow,
    }),
    [
      botName,
      websiteUrl,
      normalizedWebsiteUrl,
      discoveredUrls,
      selectedUrls,
      isDiscovering,
      trainingStage,
      trainingProgress,
      trainingPagesCrawled,
      trainingDocsCount,
      trainingStageName,
      botId,
      jobId,
      localError,
      discoverUrls,
      toggleUrl,
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
