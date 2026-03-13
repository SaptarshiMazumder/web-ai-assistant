import type { JobPipelineRunRecord, JobPipelineStepRecord } from '../../hooks/useDashboardData'

export type UnifiedSourcesProgressStepStatus = 'queued' | 'running' | 'paused' | 'done' | 'error'

export type UnifiedSourcesProgressStep = {
  id: string
  source: 'crawl' | 'pipeline'
  status: UnifiedSourcesProgressStepStatus
  progressPct: number
  labelKey: string
  labelFallback: string
  message?: string
  messageKey?: string
  messageFallback?: string
  details?: Record<string, unknown>
  error?: string
}

export type UnifiedSourcesProgress = {
  runKey: string
  steps: UnifiedSourcesProgressStep[]
  activeStepIndex: number
  activeStepProgressPct: number
  headlineMessage?: string
  headlineMessageKey?: string
  headlineMessageFallback?: string
  isTerminal: boolean
  isError: boolean
  showStop: boolean
  showResume: boolean
  activeCrawlStats?: {
    pagesCrawled: number
    docsCount: number
  }
}

export type BuildUnifiedSourcesProgressInput = {
  crawlJobId?: string | null
  crawlStatus?: CrawlProgressStatus | null
  pipelineRun?: JobPipelineRunRecord | null
  plannedPipelineJobIds?: string[] | null
  nowMs?: number
  importSubmittedGraceMs?: number
}

export type CrawlProgressStatus = {
  stage?: string | null
  pages_crawled?: number | null
  docs_count?: number | null
  last_error?: string | null
  updated_at?: string | null
}

const IMPORT_SUBMITTED_GRACE_MS = 30_000

const STEP_LABELS: Record<string, { key: string; fallback: string }> = {
  crawl_import: { key: 'botKnowledge.progressStepCrawlImport', fallback: 'Learning from website pages' },
  prompt_generation: { key: 'botKnowledge.progressStepPromptGeneration', fallback: 'Personalizing your assistant' },
  booking_link: { key: 'botKnowledge.progressStepBookingLink', fallback: 'Adding booking details' },
  asset_extraction: { key: 'botKnowledge.progressStepImageExtraction', fallback: 'Preparing your images' },
  menu_extraction: { key: 'botKnowledge.progressStepMenuExtraction', fallback: 'Preparing your menu and services' },
  reservation_url: { key: 'botKnowledge.progressStepReservationUrl', fallback: 'Setting up reservation links' },
  discovery: { key: 'botKnowledge.progressStepDiscovery', fallback: 'Finding pages to learn from' },
}

const STATUS_MESSAGES: Record<UnifiedSourcesProgressStepStatus, { key: string; fallback: string }> = {
  queued: { key: 'botKnowledge.progressStatusQueued', fallback: 'Queued' },
  running: { key: 'botKnowledge.progressStatusRunning', fallback: 'In progress' },
  paused: { key: 'botKnowledge.progressStatusPaused', fallback: 'Waiting for input' },
  done: { key: 'botKnowledge.progressStatusDone', fallback: 'Completed' },
  error: { key: 'botKnowledge.progressStatusError', fallback: 'Failed' },
}

const PIPELINE_STATUS_MAP: Record<string, UnifiedSourcesProgressStepStatus> = {
  queued: 'queued',
  running: 'running',
  paused: 'paused',
  done: 'done',
  complete: 'done',
  error: 'error',
  failed: 'error',
  cancelled: 'done',
}

const CRAWL_STAGE_META: Record<
  string,
  {
    status: UnifiedSourcesProgressStepStatus
    progress: number
    messageKey: string
    messageFallback: string
  }
> = {
  queued: {
    status: 'queued',
    progress: 8,
    messageKey: 'botKnowledge.progressCrawlQueued',
    messageFallback: 'Crawl queued',
  },
  pending: {
    status: 'queued',
    progress: 8,
    messageKey: 'botKnowledge.progressCrawlQueued',
    messageFallback: 'Crawl queued',
  },
  crawling: {
    status: 'running',
    progress: 35,
    messageKey: 'botKnowledge.progressCrawling',
    messageFallback: 'Scanning pages',
  },
  running: {
    status: 'running',
    progress: 35,
    messageKey: 'botKnowledge.progressCrawling',
    messageFallback: 'Scanning pages',
  },
  uploading: {
    status: 'running',
    progress: 65,
    messageKey: 'botKnowledge.progressUploading',
    messageFallback: 'Saving crawled data',
  },
  importing: {
    status: 'running',
    progress: 85,
    messageKey: 'botKnowledge.progressImporting',
    messageFallback: 'Importing into knowledge base',
  },
  prompt_queued: {
    status: 'running',
    progress: 90,
    messageKey: 'botKnowledge.progressPromptQueued',
    messageFallback: 'Prompt generation queued',
  },
  prompt_generating: {
    status: 'running',
    progress: 95,
    messageKey: 'botKnowledge.progressPromptGenerating',
    messageFallback: 'Preparing bot prompt',
  },
  done: {
    status: 'done',
    progress: 100,
    messageKey: 'botKnowledge.progressCrawlDone',
    messageFallback: 'Crawl and import completed',
  },
  complete: {
    status: 'done',
    progress: 100,
    messageKey: 'botKnowledge.progressCrawlDone',
    messageFallback: 'Crawl and import completed',
  },
  error: {
    status: 'error',
    progress: 100,
    messageKey: 'botKnowledge.progressCrawlError',
    messageFallback: 'Crawl or import failed',
  },
  failed: {
    status: 'error',
    progress: 100,
    messageKey: 'botKnowledge.progressCrawlError',
    messageFallback: 'Crawl or import failed',
  },
  cancelled: {
    status: 'done',
    progress: 100,
    messageKey: 'botKnowledge.progressCrawlCancelled',
    messageFallback: 'Training stopped',
  },
}

function isUserCancelledMessage(...values: Array<unknown>): boolean {
  for (const value of values) {
    const text = String(value || '').trim().toLowerCase()
    if (!text) continue
    if (text.includes('cancelled by user') || text.includes('canceled by user')) return true
  }
  return false
}

function clampPct(value: unknown): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return 0
  if (n <= 0) return 0
  if (n >= 100) return 100
  return Math.round(n)
}

function normalize(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

function formatStepId(stepId: string): string {
  const raw = String(stepId || '').trim()
  if (!raw) return 'Training in progress'
  return raw
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function stepLabel(stepId: string): { key: string; fallback: string } {
  const key = normalize(stepId)
  const found = STEP_LABELS[key]
  if (found) return found
  const safeDynamicKey = key.replace(/[^a-z0-9_]/g, '_') || 'unknown'
  return {
    // Use a non-existent dynamic key so i18n falls back to the readable step text.
    key: `botKnowledge.progressStepDynamic.${safeDynamicKey}`,
    fallback: formatStepId(stepId),
  }
}

function statusMessage(status: UnifiedSourcesProgressStepStatus): { key: string; fallback: string } {
  return STATUS_MESSAGES[status] || STATUS_MESSAGES.running
}

function isPipelineRunForCrawlJob(run: JobPipelineRunRecord, crawlJobId: string): boolean {
  if (!crawlJobId) return true
  const context = run.context && typeof run.context === 'object' ? run.context : {}
  const runIndexJobId = String((context as Record<string, unknown>).index_job_id || '').trim()
  if (!runIndexJobId) return true
  return runIndexJobId === crawlJobId
}

function resolveCrawlStep(
  crawlStatus: CrawlProgressStatus,
  options: {
    pipelineRun: JobPipelineRunRecord | null
    nowMs: number
    importSubmittedGraceMs: number
  },
): UnifiedSourcesProgressStep {
  const { pipelineRun, nowMs, importSubmittedGraceMs } = options
  const stage = normalize(crawlStatus.stage)
  const label = stepLabel('crawl_import')
  const lastError = String(crawlStatus.last_error || '').trim() || undefined

  if (stage === 'import_submitted') {
    const updatedMs = crawlStatus.updated_at ? new Date(crawlStatus.updated_at).getTime() : 0
    const beyondGrace = updatedMs > 0 ? nowMs - updatedMs >= importSubmittedGraceMs : false
    const shouldMarkDone = Boolean(pipelineRun) || beyondGrace
    if (shouldMarkDone) {
      const doneMsg = statusMessage('done')
      return {
        id: 'crawl_import',
        source: 'crawl',
        status: 'done',
        progressPct: 100,
        labelKey: label.key,
        labelFallback: label.fallback,
        messageKey: doneMsg.key,
        messageFallback: doneMsg.fallback,
      }
    }
    return {
      id: 'crawl_import',
      source: 'crawl',
      status: 'running',
      progressPct: 95,
      labelKey: label.key,
      labelFallback: label.fallback,
      messageKey: 'botKnowledge.progressWaitingForPipeline',
      messageFallback: 'Finishing this training step',
    }
  }

  const meta = CRAWL_STAGE_META[stage] || CRAWL_STAGE_META.running
  return {
    id: 'crawl_import',
    source: 'crawl',
    status: meta.status,
    progressPct: clampPct(meta.progress),
    labelKey: label.key,
    labelFallback: label.fallback,
    messageKey: meta.messageKey,
    messageFallback: meta.messageFallback,
    error: meta.status === 'error' ? lastError : undefined,
  }
}

function resolvePipelineStep(step: JobPipelineStepRecord): UnifiedSourcesProgressStep {
  const normalizedStatus = normalize(step.status)
  const cancelledByUser = isUserCancelledMessage(step.current_message, step.last_error)
  const status = cancelledByUser ? 'done' : (PIPELINE_STATUS_MAP[normalizedStatus] || 'running')
  const label = stepLabel(step.job_id)
  const message = String(step.current_message || '').trim() || undefined
  const rawOutput = step.output && typeof step.output === 'object' ? step.output : {}
  const completion =
    rawOutput &&
    typeof rawOutput === 'object' &&
    typeof (rawOutput as Record<string, unknown>).completion === 'object' &&
    (rawOutput as Record<string, unknown>).completion !== null
      ? ((rawOutput as Record<string, unknown>).completion as Record<string, unknown>)
      : null
  const details =
    completion &&
    typeof completion.details === 'object' &&
    completion.details !== null
      ? (completion.details as Record<string, unknown>)
      : undefined
  const statusMsg = statusMessage(status)
  const error = status === 'error' ? String(step.last_error || '').trim() || undefined : undefined
  return {
    id: String(step.job_id || `step_${step.step_index}`),
      source: 'pipeline',
      status,
    progressPct: status === 'done' ? 100 : clampPct(step.progress_pct),
    labelKey: label.key,
    labelFallback: label.fallback,
    message,
    messageKey: message ? undefined : statusMsg.key,
    messageFallback: message ? undefined : statusMsg.fallback,
    details,
    error,
  }
}

function resolvePlannedPipelineStep(jobId: string): UnifiedSourcesProgressStep {
  const label = stepLabel(jobId)
  const queuedStatus = statusMessage('queued')
  return {
    id: jobId || 'pipeline_step',
    source: 'pipeline',
    status: 'queued',
    progressPct: 0,
    labelKey: label.key,
    labelFallback: label.fallback,
    messageKey: queuedStatus.key,
    messageFallback: queuedStatus.fallback,
  }
}

export function buildUnifiedSourcesProgress(input: BuildUnifiedSourcesProgressInput): UnifiedSourcesProgress | null {
  const crawlStatus = input.crawlStatus || null
  const crawlJobId = String(input.crawlJobId || '').trim()
  const nowMs = Number.isFinite(input.nowMs) ? Number(input.nowMs) : Date.now()
  const importSubmittedGraceMs = input.importSubmittedGraceMs ?? IMPORT_SUBMITTED_GRACE_MS
  const plannedPipelineJobIds = Array.isArray(input.plannedPipelineJobIds)
    ? Array.from(
        new Set(
          input.plannedPipelineJobIds
            .map((jobId) => String(jobId || '').trim())
            .filter(Boolean)
        )
      )
    : []

  const candidateRun = input.pipelineRun || null
  const pipelineRun =
    candidateRun && (!crawlJobId || isPipelineRunForCrawlJob(candidateRun, crawlJobId))
      ? candidateRun
      : null

  const steps: UnifiedSourcesProgressStep[] = []
  if (crawlStatus && crawlJobId) {
    steps.push(
      resolveCrawlStep(crawlStatus, {
        pipelineRun,
        nowMs,
        importSubmittedGraceMs,
      })
    )
  }

  const crawlStep = steps.find((step) => step.source === 'crawl')

  if (pipelineRun) {
    const sortedSteps = [...(pipelineRun.steps || [])].sort((a, b) => a.step_index - b.step_index)
    if (sortedSteps.length > 0) {
      for (const step of sortedSteps) {
        steps.push(resolvePipelineStep(step))
      }
    } else {
      const syntheticCancelledByUser = isUserCancelledMessage(
        pipelineRun.current_message,
        pipelineRun.last_error,
      )
      const syntheticStatus = syntheticCancelledByUser
        ? 'done'
        : (PIPELINE_STATUS_MAP[normalize(pipelineRun.status)] || 'running')
      const syntheticLabel = stepLabel('pipeline')
      const syntheticStatusMsg = statusMessage(syntheticStatus)
      const message = String(pipelineRun.current_message || '').trim() || undefined
      steps.push({
        id: 'pipeline',
        source: 'pipeline',
        status: syntheticStatus,
        progressPct: clampPct(pipelineRun.progress_pct),
        labelKey: syntheticLabel.key,
        labelFallback: syntheticLabel.fallback,
        message,
        messageKey: message ? undefined : syntheticStatusMsg.key,
        messageFallback: message ? undefined : syntheticStatusMsg.fallback,
        error:
          syntheticStatus === 'error'
            ? String(pipelineRun.last_error || '').trim() || undefined
            : undefined,
      })
    }
  } else if (crawlStep && crawlStep.status !== 'error' && plannedPipelineJobIds.length > 0) {
    for (const jobId of plannedPipelineJobIds) {
      steps.push(resolvePlannedPipelineStep(jobId))
    }
  }

  if (steps.length === 0) return null

  let activeStepIndex = steps.findIndex((step) => step.status === 'running' || step.status === 'queued' || step.status === 'paused')
  if (activeStepIndex < 0) {
    activeStepIndex = steps.findIndex((step) => step.status === 'error')
  }
  if (activeStepIndex < 0) {
    activeStepIndex = Math.max(0, steps.length - 1)
  }

  const activeStep = steps[activeStepIndex]
  const isTerminal = steps.every((step) => step.status === 'done' || step.status === 'error')
  const isError = steps.some((step) => step.status === 'error')

  let headlineMessage = activeStep.message
  let headlineMessageKey = activeStep.messageKey
  let headlineMessageFallback = activeStep.messageFallback

  if (isTerminal && !isError) {
    const runMessage = String(pipelineRun?.current_message || '').trim()
    if (runMessage) {
      headlineMessage = runMessage
      headlineMessageKey = undefined
      headlineMessageFallback = undefined
    } else {
      headlineMessage = undefined
      headlineMessageKey = 'botKnowledge.progressCompleted'
      headlineMessageFallback = 'All steps completed'
    }
  } else if (isTerminal && isError) {
    const stepError = steps.find((step) => step.status === 'error')?.error
    const runError = String(pipelineRun?.last_error || '').trim()
    const crawlError = String(crawlStatus?.last_error || '').trim()
    headlineMessage = stepError || runError || crawlError || undefined
    headlineMessageKey = headlineMessage ? undefined : 'botKnowledge.progressFailed'
    headlineMessageFallback = headlineMessage ? undefined : 'Progress failed'
  }

  const activeCrawlStats =
    activeStep.source === 'crawl'
      ? {
          pagesCrawled: Number(crawlStatus?.pages_crawled || 0),
          docsCount: Number(crawlStatus?.docs_count || 0),
        }
      : undefined

  const runKey = pipelineRun?.run_id
    ? `pipeline:${String(pipelineRun.run_id).trim()}`
    : crawlJobId
      ? `crawl:${crawlJobId}`
      : 'none'
  const pipelineStatus = normalize(pipelineRun?.status)

  return {
    runKey,
    steps,
    activeStepIndex,
    activeStepProgressPct: clampPct(activeStep.progressPct),
    headlineMessage,
    headlineMessageKey,
    headlineMessageFallback,
    isTerminal,
    isError,
    showStop: !isTerminal && activeStep.source === 'crawl' && (activeStep.status === 'queued' || activeStep.status === 'running'),
    showResume: pipelineStatus === 'paused',
    activeCrawlStats,
  }
}
