export const ADDITIONAL_SOURCES_RUN_STORAGE_PREFIX = 'dashboard.sources.additional_run.'
export const ADDITIONAL_SOURCES_RUN_TTL_MS = 2 * 60 * 60 * 1000

const ADDITIONAL_SUCCESS_TERMINAL_STAGES = new Set([
  'import_submitted',
  'prompt_queued',
  'prompt_generating',
  'done',
  'complete',
  'cancelled',
])

const ADDITIONAL_ERROR_STAGES = new Set(['error', 'failed'])

export type AdditionalSourcesRunPayload = {
  run_id: string
  bot_id: string
  job_ids: string[]
  created_at: number
  total_sources: number
}

function normalize(value: unknown): string {
  return String(value || '').trim()
}

function normalizeJobIds(values: unknown): string[] {
  if (!Array.isArray(values)) return []
  const cleaned = values
    .map((value) => normalize(value))
    .filter(Boolean)
  return Array.from(new Set(cleaned))
}

function storageKey(botId: string): string {
  return `${ADDITIONAL_SOURCES_RUN_STORAGE_PREFIX}${botId}`
}

function parsePayload(raw: unknown): AdditionalSourcesRunPayload | null {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as Record<string, unknown>
  const runId = normalize(record.run_id)
  const botId = normalize(record.bot_id)
  const jobIds = normalizeJobIds(record.job_ids)
  const createdAtRaw = Number(record.created_at || 0)
  const createdAt = Number.isFinite(createdAtRaw) && createdAtRaw > 0 ? Math.round(createdAtRaw) : 0
  const totalRaw = Number(record.total_sources || 0)
  const totalSources = Number.isFinite(totalRaw) && totalRaw > 0 ? Math.round(totalRaw) : jobIds.length
  if (!runId || !botId || jobIds.length === 0 || !createdAt) return null
  return {
    run_id: runId,
    bot_id: botId,
    job_ids: jobIds,
    created_at: createdAt,
    total_sources: Math.max(1, totalSources),
  }
}

export function isAdditionalSourcesStageSuccess(stage: unknown): boolean {
  const key = normalize(stage).toLowerCase()
  return ADDITIONAL_SUCCESS_TERMINAL_STAGES.has(key)
}

export function isAdditionalSourcesStageTerminal(stage: unknown): boolean {
  const key = normalize(stage).toLowerCase()
  if (!key) return false
  return ADDITIONAL_SUCCESS_TERMINAL_STAGES.has(key) || ADDITIONAL_ERROR_STAGES.has(key)
}

export function clearAdditionalSourcesRun(botId: string): void {
  if (typeof window === 'undefined' || !botId) return
  try {
    window.sessionStorage.removeItem(storageKey(botId))
  } catch {
    // Best effort only.
  }
}

export function readAdditionalSourcesRun(botId: string, nowMs: number = Date.now()): AdditionalSourcesRunPayload | null {
  if (typeof window === 'undefined' || !botId) return null
  try {
    const raw = window.sessionStorage.getItem(storageKey(botId))
    if (!raw) return null
    const parsed = parsePayload(JSON.parse(raw))
    if (!parsed) {
      clearAdditionalSourcesRun(botId)
      return null
    }
    if (Math.max(0, nowMs - parsed.created_at) > ADDITIONAL_SOURCES_RUN_TTL_MS) {
      clearAdditionalSourcesRun(botId)
      return null
    }
    return parsed
  } catch {
    clearAdditionalSourcesRun(botId)
    return null
  }
}

export function writeAdditionalSourcesRun(payload: AdditionalSourcesRunPayload): void {
  if (typeof window === 'undefined') return
  const parsed = parsePayload(payload)
  if (!parsed) return
  try {
    window.sessionStorage.setItem(storageKey(parsed.bot_id), JSON.stringify(parsed))
  } catch {
    // Best effort only.
  }
}
