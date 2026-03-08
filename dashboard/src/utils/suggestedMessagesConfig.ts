import type { SuggestedMessageConfig } from '../components/WidgetDesignForm'

export type SuggestedMessagesLanguage = 'en' | 'ja'

export const SUGGESTED_MESSAGES_LANGUAGES: SuggestedMessagesLanguage[] = ['en', 'ja']

function normalizeSuggestedMessagesLanguage(value: unknown): SuggestedMessagesLanguage {
  const raw = String(value || '').trim().toLowerCase()
  return raw === 'ja' || raw === 'jp' ? 'ja' : 'en'
}

function normalizeSuggestedMessageItem(raw: unknown, index: number): SuggestedMessageConfig | null {
  if (!raw || typeof raw !== 'object') return null
  const item = raw as Record<string, unknown>
  const label = typeof item.label === 'string' ? item.label.trim() : ''
  if (!label) return null
  const type =
    item.type === 'ai_response' || item.type === 'show_menu' || item.type === 'escalate'
      ? item.type
      : 'ai_response'
  const prompt = typeof item.prompt === 'string' ? item.prompt : undefined
  const message = typeof item.message === 'string' ? item.message : undefined
  const urls = Array.isArray(item.urls)
    ? item.urls.filter((value): value is string => typeof value === 'string').map((value) => value.trim()).filter(Boolean)
    : undefined
  return {
    id: String(item.id || `suggest_${index}`),
    label,
    type,
    prompt,
    message,
    urls,
  }
}

function normalizeSuggestedMessagesList(raw: unknown): SuggestedMessageConfig[] {
  if (!Array.isArray(raw)) return []
  return raw
    .map((item, index) => normalizeSuggestedMessageItem(item, index))
    .filter((item): item is SuggestedMessageConfig => Boolean(item))
}

export function getBotSuggestedMessagesBaseLanguage(config: Record<string, unknown> | null | undefined): SuggestedMessagesLanguage {
  return normalizeSuggestedMessagesLanguage(config?.language || config?.botLanguage || 'en')
}

export function getSuggestedMessagesByLanguageFromConfig(
  config: Record<string, unknown> | null | undefined
): Record<SuggestedMessagesLanguage, SuggestedMessageConfig[]> {
  const currentLang = getBotSuggestedMessagesBaseLanguage(config)
  const resolved: Record<SuggestedMessagesLanguage, SuggestedMessageConfig[]> = { en: [], ja: [] }
  const rawByLanguage = config && typeof config === 'object'
    ? (config as Record<string, unknown>).suggestedMessagesByLanguage
    : null

  if (rawByLanguage && typeof rawByLanguage === 'object' && !Array.isArray(rawByLanguage)) {
    for (const lang of SUGGESTED_MESSAGES_LANGUAGES) {
      resolved[lang] = normalizeSuggestedMessagesList((rawByLanguage as Record<string, unknown>)[lang])
    }
  }

  if (!resolved[currentLang].length) {
    const legacy = normalizeSuggestedMessagesList(config && typeof config === 'object'
      ? (config as Record<string, unknown>).suggestedMessages
      : null)
    if (legacy.length) {
      resolved[currentLang] = legacy
    }
  }

  return resolved
}

export function getSuggestedMessagesForLanguageFromConfig(
  config: Record<string, unknown> | null | undefined,
  lang: SuggestedMessagesLanguage
): SuggestedMessageConfig[] {
  return getSuggestedMessagesByLanguageFromConfig(config)[normalizeSuggestedMessagesLanguage(lang)]
}

export function setSuggestedMessagesForLanguageInConfig(
  config: Record<string, unknown> | null | undefined,
  lang: SuggestedMessagesLanguage,
  messages: SuggestedMessageConfig[]
): Record<string, unknown> {
  const base = config && typeof config === 'object' ? { ...config } : {}
  const currentLang = getBotSuggestedMessagesBaseLanguage(base)
  const byLanguage = getSuggestedMessagesByLanguageFromConfig(base)
  const normalizedLang = normalizeSuggestedMessagesLanguage(lang)
  byLanguage[normalizedLang] = messages.map((item) => ({
    ...item,
    urls: Array.isArray(item.urls) ? [...item.urls] : undefined,
  }))
  return {
    ...base,
    suggestedMessagesByLanguage: byLanguage,
    suggestedMessages: byLanguage[currentLang],
  }
}
