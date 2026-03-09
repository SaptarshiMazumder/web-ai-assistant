export type WelcomeMessageLanguage = 'en' | 'ja'
export type WelcomeMessageChannel = 'web' | 'line'

export type WelcomeMessagesByChannel = Record<WelcomeMessageChannel, Record<WelcomeMessageLanguage, string>>

const SUPPORTED_LANGUAGES: WelcomeMessageLanguage[] = ['en', 'ja']
const SUPPORTED_CHANNELS: WelcomeMessageChannel[] = ['web', 'line']

export function getBotWelcomeMessagesBaseLanguage(config: Record<string, unknown> | null | undefined): WelcomeMessageLanguage {
  const raw = String(config?.language ?? config?.botLanguage ?? 'en').trim().toLowerCase()
  return raw === 'ja' || raw === 'jp' ? 'ja' : 'en'
}

export function getWelcomeMessagesByChannelFromConfig(
  config: Record<string, unknown> | null | undefined
): WelcomeMessagesByChannel {
  const baseLanguage = getBotWelcomeMessagesBaseLanguage(config)
  const rawByChannel = config?.welcomeMessagesByChannel
  const resolved = {
    web: { en: '', ja: '' },
    line: { en: '', ja: '' },
  } satisfies WelcomeMessagesByChannel

  for (const channel of SUPPORTED_CHANNELS) {
    const rawChannel =
      rawByChannel && typeof rawByChannel === 'object' && !Array.isArray(rawByChannel)
        ? (rawByChannel as Record<string, unknown>)[channel]
        : null

    for (const language of SUPPORTED_LANGUAGES) {
      const rawValue =
        rawChannel && typeof rawChannel === 'object' && !Array.isArray(rawChannel)
          ? (rawChannel as Record<string, unknown>)[language]
          : null
      if (typeof rawValue === 'string' && rawValue.trim()) {
        resolved[channel][language] = rawValue
      }
    }
  }

  const legacyWelcome = typeof config?.welcomeMessage === 'string' ? config.welcomeMessage.trim() : ''
  if (legacyWelcome && !resolved.web[baseLanguage]) {
    resolved.web[baseLanguage] = legacyWelcome
  }

  return resolved
}

export function getWelcomeMessageForChannelFromConfig(
  config: Record<string, unknown> | null | undefined,
  channel: WelcomeMessageChannel,
  language: WelcomeMessageLanguage
): string {
  return getWelcomeMessagesByChannelFromConfig(config)[channel][language] || ''
}

export function setWelcomeMessageForChannelInConfig(
  config: Record<string, unknown>,
  channel: WelcomeMessageChannel,
  language: WelcomeMessageLanguage,
  value: string
): Record<string, unknown> {
  const normalized = getWelcomeMessagesByChannelFromConfig(config)
  normalized[channel][language] = value
  const baseLanguage = getBotWelcomeMessagesBaseLanguage(config)
  return {
    ...(config || {}),
    welcomeMessagesByChannel: normalized,
    welcomeMessage: normalized.web[baseLanguage] || undefined,
  }
}
