import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { GlassField, UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

function normalizeOptionalUrl(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return ''
  try {
    const normalized = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
    const parsed = new URL(normalized)
    if (!/^https?:$/i.test(parsed.protocol)) return ''
    return parsed.href
  } catch {
    return ''
  }
}

function interpolateNotice(template: string, replacements: Record<string, string>): string {
  return template.replace(/\{([a-z_]+)\}/gi, (_, key: string) => replacements[key] || '')
}

export default function CreateBotActionDestinationPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { flow, step2 } = useCreateBotFlow()
  const [localError, setLocalError] = useState<string | null>(null)
  const currentScreen = flow.currentScreen

  if (!currentScreen || currentScreen.component !== 'action_destination_url') {
    return null
  }

  const actionKey = currentScreen.actionKey || 'reservation'
  const currentValue = step2.actionDestinationLinks[actionKey] || ''
  const selectedPlatform = step2.platforms.find((platform) => platform.id === step2.reservationPlatform) || null
  const fallbackUrl = actionKey === 'reservation'
    ? String(step2.platformUrls[step2.reservationPlatform] || '').trim()
    : ''
  const fallbackNotice =
    currentScreen.fallbackNotice && fallbackUrl && selectedPlatform
      ? interpolateNotice(currentScreen.fallbackNotice, {
          platform_label: selectedPlatform.label,
          platform_url: fallbackUrl,
        }).trim()
      : ''

  const handleContinue = () => {
    setLocalError(null)
    if (!currentValue.trim()) {
      if (flow.nextPath) navigate(flow.nextPath)
      return
    }
    const normalized = normalizeOptionalUrl(currentValue)
    if (!normalized) {
      setLocalError(t('createBot.enterValidUrl', 'Enter a valid URL'))
      return
    }
    step2.setActionDestinationLink(actionKey, normalized)
    if (flow.nextPath) navigate(flow.nextPath)
  }

  const handleSkip = () => {
    setLocalError(null)
    step2.setActionDestinationLink(actionKey, '')
    if (flow.nextPath) navigate(flow.nextPath)
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">{currentScreen.title || t('createBot.reservationDestinationTitle', 'Choose where reservation taps should go')}</div>
        <div className="card-subtitle">
          {currentScreen.subtitle || t('createBot.reservationDestinationSubtitle', 'You can keep the selected platform URL, or set a customer-facing destination URL of your own.')}
        </div>
      </div>

      <GlassField
        label={currentScreen.fieldLabel || t('createBot.reservationDestinationLabel', 'Customer-facing reservation URL')}
        helper={currentScreen.fieldHelper || t('createBot.reservationDestinationHelper', 'Optional. If set, this is the reservation link customers receive in chat and action buttons.')}
      >
        <input
          type="url"
          value={currentValue}
          onChange={(event) => {
            setLocalError(null)
            step2.setActionDestinationLink(actionKey, event.target.value)
          }}
          placeholder={
            currentScreen.fieldPlaceholder || t('createBot.reservationDestinationPlaceholder', 'https://your-restaurant.com/reserve')
          }
        />
      </GlassField>

      {fallbackNotice && <div className="alert info">{fallbackNotice}</div>}
      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          {t('common.back', 'Back')}
        </UiButton>
        <div style={{ display: 'flex', gap: '0.75rem', marginLeft: 'auto' }}>
          <UiButton variant="ghost" onClick={handleSkip}>
            {t('createBot.skip', 'Skip for now')}
          </UiButton>
          <UiButton variant="primary" onClick={handleContinue}>
            {t('common.continue', 'Continue')}
          </UiButton>
        </div>
      </div>
    </div>
  )
}
