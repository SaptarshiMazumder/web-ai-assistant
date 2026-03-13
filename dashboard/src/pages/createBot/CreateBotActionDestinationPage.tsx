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

  const platformLabel = selectedPlatform?.label || ''

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">
          {t('createBot.reservationDestinationTitle', 'Where should customers make reservations?')}
        </div>
        <div className="card-subtitle">
          {t('createBot.reservationDestinationSubtitle', 'When a customer asks your agent about making a reservation, where should they be sent?')}
        </div>
      </div>

      {fallbackUrl && selectedPlatform && (
        <div
          style={{
            border: '1px solid var(--flow-border)',
            borderRadius: 'var(--flow-radius)',
            padding: '1rem',
            background: 'var(--flow-surface)',
          }}
        >
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--flow-heading)', marginBottom: '0.35rem' }}>
            {t('createBot.currentReservationLink', 'Current reservation link')}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--flow-muted)', wordBreak: 'break-all' }}>
            {platformLabel}: {fallbackUrl}
          </div>
        </div>
      )}

      <GlassField
        label={t('createBot.reservationDestinationLabel', 'Use a different link instead (optional)')}
        helper={t('createBot.reservationDestinationHelper', 'Only fill this in if you want customers to go somewhere other than the link above.')}
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

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          {t('common.back', 'Back')}
        </UiButton>
        <UiButton variant="primary" onClick={handleContinue} style={{ marginLeft: 'auto' }}>
          {t('common.continue', 'Continue')}
        </UiButton>
      </div>
    </div>
  )
}
