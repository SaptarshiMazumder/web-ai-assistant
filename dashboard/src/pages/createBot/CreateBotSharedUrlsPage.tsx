import { useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { FlowIcon } from '../../components/FlowIcon'
import { GlassField, UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

type SuggestedBubble = { key: string; label: string; description: string }

export default function CreateBotSharedUrlsPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { step1, step2, flow } = useCreateBotFlow()
  const { businessType } = step1
  const {
    sharedUrlRows,
    setSharedUrlRows,
    localError,
    platforms,
  } = step2
  const urlInputRefs = useRef<Record<number, HTMLInputElement | null>>({})
  const normalizeLabelKey = (s: string) => (s || '').trim().toLowerCase()

  const suggestedBubbles: SuggestedBubble[] = useMemo(() => {
    const base: SuggestedBubble[] = [
      {
        key: 'pricing',
        label: t('createBot.quickTopicPricing', 'Pricing'),
        description: t('createBot.quickTopicPricingDesc', 'Plans, packages, pricing tables'),
      },
      {
        key: 'location',
        label: t('createBot.quickTopicLocation', 'Location'),
        description: t('createBot.quickTopicLocationDesc', 'Address, map, directions'),
      },
      {
        key: 'hours',
        label: t('createBot.quickTopicHours', 'Hours'),
        description: t('createBot.quickTopicHoursDesc', 'Opening hours, business hours'),
      },
      {
        key: 'contact',
        label: t('createBot.quickTopicContact', 'Contact'),
        description: t('createBot.quickTopicContactDesc', 'Phone, email, contact form'),
      },
      {
        key: 'services',
        label: t('createBot.quickTopicServices', 'Services'),
        description: t('createBot.quickTopicServicesDesc', 'Service menu and details'),
      },
      {
        key: 'faq',
        label: t('createBot.quickTopicFaq', 'FAQ'),
        description: t('createBot.quickTopicFaqDesc', 'Common questions and answers'),
      },
      {
        key: 'policies',
        label: t('createBot.quickTopicPolicies', 'Policies'),
        description: t('createBot.quickTopicPoliciesDesc', 'Refunds, cancellation, terms'),
      },
    ]
    if (businessType === 'hotel') {
      base.unshift(
        {
          key: 'availability',
          label: t('createBot.quickTopicAvailability', 'Availability'),
          description: t('createBot.quickTopicAvailabilityDesc', 'Availability & rates page'),
        },
        {
          key: 'booking',
          label: t('createBot.quickTopicBooking', 'Booking'),
          description: t('createBot.quickTopicBookingDesc', 'Booking / reservation page'),
        }
      )
    }
    if (businessType === 'restaurant') {
      for (const p of platforms) {
        base.unshift({ key: p.id, label: p.label, description: p.url_placeholder || `${p.label} page` })
      }
      base.unshift(
        { key: 'menu', label: t('createBot.quickTopicMenu', 'Menu'), description: t('createBot.quickTopicMenuDesc', 'Food and drink menu') },
        { key: 'reservation', label: t('createBot.quickTopicReservation', 'Reservation'), description: t('createBot.quickTopicReservationDesc', 'Reservation or booking page') }
      )
    }
    return base
  }, [businessType, platforms, t])

  const bubbleHasFilledUrl = (bubbleLabel: string) => {
    const k = normalizeLabelKey(bubbleLabel)
    return sharedUrlRows.some((r) => normalizeLabelKey(r.label) === k && !!r.url.trim())
  }

  const focusRowUrl = (idx: number) => {
    window.setTimeout(() => {
      const el = urlInputRefs.current[idx]
      if (el) {
        el.focus()
        el.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
      }
    }, 0)
  }

  const handleBubbleClick = (bubbleLabel: string) => {
    const k = normalizeLabelKey(bubbleLabel)
    const existingIdx = sharedUrlRows.findIndex((r) => normalizeLabelKey(r.label) === k && !r.url.trim())
    if (existingIdx >= 0) {
      focusRowUrl(existingIdx)
      return
    }
    const blankIdx = sharedUrlRows.findIndex((r) => !r.url.trim() && !r.label.trim())
    if (blankIdx >= 0) {
      const next = [...sharedUrlRows]
      next[blankIdx] = { ...next[blankIdx], label: bubbleLabel }
      setSharedUrlRows(next)
      focusRowUrl(blankIdx)
      return
    }
    const next = [...sharedUrlRows, { url: '', label: bubbleLabel }]
    setSharedUrlRows(next)
    focusRowUrl(next.length - 1)
  }

  // Guard route + ensure at least one row exists.
  useEffect(() => {
    if (sharedUrlRows.length === 0) {
      setSharedUrlRows([{ url: '', label: '' }])
      return
    }
  }, [navigate, sharedUrlRows, setSharedUrlRows])

  // Prefill first row label so the first open field matches the first bubble.
  useEffect(() => {
    const firstBubble = suggestedBubbles[0]?.label || ''
    if (!firstBubble) return
    if (sharedUrlRows.length === 0) return
    const first = sharedUrlRows[0]
    if (!first) return
    if (!first.url.trim() && !first.label.trim()) {
      const next = [...sharedUrlRows]
      next[0] = { ...next[0], label: firstBubble }
      setSharedUrlRows(next)
    }
  }, [suggestedBubbles, sharedUrlRows, setSharedUrlRows])

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">{t('createBot.helpfulLinksOptional', 'Helpful links (optional)')}</div>
        <div className="card-subtitle">
          {t(
            'createBot.helpfulLinksOptionalSubtitle',
            'Add a few important links so your assistant can guide customers to the right page.'
          )}
        </div>
      </div>

      {/* Suggested topics to help users add links quickly */}
      <div>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>
          {t('createBot.quickTopics', 'Quick topics')}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          {suggestedBubbles.map((b) => {
            const done = bubbleHasFilledUrl(b.label)
            return (
              <button
                key={b.key}
                type="button"
                onClick={() => handleBubbleClick(b.label)}
                title={b.description}
                style={{
                  appearance: 'none',
                  border: `1px solid ${done ? 'var(--flow-accent, #e4587a)' : 'var(--flow-border)'}`,
                  background: done ? 'var(--flow-accent-soft, #fff1ef)' : 'var(--flow-surface)',
                  color: done ? 'var(--flow-accent, #e4587a)' : 'var(--flow-text)',
                  borderRadius: '999px',
                  padding: '0.45rem 0.85rem',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  lineHeight: 1.1,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                <span>{b.label}</span>
                {done && (
                  <span
                    aria-label={t('createBot.added', 'Added')}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '16px',
                      height: '16px',
                      borderRadius: '999px',
                      background: 'var(--flow-accent, #e4587a)',
                      color: '#fff',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                    }}
                  >
                    ✓
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* URL rows */}
      <div
        className="flow-url-container"
        style={{
          border: '1px solid var(--flow-border)',
          borderRadius: 'var(--flow-radius)',
          background: 'var(--flow-surface)',
          padding: '0.9rem',
          height: '460px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
        }}
      >
        <div className="flow-hint-text" style={{ marginBottom: '4px' }}>
          {t(
            'createBot.addLinksHint',
            'Add links for the pages customers ask about most (menu/services, pricing, hours, booking, contact, FAQ).'
          )}
        </div>
        <div style={{ overflowY: 'auto', minHeight: 0, paddingRight: '0.2rem' }}>
          {sharedUrlRows.map((row, idx) => (
            <div key={`row-${idx}`} className="flow-shared-url-row" style={{
              display: 'flex',
              gap: '10px',
              alignItems: 'flex-start',
              flexWrap: 'wrap',
              marginBottom: '10px',
            }}>
              <div style={{ flex: 2, minWidth: 220 }}>
                <GlassField label={t('createBot.urlLabel', 'URL')} style={{ maxWidth: 'none' }}>
                  <input
                    type="url"
                    ref={(el) => {
                      urlInputRefs.current[idx] = el
                    }}
                    value={row.url}
                    onChange={(e) => {
                      const next = [...sharedUrlRows]
                      next[idx] = { ...next[idx], url: e.target.value }
                      setSharedUrlRows(next)
                    }}
                    placeholder={t('createBot.sharedUrlPlaceholder', 'https://example.com/pricing')}
                    style={{ width: '100%' }}
                  />
                </GlassField>
              </div>
              <div style={{ flex: 1, minWidth: 160 }}>
                <GlassField label={t('createBot.labelLabel', 'Label')} style={{ maxWidth: 'none' }}>
                  <input
                    type="text"
                    value={row.label}
                    onChange={(e) => {
                      const next = [...sharedUrlRows]
                      next[idx] = { ...next[idx], label: e.target.value }
                      setSharedUrlRows(next)
                    }}
                    placeholder={t('createBot.sharedLabelPlaceholder', 'What is this page about? (e.g. Pricing)')}
                    style={{ width: '100%' }}
                  />
                </GlassField>
              </div>
              <UiButton
                variant="ghost"
                onClick={() => {
                  const next = sharedUrlRows.filter((_, i) => i !== idx)
                  setSharedUrlRows(next.length ? next : [{ url: '', label: '' }])
                }}
                disabled={sharedUrlRows.length <= 1}
                aria-label={t('createBot.removeRow', 'Remove row')}
                title={t('common.remove', 'Remove')}
                style={{ color: 'var(--flow-muted, #64748b)', padding: '0.6rem', marginTop: '1px' }}
              >
                <FlowIcon name="delete" size="sm" />
              </UiButton>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '0.25rem' }}>
          <UiButton
            variant="ghost"
            onClick={() => setSharedUrlRows([...sharedUrlRows, { url: '', label: '' }])}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--flow-accent)' }}
          >
            <FlowIcon name="add" size="xs" />
            {t('createBot.addOneMoreLink', 'Add one more link')}
          </UiButton>
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          {t('common.back', 'Back')}
        </UiButton>
        <UiButton variant="primary" onClick={() => flow.nextPath && navigate(flow.nextPath)}>
          {t('common.continue', 'Continue')}
        </UiButton>
      </div>
    </div>
  )
}
