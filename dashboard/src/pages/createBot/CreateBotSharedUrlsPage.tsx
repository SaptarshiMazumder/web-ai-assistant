import { useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlowIcon } from '../../components/FlowIcon'
import { UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

type SuggestedBubble = { key: string; label: string; description: string }

export default function CreateBotSharedUrlsPage() {
  const navigate = useNavigate()
  const { step1, step2, flow } = useCreateBotFlow()
  const { businessType } = step1
  const {
    sharedUrlRows,
    setSharedUrlRows,
    localError,
  } = step2
  const urlInputRefs = useRef<Record<number, HTMLInputElement | null>>({})
  const normalizeLabelKey = (s: string) => (s || '').trim().toLowerCase()

  const suggestedBubbles: SuggestedBubble[] = useMemo(() => {
    const base: SuggestedBubble[] = [
      { key: 'pricing', label: 'Pricing', description: 'Plans, packages, pricing tables' },
      { key: 'location', label: 'Location', description: 'Address, map, directions' },
      { key: 'hours', label: 'Hours', description: 'Opening hours, business hours' },
      { key: 'contact', label: 'Contact', description: 'Phone, email, contact form' },
      { key: 'services', label: 'Services', description: 'Service menu and details' },
      { key: 'faq', label: 'FAQ', description: 'Common questions and answers' },
      { key: 'policies', label: 'Policies', description: 'Refunds, cancellation, terms' },
    ]
    if (businessType === 'hotel') {
      base.unshift(
        { key: 'availability', label: 'Availability', description: 'Availability & rates page' },
        { key: 'booking', label: 'Booking', description: 'Booking / reservation page' }
      )
    }
    return base
  }, [businessType])

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
        <div className="card-title">Helpful links (optional)</div>
        <div className="card-subtitle">
          Add a few important links so your assistant can guide customers to the right page.
        </div>
      </div>

      {/* Suggested topics to help users add links quickly */}
      <div>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>
          Quick topics
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
                    aria-label="Added"
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
          Add links for the pages customers ask about most (menu/services, pricing, hours, booking, contact, FAQ).
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
                <div className="flow-field-input-wrap">
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
                    placeholder="https://example.com/pricing"
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
              <div style={{ flex: 1, minWidth: 160 }}>
                <div className="flow-field-input-wrap">
                  <input
                    type="text"
                    value={row.label}
                    onChange={(e) => {
                      const next = [...sharedUrlRows]
                      next[idx] = { ...next[idx], label: e.target.value }
                      setSharedUrlRows(next)
                    }}
                    placeholder="What is this page about? (e.g. Pricing)"
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
              <UiButton
                variant="ghost"
                onClick={() => {
                  const next = sharedUrlRows.filter((_, i) => i !== idx)
                  setSharedUrlRows(next.length ? next : [{ url: '', label: '' }])
                }}
                disabled={sharedUrlRows.length <= 1}
                aria-label="Remove row"
                title="Remove"
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
            Add one more link
          </UiButton>
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        <UiButton variant="primary" onClick={() => flow.nextPath && navigate(flow.nextPath)}>
          Continue
        </UiButton>
      </div>
    </div>
  )
}
