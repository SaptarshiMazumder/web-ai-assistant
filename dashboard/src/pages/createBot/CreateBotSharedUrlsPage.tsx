import { useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlowIcon } from '../../components/FlowIcon'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon } from './DiscoveryIcons'
import StarBorder from '../../components/StarBorder'

type SuggestedBubble = { key: string; label: string; description: string }

export default function CreateBotSharedUrlsPage() {
  const navigate = useNavigate()
  const { step1, step2, flow } = useCreateBotFlow()
  const { businessType } = step1
  const {
    contentHosting,
    sharedUrlRows,
    setSharedUrlRows,
    sharedUrls,
    pdfFiles,
    isStartingTraining,
    continueWithoutSources,
    startTraining,
    localError,
  } = step2

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

  const urlInputRefs = useRef<Record<number, HTMLInputElement | null>>({})
  const normalizeLabelKey = (s: string) => (s || '').trim().toLowerCase()

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
    if (contentHosting !== 'shared') {
      navigate('/create-bot/sources', { replace: true })
      return
    }
    if (sharedUrlRows.length === 0) {
      setSharedUrlRows([{ url: '', label: '' }])
      return
    }
  }, [contentHosting, navigate, sharedUrlRows, setSharedUrlRows])

  // Prefill first row label so the first open field matches the first bubble.
  useEffect(() => {
    if (contentHosting !== 'shared') return
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
  }, [contentHosting, suggestedBubbles, sharedUrlRows, setSharedUrlRows])

  const hasAnySources = sharedUrls.length > 0 || pdfFiles.length > 0

  const handleStartTraining = async () => {
    const botId = await startTraining()
    if (botId && flow.nextPath) navigate(flow.nextPath)
  }

  const handleSkip = async () => {
    const botId = await continueWithoutSources()
    if (botId && flow.nextPath) navigate(flow.nextPath)
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Add your page links</div>
        <div className="card-subtitle">
          Add links to important business pages. The agent will guide visitors to these when they ask questions.
        </div>
      </div>

      {/* Suggested topic bubbles */}
      <div>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>
          Suggested topics
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {suggestedBubbles.map((b) => {
            const done = bubbleHasFilledUrl(b.label)
            return (
              <button
                key={b.key}
                type="button"
                className="icon-pill icon-pill--with-label"
                onClick={() => handleBubbleClick(b.label)}
                title={b.description}
                style={done ? {
                  background: 'var(--flow-accent-soft, #fff1ef)',
                  color: 'var(--flow-accent, #e4587a)',
                  borderColor: 'var(--flow-accent, #e4587a)',
                } : undefined}
              >
                <span>{b.label}</span>
                {done && <span aria-label="Added" style={{ fontWeight: 700, marginLeft: '2px' }}>&#10003;</span>}
              </button>
            )
          })}
        </div>
      </div>

      {/* URL rows */}
      <div style={{
        display: 'grid',
        gap: '10px',
      }}>
        <div className="flow-hint-text" style={{ marginBottom: '4px' }}>
          Add a link for each important page (services, pricing, hours, booking, contact, FAQs).
        </div>

        {sharedUrlRows.map((row, idx) => (
          <div key={`row-${idx}`} style={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-start',
            flexWrap: 'wrap',
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
                  placeholder="Topic (e.g. Pricing)"
                  style={{ width: '100%' }}
                />
              </div>
            </div>
            <button
              type="button"
              className="ghost"
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
            </button>
          </div>
        ))}

        <div>
          <button
            type="button"
            className="ghost"
            onClick={() => setSharedUrlRows([...sharedUrlRows, { url: '', label: '' }])}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--flow-accent)' }}
          >
            <FlowIcon name="add" size="xs" />
            Add another link
          </button>
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
          <button type="button" className="ghost" onClick={() => void handleSkip()} disabled={isStartingTraining}>
            Skip for now
          </button>
          {hasAnySources && (
            <StarBorder
              as="button"
              type="button"
              className="star-border--primary"
              onClick={handleStartTraining}
              disabled={isStartingTraining}
              color="#e4587a"
              speed="5s"
              aria-disabled={isStartingTraining}
            >
              <PlayIcon />
              {isStartingTraining ? 'Starting...' : 'Start training'}
            </StarBorder>
          )}
        </div>
      </div>
    </div>
  )
}
