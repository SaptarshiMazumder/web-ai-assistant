import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlowIcon } from '../../components/FlowIcon'
import { UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon, StopIcon } from './DiscoveryIcons'
import StarBorder from '../../components/StarBorder'
import { useDashboardData } from '../../hooks/useDashboardData'

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
  const { discoverUrls: discoverUrlsFromHook } = useDashboardData()
  
  // Discovery state
  const [showDiscovery, setShowDiscovery] = useState(false)
  const [discoveryUrl, setDiscoveryUrl] = useState('')
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedDiscoveredUrls, setSelectedDiscoveredUrls] = useState<Set<string>>(new Set())
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [discoveryError, setDiscoveryError] = useState<string | null>(null)
  const discoveryAbortRef = useRef<AbortController | null>(null)

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

  const handleDiscoverUrls = useCallback(async () => {
    setDiscoveryError(null)
    const trimmedUrl = discoveryUrl.trim()
    
    if (!trimmedUrl) {
      setDiscoveryError('Enter a URL to discover pages')
      return
    }

    // Normalize URL
    let normalizedUrl = ''
    try {
      const withProtocol = /^https?:\/\//i.test(trimmedUrl) ? trimmedUrl : `https://${trimmedUrl}`
      const parsed = new URL(withProtocol)
      normalizedUrl = parsed.href
    } catch {
      setDiscoveryError('Enter a valid URL')
      return
    }

    setIsDiscovering(true)
    setDiscoveredUrls([])
    setSelectedDiscoveredUrls(new Set())

    const controller = new AbortController()
    discoveryAbortRef.current = controller

    try {
      await discoverUrlsFromHook(normalizedUrl, 'auto', (evt) => {
        if (evt.type === 'discovered' && typeof evt.url === 'string') {
          const url = evt.url
          setDiscoveredUrls((prev) => {
            if (prev.includes(url)) return prev
            return [...prev, url]
          })
          // Auto-select discovered URLs
          setSelectedDiscoveredUrls((prev) => new Set([...prev, url]))
        }

        if (evt.type === 'error' && typeof evt.message === 'string') {
          setDiscoveryError(evt.message)
        }

        if (evt.type === 'done') {
          setIsDiscovering(false)
        }
      }, controller.signal, { max_duration_sec: 60 })
    } catch (err) {
      const e = err as Error & { name?: string }
      if (e.name !== 'AbortError') {
        setDiscoveryError(e.message || 'Discovery failed')
      }
    } finally {
      setIsDiscovering(false)
      discoveryAbortRef.current = null
    }
  }, [discoveryUrl, discoverUrlsFromHook])

  const handleStopDiscovery = useCallback(() => {
    discoveryAbortRef.current?.abort()
    setIsDiscovering(false)
  }, [])

  const handleAddDiscoveredUrls = useCallback(() => {
    const urlsToAdd = Array.from(selectedDiscoveredUrls)
    const newRows: typeof sharedUrlRows = []
    
    // Keep existing rows that have URLs
    newRows.push(...sharedUrlRows.filter(row => row.url.trim()))
    
    // Add discovered URLs that aren't already in the list
    const existingUrls = new Set(sharedUrlRows.map(r => r.url.trim()))
    
    for (const url of urlsToAdd) {
      if (!existingUrls.has(url)) {
        // Try to extract a label from the URL path
        let label = 'Link'
        try {
          const parsed = new URL(url)
          const pathParts = parsed.pathname.split('/').filter(Boolean)
          if (pathParts.length > 0) {
            const lastPart = pathParts[pathParts.length - 1]
            // Remove file extensions if present
            const withoutExt = lastPart.replace(/\.(html?|php|aspx?)$/i, '')
            // Clean up the label (remove dashes, underscores, make it more readable)
            if (withoutExt.length > 0) {
              label = withoutExt
                .replace(/[-_]/g, ' ')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ')
            }
          } else {
            // If no path parts, try to use a sensible default
            label = 'Home'
          }
        } catch {
          label = 'Link'
        }
        
        newRows.push({ url, label })
        existingUrls.add(url)
      }
    }
    
    // Ensure at least one empty row at the end
    if (newRows.length === 0 || newRows[newRows.length - 1].url.trim()) {
      newRows.push({ url: '', label: '' })
    }
    
    setSharedUrlRows(newRows)
    setShowDiscovery(false)
    setDiscoveredUrls([])
    setSelectedDiscoveredUrls(new Set())
    setDiscoveryUrl('')
  }, [selectedDiscoveredUrls, sharedUrlRows, setSharedUrlRows])

  const toggleDiscoveredUrl = useCallback((url: string) => {
    setSelectedDiscoveredUrls(prev => {
      const next = new Set(prev)
      if (next.has(url)) {
        next.delete(url)
      } else {
        next.add(url)
      }
      return next
    })
  }, [])

  const selectAllDiscovered = useCallback(() => {
    setSelectedDiscoveredUrls(new Set(discoveredUrls))
  }, [discoveredUrls])

  const deselectAllDiscovered = useCallback(() => {
    setSelectedDiscoveredUrls(new Set())
  }, [])

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Add your page links</div>
        <div className="card-subtitle">
          Add links to important business pages. The agent will guide visitors to these when they ask questions.
        </div>
      </div>

      {/* URL Discovery Section */}
      <div style={{ marginBottom: '1rem' }}>
        {!showDiscovery ? (
          <UiButton
            variant="secondary"
            onClick={() => setShowDiscovery(true)}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
          >
            <FlowIcon name="search" size="sm" />
            Discover pages from URL
          </UiButton>
        ) : (
          <div style={{
            border: '1px solid var(--flow-border)',
            borderRadius: 'var(--flow-radius)',
            padding: '1rem',
            background: 'var(--flow-surface)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Discover pages</div>
              <UiButton
                variant="ghost"
                onClick={() => {
                  setShowDiscovery(false)
                  setDiscoveredUrls([])
                  setSelectedDiscoveredUrls(new Set())
                  setDiscoveryUrl('')
                  handleStopDiscovery()
                }}
                style={{ padding: '0.4rem' }}
              >
                <FlowIcon name="close" size="xs" />
              </UiButton>
            </div>

            <div className="flow-hint-text" style={{ marginBottom: '0.75rem' }}>
              Enter a URL to discover all pages within that path. For example, <code>https://example.com/hotel/tokyo/</code> will find all pages under that hotel section.
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '0.75rem' }}>
              <div style={{ flex: 1 }}>
                <input
                  type="url"
                  value={discoveryUrl}
                  onChange={(e) => setDiscoveryUrl(e.target.value)}
                  placeholder="https://example.com/your-section/"
                  disabled={isDiscovering}
                  style={{ width: '100%' }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !isDiscovering) {
                      void handleDiscoverUrls()
                    }
                  }}
                />
              </div>
              {!isDiscovering ? (
                <UiButton
                  variant="primary"
                  onClick={() => void handleDiscoverUrls()}
                  disabled={!discoveryUrl.trim()}
                >
                  Discover
                </UiButton>
              ) : (
                <UiButton
                  variant="secondary"
                  onClick={handleStopDiscovery}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                >
                  <StopIcon />
                  Stop
                </UiButton>
              )}
            </div>

            {discoveryError && (
              <div className="alert error" style={{ marginBottom: '0.75rem' }}>
                {discoveryError}
              </div>
            )}

            {isDiscovering && (
              <div className="flow-hint-text" style={{ marginBottom: '0.75rem', color: 'var(--flow-accent)' }}>
                <span className="discovery-loading-dots" aria-hidden style={{ display: 'inline-flex', gap: '4px', marginRight: '8px' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'currentColor', animation: 'discovery-dot 1.4s infinite ease-in-out' }} />
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'currentColor', animation: 'discovery-dot 1.4s infinite ease-in-out 0.2s' }} />
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'currentColor', animation: 'discovery-dot 1.4s infinite ease-in-out 0.4s' }} />
                </span>
                Discovering pages... {discoveredUrls.length} found so far
              </div>
            )}

            {discoveredUrls.length > 0 && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                    Found {discoveredUrls.length} page{discoveredUrls.length !== 1 ? 's' : ''}
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <UiButton
                      variant="ghost"
                      onClick={selectedDiscoveredUrls.size === discoveredUrls.length ? deselectAllDiscovered : selectAllDiscovered}
                      style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}
                    >
                      {selectedDiscoveredUrls.size === discoveredUrls.length ? 'Deselect all' : 'Select all'}
                    </UiButton>
                  </div>
                </div>

                <div style={{
                  maxHeight: '300px',
                  overflowY: 'auto',
                  border: '1px solid var(--flow-border)',
                  borderRadius: 'var(--flow-radius)',
                  padding: '0.5rem',
                  background: 'var(--flow-bg)',
                  marginBottom: '0.75rem',
                }}>
                  {discoveredUrls.map((url) => (
                    <label
                      key={url}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        padding: '0.4rem 0.5rem',
                        cursor: 'pointer',
                        borderRadius: '4px',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--flow-surface)')}
                      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                    >
                      <input
                        type="checkbox"
                        checked={selectedDiscoveredUrls.has(url)}
                        onChange={() => toggleDiscoveredUrl(url)}
                        style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--flow-accent)' }}
                      />
                      <span style={{ fontSize: '0.85rem', color: 'var(--flow-text)', wordBreak: 'break-all' }}>
                        {url}
                      </span>
                    </label>
                  ))}
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                  <UiButton
                    variant="secondary"
                    onClick={() => {
                      setShowDiscovery(false)
                      setDiscoveredUrls([])
                      setSelectedDiscoveredUrls(new Set())
                      setDiscoveryUrl('')
                    }}
                  >
                    Cancel
                  </UiButton>
                  <UiButton
                    variant="primary"
                    onClick={handleAddDiscoveredUrls}
                    disabled={selectedDiscoveredUrls.size === 0}
                  >
                    Add {selectedDiscoveredUrls.size} page{selectedDiscoveredUrls.size !== 1 ? 's' : ''}
                  </UiButton>
                </div>
              </>
            )}
          </div>
        )}
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

        <div>
          <UiButton
            variant="ghost"
            onClick={() => setSharedUrlRows([...sharedUrlRows, { url: '', label: '' }])}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--flow-accent)' }}
          >
            <FlowIcon name="add" size="xs" />
            Add another link
          </UiButton>
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
          <UiButton variant="ghost" onClick={() => void handleSkip()} disabled={isStartingTraining}>
            Skip for now
          </UiButton>
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
