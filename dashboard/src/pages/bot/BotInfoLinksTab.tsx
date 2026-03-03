import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Check, Trash2 } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, GlassField } from '../../components/ui'

type UrlBankRow = {
  label: string
  url: string
}

type SuggestedBubble = {
  key: string
  label: string
  description: string
}

function normalizeUrlBankUrl(entry: string): string {
  const raw = (entry || '').trim()
  if (!raw) return ''
  try {
    const parsed = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return ''
    return parsed.toString()
  } catch {
    return ''
  }
}

export default function BotInfoLinksTab() {
  const { selectedBot, selectedBotWidgetConfig, saveWidgetConfig, syncUrlBankTopics, fetchPlatformConfig } = useDashboardData()
  const [urlBankRows, setUrlBankRows] = useState<UrlBankRow[]>([{ label: '', url: '' }])
  const [platforms, setPlatforms] = useState<Array<{ id: string; widget_key: string; domain_key: string; label: string; url_placeholder?: string }>>([])

  useEffect(() => {
    fetchPlatformConfig('en').then((r) => setPlatforms(r.platforms))
  }, [fetchPlatformConfig])
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const urlInputRefs = useRef<Record<number, HTMLInputElement | null>>({})
  const normalizeLabelKey = (s: string) => (s || '').trim().toLowerCase()

  const businessType =
    selectedBotWidgetConfig &&
    typeof selectedBotWidgetConfig === 'object' &&
    typeof (selectedBotWidgetConfig as Record<string, unknown>).businessType === 'string'
      ? (selectedBotWidgetConfig as Record<string, unknown>).businessType
      : ''

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
        { key: 'availability', label: 'Availability', description: 'Availability and rates page' },
        { key: 'booking', label: 'Booking', description: 'Booking or reservation page' }
      )
    }
    if (businessType === 'restaurant') {
      for (const p of platforms) {
        base.unshift({ key: p.id, label: p.label, description: p.url_placeholder || `${p.label} page` })
      }
      base.unshift(
        { key: 'menu', label: 'Menu', description: 'Food and drink menu' },
        { key: 'reservation', label: 'Reservation', description: 'Reservation or booking page' }
      )
    }
    return base
  }, [businessType, platforms])

  const bubbleHasFilledUrl = (bubbleLabel: string) => {
    const key = normalizeLabelKey(bubbleLabel)
    return urlBankRows.some((row) => normalizeLabelKey(row.label) === key && !!row.url.trim())
  }

  const focusRowUrl = (idx: number) => {
    window.setTimeout(() => {
      const input = urlInputRefs.current[idx]
      if (!input) return
      input.focus()
      input.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
    }, 0)
  }

  const handleBubbleClick = (bubbleLabel: string) => {
    const key = normalizeLabelKey(bubbleLabel)
    const existingIdx = urlBankRows.findIndex((row) => normalizeLabelKey(row.label) === key && !row.url.trim())
    if (existingIdx >= 0) {
      focusRowUrl(existingIdx)
      return
    }

    const blankIdx = urlBankRows.findIndex((row) => !row.url.trim() && !row.label.trim())
    if (blankIdx >= 0) {
      const next = [...urlBankRows]
      next[blankIdx] = { ...next[blankIdx], label: bubbleLabel }
      setUrlBankRows(next)
      focusRowUrl(blankIdx)
      return
    }

    const next = [...urlBankRows, { label: bubbleLabel, url: '' }]
    setUrlBankRows(next)
    focusRowUrl(next.length - 1)
  }

  useEffect(() => {
    const cfg = selectedBotWidgetConfig
    const raw = cfg && typeof cfg === 'object' ? (cfg as Record<string, unknown>).urlBank : null
    if (!Array.isArray(raw)) {
      setUrlBankRows([{ label: '', url: '' }])
      return
    }
    const cleaned: UrlBankRow[] = []
    for (const item of raw) {
      if (!item || typeof item !== 'object') continue
      const entry = item as Record<string, unknown>
      const label = typeof entry.label === 'string' ? entry.label.trim() : ''
      const url = typeof entry.url === 'string' ? entry.url.trim() : ''
      if (label || url) cleaned.push({ label, url })
    }
    setUrlBankRows(cleaned.length ? cleaned : [{ label: '', url: '' }])
  }, [selectedBotWidgetConfig, selectedBot?.bot_id])

  // Keep at least one editable row available.
  useEffect(() => {
    if (urlBankRows.length === 0) {
      setUrlBankRows([{ label: '', url: '' }])
    }
  }, [urlBankRows.length])

  // Prefill the first row label if the first row is still blank.
  useEffect(() => {
    const firstBubble = suggestedBubbles[0]?.label || ''
    if (!firstBubble) return
    if (urlBankRows.length === 0) return
    const first = urlBankRows[0]
    if (!first) return
    if (!first.url.trim() && !first.label.trim()) {
      const next = [...urlBankRows]
      next[0] = { ...next[0], label: firstBubble }
      setUrlBankRows(next)
    }
  }, [suggestedBubbles, urlBankRows])

  const handleSave = useCallback(async () => {
    if (!selectedBot || saving) return
    setSaving(true)
    setSaved(false)
    try {
      const base =
        selectedBotWidgetConfig && typeof selectedBotWidgetConfig === 'object'
          ? (selectedBotWidgetConfig as Record<string, unknown>)
          : {}
      const byUrl = new Map<string, UrlBankRow>()
      for (const row of urlBankRows) {
        const url = normalizeUrlBankUrl(row.url)
        if (!url) continue
        const rawLabel = (row.label || '').trim()
        const label = rawLabel || 'Link'
        const previous = byUrl.get(url)
        if (!previous || (previous.label === 'Link' && rawLabel)) {
          byUrl.set(url, { url, label })
        }
      }
      const urlBankEntries = Array.from(byUrl.values())
      await saveWidgetConfig(selectedBot.bot_id, { ...base, urlBank: urlBankEntries })
      // Mirror these links into topics for retrieval ranking; failures are non-blocking.
      syncUrlBankTopics(selectedBot.bot_id, urlBankEntries).catch(() => { /* non-fatal */ })
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2200)
    } finally {
      setSaving(false)
    }
  }, [saveWidgetConfig, saving, selectedBot, selectedBotWidgetConfig, syncUrlBankTopics, urlBankRows])

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to manage info links.</div>
  }

  return (
    <AnimatedPage className="card-grid knowledge-redesign">
      <GlassCard style={{ gridColumn: '1 / -1' }}>
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div>
            <div className="card-title">Additional links</div>
            <p className="card-subtitle" style={{ marginTop: '0.25rem' }}>
              Add links your AI agent can share for common customer questions.
            </p>
          </div>
          <button type="button" className="primary" onClick={() => void handleSave()} disabled={saving}>
            {saving ? 'Saving...' : 'Save links'}
          </button>
        </div>

        <div style={{ marginTop: '1rem' }}>
          <div className="info-links-topics-title">
            Quick topics
          </div>
          <div className="info-links-bubbles">
            {suggestedBubbles.map((bubble) => {
              const done = bubbleHasFilledUrl(bubble.label)
              return (
                <button
                  key={bubble.key}
                  type="button"
                  className={`info-links-bubble${done ? ' is-done' : ''}`}
                  onClick={() => handleBubbleClick(bubble.label)}
                  title={bubble.description}
                >
                  <span>{bubble.label}</span>
                  {done && (
                    <span aria-label="Added" className="info-links-bubble-check">
                      <Check size={11} strokeWidth={3} aria-hidden />
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        <div style={{ display: 'grid', gap: '1.25rem', marginTop: '1.5rem' }}>
          {urlBankRows.map((row, idx) => (
            <div key={`urlbank-${idx}`} className="info-links-row" style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <GlassField label="Topic" style={{ flex: 1, minWidth: '160px' }}>
                <input
                  type="text"
                  value={row.label}
                  onChange={(e) => {
                    const next = [...urlBankRows]
                    next[idx] = { ...next[idx], label: e.target.value }
                    setUrlBankRows(next)
                  }}
                  placeholder="e.g. Pricing"
                />
              </GlassField>
              <GlassField label="Link" style={{ flex: 2, minWidth: '240px' }}>
                <input
                  type="url"
                  ref={(el) => {
                    urlInputRefs.current[idx] = el
                  }}
                  value={row.url}
                  onChange={(e) => {
                    const next = [...urlBankRows]
                    next[idx] = { ...next[idx], url: e.target.value }
                    setUrlBankRows(next)
                  }}
                  placeholder="e.g. https://example.com/pricing"
                />
              </GlassField>
              <button
                type="button"
                className="delete-btn"
                onClick={() => {
                  const next = urlBankRows.filter((_, i) => i !== idx)
                  setUrlBankRows(next.length ? next : [{ label: '', url: '' }])
                }}
                disabled={urlBankRows.length <= 1}
                aria-label="Remove link"
                title="Remove"
              >
                <Trash2 size={18} aria-hidden />
              </button>
            </div>
          ))}
        </div>

        <div className="row" style={{ gap: '0.75rem', flexWrap: 'wrap', marginTop: '10px' }}>
          <button type="button" className="secondary" onClick={() => setUrlBankRows([...urlBankRows, { label: '', url: '' }])}>
            + Add link
          </button>
          {saved && <span className="muted">Saved.</span>}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
