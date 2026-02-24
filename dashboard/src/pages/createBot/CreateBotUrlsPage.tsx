import React, { useEffect, useState, useMemo, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { ScanSearch, CheckCircle2 } from 'lucide-react'
import { UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'
import { StopIcon } from './DiscoveryIcons'
import { categorizeUrls, getAllUrlsFromCategory, getCategoryUrlCount, getCategoryDisplayPath, getAllExpandablePaths, type UrlCategory } from './urlCategorizer'
import { FileDropzone } from '../../components/FileDropzone'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function CreateBotUrlsPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { discoverUrls: discoverUrlsFromHook } = useDashboardData()
  const { step2, flow } = useCreateBotFlow()
  const {
    discoveredUrls,
    selectedUrls,
    normalizedWebsiteUrl,
    contentHosting,
    websiteUrl,
    setTrainingUrls,
    pdfFiles,
    setPdfFiles,
    isDiscovering,
    discoveryMethod,
    discoveryDurationMs,
    discoveryTimedOutMessage,
    toggleUrl,
    toggleCategory,
    selectAll,
    deselectAll,
    stopDiscovery,
    localError,
    localErrorType,
  } = step2

  const discoveryDurationLabel =
    discoveryDurationMs != null && !isDiscovering
      ? (() => {
        const sec = Math.round(discoveryDurationMs / 1000)
        if (sec < 60) return `${sec}s`
        const m = Math.floor(sec / 60)
        const s = sec % 60
        return s ? `${m}m ${s}s` : `${m}m`
      })()
      : null

  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())
  const [sharedDiscoveryUrl, setSharedDiscoveryUrl] = useState('')
  const [sharedNormalizedDiscoveryUrl, setSharedNormalizedDiscoveryUrl] = useState('')
  const [sharedDiscoveredUrls, setSharedDiscoveredUrls] = useState<string[]>([])
  const [sharedSelectedDiscoveredUrls, setSharedSelectedDiscoveredUrls] = useState<Set<string>>(new Set())
  const [isSharedDiscovering, setIsSharedDiscovering] = useState(false)
  const [sharedDiscoveryError, setSharedDiscoveryError] = useState<string | null>(null)
  const [sharedDiscoveryErrorType, setSharedDiscoveryErrorType] = useState<'error' | 'warning' | null>(null)
  const [sharedDiscoveryDurationMs, setSharedDiscoveryDurationMs] = useState<number | null>(null)
  const [sharedDiscoveryTimedOutMessage, setSharedDiscoveryTimedOutMessage] = useState<string | null>(null)
  const [showPdfFallback, setShowPdfFallback] = useState(false)
  const sharedDiscoveryAbortRef = useRef<AbortController | null>(null)
  const sharedDiscoveryStartTimeRef = useRef<number | null>(null)
  const sharedDiscovery60sTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const sharedDiscoveryTimedOutByTimerRef = useRef(false)
  const sharedSelectionTouchedRef = useRef(false)
  const [expandedSharedCategories, setExpandedSharedCategories] = useState<Set<string>>(new Set())
  const sharedDiscoveryDurationLabel =
    sharedDiscoveryDurationMs != null && !isSharedDiscovering
      ? (() => {
        const sec = Math.round(sharedDiscoveryDurationMs / 1000)
        if (sec < 60) return `${sec}s`
        const m = Math.floor(sec / 60)
        const s = sec % 60
        return s ? `${m}m ${s}s` : `${m}m`
      })()
      : null

  const urlCategories = useMemo(() => {
    if (!discoveredUrls.length || !normalizedWebsiteUrl) return null
    return categorizeUrls(discoveredUrls, normalizedWebsiteUrl)
  }, [discoveredUrls, normalizedWebsiteUrl])
  const sharedUrlCategories = useMemo(() => {
    if (!sharedDiscoveredUrls.length || !sharedNormalizedDiscoveryUrl) return null
    return categorizeUrls(sharedDiscoveredUrls, sharedNormalizedDiscoveryUrl)
  }, [sharedDiscoveredUrls, sharedNormalizedDiscoveryUrl])

  useEffect(() => {
    if (!sharedDiscoveryUrl.trim() && websiteUrl.trim()) {
      setSharedDiscoveryUrl(websiteUrl.trim())
    }
  }, [websiteUrl, sharedDiscoveryUrl])

  const hasExpandedDefault = useRef(false)
  useEffect(() => {
    if (urlCategories && !hasExpandedDefault.current) {
      setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
      hasExpandedDefault.current = true
    }
  }, [urlCategories])
  const hasExpandedSharedDefault = useRef(false)
  useEffect(() => {
    if (sharedUrlCategories && !hasExpandedSharedDefault.current) {
      setExpandedSharedCategories(new Set(getAllExpandablePaths(sharedUrlCategories)))
      hasExpandedSharedDefault.current = true
    }
  }, [sharedUrlCategories])
  useEffect(() => {
    if (!sharedDiscoveredUrls.length) {
      hasExpandedSharedDefault.current = false
      setExpandedSharedCategories(new Set())
    }
  }, [sharedDiscoveredUrls.length])

  const expandAll = () => {
    if (urlCategories) setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
  }
  const collapseAll = () => setExpandedCategories(new Set())

  const persistSharedSelectionToRows = useCallback(() => {
    if (contentHosting === 'own') {
      setTrainingUrls([])
      return
    }
    const selected = Array.from(sharedSelectedDiscoveredUrls)
    setTrainingUrls(selected)
  }, [contentHosting, setTrainingUrls, sharedSelectedDiscoveredUrls])

  const handleContinue = () => {
    persistSharedSelectionToRows()
    if (flow.nextPath) navigate(flow.nextPath)
  }

  const handleSkip = () => {
    persistSharedSelectionToRows()
    if (flow.nextPath) navigate(flow.nextPath)
  }

  const handleSharedDiscoverUrls = useCallback(async () => {
    setSharedDiscoveryError(null)
    setSharedDiscoveryErrorType(null)
    setShowPdfFallback(false)
    const trimmedUrl = sharedDiscoveryUrl.trim()

    if (!trimmedUrl) {
      setSharedDiscoveryError('Enter a URL to discover pages')
      setSharedDiscoveryErrorType('error')
      return
    }

    let normalizedUrl = ''
    try {
      const withProtocol = /^https?:\/\//i.test(trimmedUrl) ? trimmedUrl : `https://${trimmedUrl}`
      const parsed = new URL(withProtocol)
      normalizedUrl = parsed.href
    } catch {
      setSharedDiscoveryError('Enter a valid URL')
      setSharedDiscoveryErrorType('error')
      return
    }

    setIsSharedDiscovering(true)
    setSharedDiscoveredUrls([])
    setSharedSelectedDiscoveredUrls(new Set())
    setSharedDiscoveryDurationMs(null)
    setSharedDiscoveryTimedOutMessage(null)
    setSharedNormalizedDiscoveryUrl(normalizedUrl)
    sharedSelectionTouchedRef.current = false
    sharedDiscoveryStartTimeRef.current = Date.now()

    const controller = new AbortController()
    sharedDiscoveryAbortRef.current = controller
    sharedDiscoveryTimedOutByTimerRef.current = false
    sharedDiscovery60sTimerRef.current = setTimeout(() => {
      sharedDiscovery60sTimerRef.current = null
      sharedDiscoveryTimedOutByTimerRef.current = true
      controller.abort()
    }, 90_000)

    // Track discovered count locally to avoid stale state in finally block
    let localDiscoveredCount = 0
    let hasShownError = false

    try {
      const final = await discoverUrlsFromHook(
        normalizedUrl,
        discoveryMethod,
        (evt) => {
          if (evt.type === 'discovered' && typeof evt.url === 'string') {
            const url = evt.url
            localDiscoveredCount++
            setSharedDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
            if (!sharedSelectionTouchedRef.current) {
              setSharedSelectedDiscoveredUrls((prev) => new Set([...prev, url]))
            }
            // Clear stale warning once we actually get URLs.
            setSharedDiscoveryError(null)
            setSharedDiscoveryErrorType(null)
            setShowPdfFallback(false)
          }
          if (evt.type === 'error' && typeof evt.message === 'string') {
            hasShownError = true
            const reason = evt.failure_reason as string | undefined
            if (reason === 'robots_blocked') {
              setSharedDiscoveryError('This website blocks automatic scanning.')
              setSharedDiscoveryErrorType('error')
              setShowPdfFallback(true)
            } else if (reason === 'sitemap_empty') {
              setSharedDiscoveryError("No sitemap found. Switch to 'Automatic' discovery (recommended).")
              setSharedDiscoveryErrorType('warning')
            } else {
              setSharedDiscoveryError(evt.message)
              setSharedDiscoveryErrorType('error')
            }
          }
          if (evt.type === 'warning' && typeof evt.message === 'string') {
            hasShownError = true
            setSharedDiscoveryError(evt.message)
            setSharedDiscoveryErrorType('warning')
          }
          if (evt.type === 'done') {
            if (sharedDiscovery60sTimerRef.current) {
              clearTimeout(sharedDiscovery60sTimerRef.current)
              sharedDiscovery60sTimerRef.current = null
            }
            const start = sharedDiscoveryStartTimeRef.current
            if (start != null) setSharedDiscoveryDurationMs(Date.now() - start)
            setIsSharedDiscovering(false)
            if ((evt as { timed_out?: boolean }).timed_out === true) {
              setSharedDiscoveryTimedOutMessage('Found main URLs. You can train on these now.')
            }
            const urls = (evt as { urls?: unknown[] }).urls || []
            const reason = (evt as { failure_reason?: string }).failure_reason
            if (reason === 'no_results') {
              hasShownError = true
              setSharedDiscoveryError('Could not discover pages. It\'s likely that the site is blocking our crawling agent.')
              setSharedDiscoveryErrorType('warning')
              setShowPdfFallback(true)
            } else if (Array.isArray(urls) && urls.length === 0) {
              hasShownError = true
              if (reason === 'robots_blocked') {
                setSharedDiscoveryError('This website blocks automatic scanning.')
                setSharedDiscoveryErrorType('error')
                setShowPdfFallback(true)
              } else if (reason === 'sitemap_empty') {
                setSharedDiscoveryError("No sitemap found. Switch to 'Automatic' discovery.")
                setSharedDiscoveryErrorType('warning')
              } else if (reason === 'no_results') {
                setSharedDiscoveryError('We couldn\'t find any pages on this website.')
                setSharedDiscoveryErrorType('warning')
                setShowPdfFallback(true)
              } else {
                setSharedDiscoveryError(
                  discoveryMethod === 'sitemap'
                    ? "Could not discover via sitemap. Switch to 'Automatic' (recommended)."
                    : 'No pages found for this site.'
                )
                setSharedDiscoveryErrorType('warning')
                setShowPdfFallback(true)
              }
            }
          }
        },
        controller.signal,
        { max_duration_sec: 90 }
      )
      // Final check: if we got ≤1 URL, treat as discovery failure.
      const urlCount = final?.urls?.length ?? 0
      localDiscoveredCount = Math.max(localDiscoveredCount, urlCount)
      if (localDiscoveredCount <= 1 || final?.failureReason === 'no_results') {
        hasShownError = true
        setSharedDiscoveryError('Could not discover pages. It\'s likely that the site is blocking our crawling agent.')
        setSharedDiscoveryErrorType('warning')
        setShowPdfFallback(true)
      } else if (final && !final.urls?.length && final.error) {
        hasShownError = true
        setSharedDiscoveryError(final.error)
        setSharedDiscoveryErrorType('error')
        setShowPdfFallback(true)
      }
      const start = sharedDiscoveryStartTimeRef.current
      if (start != null) setSharedDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
    } catch (err) {
      const e = err as Error & { name?: string }
      if (e.name === 'AbortError') {
        const start = sharedDiscoveryStartTimeRef.current
        if (start != null) setSharedDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
        if (sharedDiscoveryTimedOutByTimerRef.current) {
          setSharedDiscoveryTimedOutMessage('Found main URLs. You can train on these now.')
        }
      } else {
        hasShownError = true
        setSharedDiscoveryError(e.message || 'Discovery failed')
        setSharedDiscoveryErrorType('error')
        setShowPdfFallback(true)
      }
    } finally {
      if (sharedDiscovery60sTimerRef.current) {
        clearTimeout(sharedDiscovery60sTimerRef.current)
        sharedDiscovery60sTimerRef.current = null
      }
      setIsSharedDiscovering(false)
      sharedDiscoveryAbortRef.current = null

      // CRITICAL SAFETY: If ≤1 URL discovered and no error shown, FORCE show PDF fallback.
      // Use local count to avoid stale React state in closure.
      if (localDiscoveredCount <= 1 && !hasShownError) {
        setSharedDiscoveryError('Discovery completed but found no usable pages. Please use the PDF upload method below.')
        setSharedDiscoveryErrorType('warning')
        setShowPdfFallback(true)
      }
    }
  }, [sharedDiscoveryUrl, discoverUrlsFromHook, discoveryMethod])

  const handleStopSharedDiscovery = useCallback(() => {
    if (sharedDiscovery60sTimerRef.current) {
      clearTimeout(sharedDiscovery60sTimerRef.current)
      sharedDiscovery60sTimerRef.current = null
    }
    sharedDiscoveryAbortRef.current?.abort()
    setIsSharedDiscovering(false)
  }, [])

  const handleToggleSharedDiscoveredUrl = useCallback((url: string) => {
    sharedSelectionTouchedRef.current = true
    setSharedSelectedDiscoveredUrls((prev) => {
      const next = new Set(prev)
      if (next.has(url)) next.delete(url)
      else next.add(url)
      return next
    })
  }, [])

  const handleToggleSharedCategory = useCallback((categoryUrls: string[]) => {
    sharedSelectionTouchedRef.current = true
    const allSelected = categoryUrls.every((url) => sharedSelectedDiscoveredUrls.has(url))
    setSharedSelectedDiscoveredUrls((prev) => {
      const next = new Set(prev)
      if (allSelected) {
        categoryUrls.forEach((url) => next.delete(url))
      } else {
        categoryUrls.forEach((url) => next.add(url))
      }
      return next
    })
  }, [sharedSelectedDiscoveredUrls])

  const handleSelectAllSharedDiscovered = useCallback(() => {
    sharedSelectionTouchedRef.current = true
    setSharedSelectedDiscoveredUrls(new Set(sharedDiscoveredUrls))
  }, [sharedDiscoveredUrls])

  const handleDeselectAllSharedDiscovered = useCallback(() => {
    sharedSelectionTouchedRef.current = true
    setSharedSelectedDiscoveredUrls(new Set())
  }, [])

  const toggleSharedCategoryExpand = (path: string) => {
    setExpandedSharedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const expandAllShared = () => {
    if (sharedUrlCategories) setExpandedSharedCategories(new Set(getAllExpandablePaths(sharedUrlCategories)))
  }

  const collapseAllShared = () => setExpandedSharedCategories(new Set())

  const isSharedCategorySelected = (category: UrlCategory): boolean => {
    const categoryUrls = getAllUrlsFromCategory(category)
    return categoryUrls.length > 0 && categoryUrls.every((url) => sharedSelectedDiscoveredUrls.has(url))
  }

  const isSharedCategoryPartiallySelected = (category: UrlCategory): boolean => {
    const categoryUrls = getAllUrlsFromCategory(category)
    const selectedCount = categoryUrls.filter((url) => sharedSelectedDiscoveredUrls.has(url)).length
    return selectedCount > 0 && selectedCount < categoryUrls.length
  }

  const renderSharedCategory = (category: UrlCategory): React.ReactNode => {
    const categoryUrls = getAllUrlsFromCategory(category)
    const urlCount = getCategoryUrlCount(category)
    const isExpanded = expandedSharedCategories.has(category.path)
    const isSelected = isSharedCategorySelected(category)
    const isPartial = isSharedCategoryPartiallySelected(category)
    const hasChildCategories = category.children.size > 0
    const hasExpandableContent = hasChildCategories || category.urls.length > 0

    return (
      <div key={category.path || 'root'} style={{ marginLeft: `${category.level * 20}px` }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '6px 0',
            cursor: 'pointer',
            userSelect: 'none',
          }}
        >
          {hasExpandableContent ? (
            <span
              onClick={(e) => {
                e.stopPropagation()
                toggleSharedCategoryExpand(category.path)
              }}
              style={{
                marginRight: '8px',
                width: '16px',
                height: '16px',
                transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: 'var(--flow-muted)',
              }}
              aria-hidden
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </span>
          ) : (
            <span style={{ marginRight: '16px', width: '12px' }} />
          )}
          <input
            type="checkbox"
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation()
              handleToggleSharedCategory(categoryUrls)
            }}
            ref={(input) => {
              if (input) {
                input.indeterminate = isPartial
              }
            }}
            style={{ marginRight: '8px', cursor: 'pointer' }}
          />
          <span
            onClick={() => hasExpandableContent && toggleSharedCategoryExpand(category.path)}
            style={{
              flex: 1,
              cursor: hasExpandableContent ? 'pointer' : 'default',
              fontWeight: 500,
              color: 'var(--flow-text)',
              fontSize: '0.9rem',
            }}
          >
            {getCategoryDisplayPath(category)}
          </span>
          <span
            style={{
              marginLeft: '8px',
              padding: '2px 10px',
              borderRadius: '999px',
              background: 'var(--flow-accent-soft)',
              color: 'var(--flow-accent)',
              fontSize: '0.78rem',
              fontWeight: 600,
            }}
          >
            {urlCount}
          </span>
        </div>
        {hasExpandableContent && isExpanded && (
          <div>
            {Array.from(category.children.values())
              .sort((a, b) => {
                const countA = getCategoryUrlCount(a)
                const countB = getCategoryUrlCount(b)
                if (countA !== countB) return countB - countA
                return a.name.localeCompare(b.name)
              })
              .map((child) => renderSharedCategory(child))}
            {category.urls.length > 0 && (
              <div style={{ marginLeft: '20px', paddingLeft: '20px' }}>
                {category.urls.map((url) => (
                  <label
                    key={url}
                    className="url-list-item"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={sharedSelectedDiscoveredUrls.has(url)}
                      onChange={() => handleToggleSharedDiscoveredUrl(url)}
                      style={{ marginRight: '8px', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.85rem', color: 'var(--flow-muted)' }}>{url}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    )
  }


  /* ── Shared-hosting sub-view ───────────────────────────────────────────── */
  if (contentHosting !== 'own') {
    return (
      <div className="flow-panel-body">
        <div>
          <div className="card-title">{t('createBot.addWebsitePages', 'Add website pages and files')}</div>
          <div className="card-subtitle">
            {t('createBot.addWebsitePagesSubtitle', 'Find pages from your website and upload PDFs to teach your assistant.')}
          </div>
        </div>

        <div
          style={{
            border: '1px solid var(--flow-border)',
            borderRadius: 'var(--flow-radius)',
            padding: '1rem',
            background: 'var(--flow-surface)',
            marginBottom: '0.5rem',
          }}
        >
          <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.75rem' }}>{t('createBot.discoverPages', 'Discover pages')}</div>

          {!isSharedDiscovering && sharedDiscoveredUrls.length > 0 && sharedDiscoveryDurationLabel != null ? (
            <div className="flow-hint-text" style={{ marginBottom: '0.75rem', color: 'var(--flow-accent)', fontWeight: 600 }}>
              Discovered {sharedDiscoveredUrls.length} page{sharedDiscoveredUrls.length !== 1 ? 's' : ''} in {sharedDiscoveryDurationLabel}.
            </div>
          ) : isSharedDiscovering ? (
            <div className="flow-hint-text discovery-loading" style={{ marginBottom: '0.75rem', color: 'var(--flow-accent)', fontWeight: 600 }}>
              <span className="discovery-loading-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              Discovering pages... {sharedDiscoveredUrls.length} found so far
            </div>
          ) : (
            <div className="flow-hint-text" style={{ marginBottom: '0.75rem' }}>
              {t('createBot.discoverHint', 'Enter your website (or a section of it) to find related pages automatically.')}
            </div>
          )}
          {sharedDiscoveryTimedOutMessage && !isSharedDiscovering && (
            <div className="alert info" style={{ marginBottom: '0.75rem' }}>
              {sharedDiscoveryTimedOutMessage}
            </div>
          )}

          <div style={{ display: 'flex', gap: '8px', marginBottom: '0.75rem' }}>
            <div style={{ flex: 1 }}>
              <input
                type="url"
                value={sharedDiscoveryUrl}
                onChange={(e) => setSharedDiscoveryUrl(e.target.value)}
                placeholder="https://example.com/your-section/"
                disabled={isSharedDiscovering}
                style={{ width: '100%' }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isSharedDiscovering) {
                    void handleSharedDiscoverUrls()
                  }
                }}
              />
            </div>
            {!isSharedDiscovering ? (
              <UiButton
                variant="primary"
                onClick={() => void handleSharedDiscoverUrls()}
                disabled={!sharedDiscoveryUrl.trim()}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                <ScanSearch size={16} />
                {t('createBot.scan', 'Scan')}
              </UiButton>
            ) : (
              <UiButton
                variant="secondary"
                onClick={handleStopSharedDiscovery}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                <StopIcon />
                {t('createBot.stop', 'Stop')}
              </UiButton>
            )}
          </div>

          {sharedDiscoveryError && (
            <div className={`alert ${sharedDiscoveryErrorType || 'error'}`} style={{ marginBottom: '0.75rem' }}>
              {sharedDiscoveryError}
            </div>
          )}

          {showPdfFallback && !isSharedDiscovering && (
            <div
              style={{
                background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                border: '2px solid #0ea5e9',
                borderRadius: '16px',
                padding: '1.25rem',
                marginBottom: '1rem',
              }}
            >
              <div style={{ marginBottom: '0.75rem', color: '#0f172a', fontWeight: 600 }}>
                Automatic scanning is blocked for this website. Upload PDF pages instead.
              </div>
              {pdfFiles.length > 0 && (
                <div style={{
                  background: 'white',
                  borderRadius: '12px',
                  padding: '1rem 1.25rem',
                  marginBottom: '1rem',
                  border: '1px solid rgba(14, 165, 233, 0.3)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem'
                }}>
                  <CheckCircle2 size={24} color="#0ea5e9" strokeWidth={2.5} />
                  <div>
                    <p style={{ margin: 0, fontWeight: 700, color: '#0f172a', fontSize: '1rem' }}>
                      {pdfFiles.length} PDF{pdfFiles.length > 1 ? 's' : ''} ready
                    </p>
                  </div>
                </div>
              )}
              <FileDropzone
                label="Drop your PDFs here"
                helperText="Upload up to 20 PDF files."
                files={pdfFiles}
                setFiles={setPdfFiles}
                accept="application/pdf"
                multiple
                maxFiles={20}
              />
            </div>
          )}

          {(isSharedDiscovering || sharedDiscoveredUrls.length > 0) && (
            <>
              <div className="flow-toolbar" style={{ marginBottom: '0.75rem' }}>
                <UiButton
                  variant={sharedSelectedDiscoveredUrls.size === sharedDiscoveredUrls.length && sharedDiscoveredUrls.length > 0 ? 'ghost' : 'secondary'}
                  onClick={sharedSelectedDiscoveredUrls.size === sharedDiscoveredUrls.length ? handleDeselectAllSharedDiscovered : handleSelectAllSharedDiscovered}
                >
                  {sharedSelectedDiscoveredUrls.size === sharedDiscoveredUrls.length && sharedDiscoveredUrls.length > 0 ? t('common.deselectAll', 'Deselect All') : t('common.selectAll', 'Select All')}
                </UiButton>
                <UiButton
                  variant={expandedSharedCategories.size > 0 ? 'ghost' : 'secondary'}
                  onClick={expandedSharedCategories.size > 0 ? collapseAllShared : expandAllShared}
                >
                  {expandedSharedCategories.size > 0 ? t('common.collapseAll', 'Collapse All') : t('common.expandAll', 'Expand All')}
                </UiButton>
                <span className="muted" style={{ marginLeft: 'auto' }}>
                  {sharedSelectedDiscoveredUrls.size} of {sharedDiscoveredUrls.length} selected
                </span>
              </div>

              <div
                className="url-list"
                style={{
                  maxHeight: '320px',
                  overflowY: 'auto',
                  border: '1px solid var(--flow-border)',
                  borderRadius: 'var(--flow-radius)',
                  padding: '1rem 1.25rem',
                  background: 'var(--flow-bg)',
                  marginBottom: '0.75rem',
                }}
              >
                {isSharedDiscovering && (
                  <div style={{ marginBottom: '12px', color: 'var(--flow-muted)', fontSize: '0.85rem' }}>
                    Scanning... ({sharedDiscoveredUrls.length} found so far)
                  </div>
                )}
                {sharedUrlCategories ? (
                  <div>
                    {Array.from(sharedUrlCategories.children.values())
                      .sort((a, b) => {
                        const countA = getCategoryUrlCount(a)
                        const countB = getCategoryUrlCount(b)
                        if (countA !== countB) return countB - countA
                        return a.name.localeCompare(b.name)
                      })
                      .map((category) => renderSharedCategory(category))}
                    {sharedUrlCategories.urls.length > 0 && (
                      <div style={{ marginLeft: '0px' }}>
                        {sharedUrlCategories.urls.map((url) => (
                          <label
                            key={url}
                            className="url-list-item"
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              cursor: 'pointer',
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={sharedSelectedDiscoveredUrls.has(url)}
                              onChange={() => handleToggleSharedDiscoveredUrl(url)}
                              style={{ marginRight: '8px', cursor: 'pointer' }}
                            />
                            <span style={{ fontSize: '0.85rem', color: 'var(--flow-muted)' }}>{url}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ color: 'var(--flow-muted)' }}>
                    {isSharedDiscovering ? 'Discovering...' : 'No discovered pages yet.'}
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <UiButton
                  variant="secondary"
                  onClick={() => {
                    setSharedDiscoveredUrls([])
                    setSharedSelectedDiscoveredUrls(new Set())
                    setSharedDiscoveryUrl('')
                    setSharedDiscoveryDurationMs(null)
                    setSharedDiscoveryTimedOutMessage(null)
                    setSharedDiscoveryError(null)
                  }}
                  disabled={isSharedDiscovering}
                >
                  Clear
                </UiButton>
              </div>
            </>
          )}
        </div>



        {localError && <div className={`alert ${localErrorType || 'error'}`}>{localError}</div>}

        <div className="flow-actions">
          <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
            Back
          </UiButton>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
            <UiButton variant="ghost" onClick={() => void handleSkip()}>
              {t('createBot.skip', 'Skip for now')}
            </UiButton>
            <UiButton
              variant="primary"
              onClick={() => void handleContinue()}
              disabled={isSharedDiscovering && sharedDiscoveredUrls.length === 0}
            >
              {t('createBot.startTraining', 'Start training')}
            </UiButton>
          </div>
        </div>
      </div>
    )
  }

  /* ── Own-hosting URL tree view ─────────────────────────────────────────── */
  const toggleCategoryExpand = (path: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  const isCategorySelected = (category: UrlCategory): boolean => {
    const categoryUrls = getAllUrlsFromCategory(category)
    return categoryUrls.length > 0 && categoryUrls.every(url => selectedUrls.includes(url))
  }

  const isCategoryPartiallySelected = (category: UrlCategory): boolean => {
    const categoryUrls = getAllUrlsFromCategory(category)
    const selectedCount = categoryUrls.filter(url => selectedUrls.includes(url)).length
    return selectedCount > 0 && selectedCount < categoryUrls.length
  }

  const renderCategory = (category: UrlCategory): React.ReactNode => {
    const categoryUrls = getAllUrlsFromCategory(category)
    const urlCount = getCategoryUrlCount(category)
    const isExpanded = expandedCategories.has(category.path)
    const isSelected = isCategorySelected(category)
    const isPartial = isCategoryPartiallySelected(category)
    const hasChildCategories = category.children.size > 0
    const hasExpandableContent = hasChildCategories || category.urls.length > 0

    return (
      <div key={category.path || 'root'} style={{ marginLeft: `${category.level * 20}px` }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '6px 0',
            cursor: 'pointer',
            userSelect: 'none',
          }}
        >
          {hasExpandableContent ? (
            <span
              onClick={(e) => {
                e.stopPropagation()
                toggleCategoryExpand(category.path)
              }}
              style={{
                marginRight: '8px',
                width: '16px',
                height: '16px',
                transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: 'var(--flow-muted)',
              }}
              aria-hidden
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </span>
          ) : (
            <span style={{ marginRight: '16px', width: '12px' }} />
          )}
          <input
            type="checkbox"
            className="url-checkbox"
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation()
              toggleCategory(category.path, categoryUrls)
            }}
            ref={(input) => {
              if (input) {
                input.indeterminate = isPartial
              }
            }}
            style={{ marginRight: '8px', cursor: 'pointer' }}
          />
          <span
            onClick={() => hasExpandableContent && toggleCategoryExpand(category.path)}
            style={{
              flex: 1,
              cursor: hasExpandableContent ? 'pointer' : 'default',
              fontWeight: 500,
              color: 'var(--flow-text)',
              fontSize: '0.9rem',
            }}
          >
            {getCategoryDisplayPath(category)}
          </span>
          <span
            style={{
              marginLeft: '8px',
              padding: '2px 10px',
              borderRadius: '999px',
              background: 'var(--flow-accent-soft)',
              color: 'var(--flow-accent)',
              fontSize: '0.78rem',
              fontWeight: 600,
            }}
          >
            {urlCount}
          </span>
        </div>
        {hasExpandableContent && isExpanded && (
          <div>
            {Array.from(category.children.values())
              .sort((a, b) => {
                const countA = getCategoryUrlCount(a)
                const countB = getCategoryUrlCount(b)
                if (countA !== countB) return countB - countA
                return a.name.localeCompare(b.name)
              })
              .map(child => renderCategory(child))}
            {category.urls.length > 0 && (
              <div className="url-list-nested" style={{ marginLeft: '20px', paddingLeft: '20px' }}>
                {category.urls.map(url => (
                  <label
                    key={url}
                    className="url-list-item"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="checkbox"
                      className="url-checkbox"
                      checked={selectedUrls.includes(url)}
                      onChange={() => toggleUrl(url)}
                      style={{ marginRight: '8px', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.85rem', color: 'var(--flow-muted)' }}>{url}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flow-panel-body">
      <div>
        {discoveryTimedOutMessage && !isDiscovering && (
          <div className="alert info" style={{ marginBottom: '12px' }}>
            {discoveryTimedOutMessage}
          </div>
        )}
        <div className="card-title">Pick pages to learn from</div>
        <div className="card-subtitle">
          {isDiscovering ? (
            <span className="discovery-loading">
              <span className="discovery-loading-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              <span style={{ color: 'var(--flow-accent)', fontWeight: 500 }}>
                Scanning your website... {discoveredUrls.length} found so far
              </span>
            </span>
          ) : (
            <>
              We found <span style={{ color: 'var(--flow-accent)', fontWeight: 600 }}>{discoveredUrls.length}</span> pages
              on {normalizedWebsiteUrl}. Choose the ones your agent should learn from.
              {discoveryDurationLabel != null && (
                <span style={{ marginLeft: '6px', color: 'var(--flow-muted)', fontSize: '0.85rem' }}>
                  ({discoveryDurationLabel})
                </span>
              )}
            </>
          )}
        </div>
      </div>

      <div className="flow-toolbar">
        <UiButton
          variant={selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0 ? 'ghost' : 'secondary'}
          onClick={selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0 ? deselectAll : selectAll}
        >
          {selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0 ? 'Deselect all' : 'Select all'}
        </UiButton>
        <UiButton
          variant={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
          onClick={expandedCategories.size > 0 ? collapseAll : expandAll}
        >
          {expandedCategories.size > 0 ? 'Collapse all' : 'Expand all'}
        </UiButton>
        <span className="muted" style={{ marginLeft: 'auto' }}>
          {selectedUrls.length} of {discoveredUrls.length} selected
        </span>
      </div>

      <div className="url-list" style={{
        maxHeight: '460px',
        overflowY: 'auto',
        border: '1px solid var(--flow-border)',
        borderRadius: 'var(--flow-radius)',
        padding: '1rem 1.25rem',
        background: 'var(--flow-surface)',
      }}>
        {isDiscovering && (
          <div style={{ marginBottom: '12px', color: 'var(--flow-muted)', fontSize: '0.85rem' }}>
            Scanning... ({discoveredUrls.length} found so far)
          </div>
        )}
        {urlCategories ? (
          <div>
            {Array.from(urlCategories.children.values())
              .sort((a, b) => {
                const countA = getCategoryUrlCount(a)
                const countB = getCategoryUrlCount(b)
                if (countA !== countB) return countB - countA
                return a.name.localeCompare(b.name)
              })
              .map(category => renderCategory(category))}
            {urlCategories.urls.length > 0 && (
              <div style={{ marginLeft: '0px' }}>
                {urlCategories.urls.map(url => (
                  <label
                    key={url}
                    className="url-list-item"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={selectedUrls.includes(url)}
                      onChange={() => toggleUrl(url)}
                      style={{ marginRight: '8px', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.85rem', color: 'var(--flow-muted)' }}>{url}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: 'var(--flow-muted)' }}>{isDiscovering ? 'Discovering...' : 'Loading categories...'}</div>
        )}
      </div>

      <FileDropzone
        label="PDF files (optional)"
        helperText="Drag & drop PDFs here. Your agent can learn from these too."
        files={pdfFiles}
        setFiles={setPdfFiles}
        accept="application/pdf"
        multiple
        maxFiles={20}
      />

      {localError && <div className={`alert ${localErrorType || 'error'}`}>{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        {isDiscovering ? (
          <UiButton variant="primary" onClick={stopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <StopIcon />
            Stop
          </UiButton>
        ) : (
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
            <UiButton variant="ghost" onClick={() => void handleSkip()}>
              Skip for now
            </UiButton>
            <UiButton variant="primary" onClick={handleContinue}>
              Continue
            </UiButton>
          </div>
        )}
      </div>
    </div>
  )
}
