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
  const { step1, step2, flow } = useCreateBotFlow()
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
    reservationPlatform,
    setReservationPlatform,
    restaurantTableCheckUrl,
    setRestaurantTableCheckUrl,
    restaurantTabelogUrl,
    setRestaurantTabelogUrl,
    restaurantHotPepperUrl,
    setRestaurantHotPepperUrl,
  } = step2
  const { businessType } = step1

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

    // Build list of all URLs to discover (for restaurants: main + platform URLs)
    const isRestaurant = businessType === 'restaurant'
    const allUrlsToDiscover: string[] = []

    // Add main shared discovery URL if provided
    const trimmedUrl = sharedDiscoveryUrl.trim()
    if (trimmedUrl) {
      try {
        const withProtocol = /^https?:\/\//i.test(trimmedUrl) ? trimmedUrl : `https://${trimmedUrl}`
        allUrlsToDiscover.push(new URL(withProtocol).href)
      } catch {
        setSharedDiscoveryError(t('createBot.enterValidUrl', 'Enter a valid URL'))
        setSharedDiscoveryErrorType('error')
        return
      }
    }

    // Add platform URL for restaurants (one profile per agent)
    if (isRestaurant && reservationPlatform) {
      const platformUrl = reservationPlatform === 'tabelog' ? restaurantTabelogUrl : reservationPlatform === 'hotpepper' ? restaurantHotPepperUrl : restaurantTableCheckUrl
      const platformUrls = [platformUrl]
        .map((u) => {
          const trimmed = u.trim()
          if (!trimmed) return null
          try {
            const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
            return new URL(withProtocol).href
          } catch {
            return null
          }
        })
        .filter((u) => u !== null) as string[]
      allUrlsToDiscover.push(...platformUrls)
    }

    // Validate that we have at least one URL to discover
    if (allUrlsToDiscover.length === 0) {
      const msg = isRestaurant
        ? 'Enter a website URL or at least one reservation platform URL to discover pages'
        : 'Enter a URL to discover pages'
      setSharedDiscoveryError(t('createBot.enterUrlToDiscoverPages', msg))
      setSharedDiscoveryErrorType('error')
      return
    }

    setIsSharedDiscovering(true)
    setSharedDiscoveredUrls([])
    setSharedSelectedDiscoveredUrls(new Set())
    setSharedDiscoveryDurationMs(null)
    setSharedDiscoveryTimedOutMessage(null)
    setSharedNormalizedDiscoveryUrl(allUrlsToDiscover[0])
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

    let localDiscoveredCount = 0
    let hasShownError = false

    try {
      // Discover all URLs (sequentially for simplicity; results merge into shared state)
      for (const urlToDiscover of allUrlsToDiscover) {
        await discoverUrlsFromHook(
          urlToDiscover,
          discoveryMethod,
          (evt) => {
            if (evt.type === 'discovered' && typeof evt.url === 'string') {
              const url = evt.url
              localDiscoveredCount++
              setSharedDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
              if (!sharedSelectionTouchedRef.current) {
                setSharedSelectedDiscoveredUrls((prev) => new Set([...prev, url]))
              }
              setSharedDiscoveryError(null)
              setSharedDiscoveryErrorType(null)
              setShowPdfFallback(false)
            }
            if (evt.type === 'error' && typeof evt.message === 'string') {
              hasShownError = true
              const reason = (evt as { failure_reason?: string }).failure_reason
              if (reason === 'robots_blocked') {
                setSharedDiscoveryError(t('createBot.websiteBlocksAutomaticScanning', 'This website blocks automatic scanning.'))
                setSharedDiscoveryErrorType('error')
                setShowPdfFallback(true)
              } else if (reason === 'sitemap_empty') {
                setSharedDiscoveryError(
                  t(
                    'createBot.noSitemapSwitchAutomaticRecommended',
                    "No sitemap found. Switch to 'Automatic' discovery (recommended)."
                  )
                )
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
              if ((evt as { timed_out?: boolean }).timed_out === true) {
                setSharedDiscoveryTimedOutMessage(
                  t('createBot.foundMainUrlsTrainNow', 'Found main URLs. You can train on these now.')
                )
              }
            }
          },
          controller.signal,
          { max_duration_sec: 90 }
        )
      }

      // Final check
      if (localDiscoveredCount <= 1 && !hasShownError) {
        hasShownError = true
        setSharedDiscoveryError(
          t(
            'createBot.discoveryNoUsablePagesUsePdf',
            'Discovery completed but found no usable pages. Please use the PDF upload method below.'
          )
        )
        setSharedDiscoveryErrorType('warning')
        setShowPdfFallback(true)
      }
    } catch (err) {
      const e = err as Error & { name?: string }
      if (e.name !== 'AbortError') {
        hasShownError = true
        setSharedDiscoveryError(e.message || t('createBot.discoveryFailed', 'Discovery failed'))
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
    }
  }, [sharedDiscoveryUrl, businessType, reservationPlatform, restaurantTableCheckUrl, restaurantTabelogUrl, restaurantHotPepperUrl, discoverUrlsFromHook, discoveryMethod, t])

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
              {t('createBot.discoveredPagesInDuration', 'Discovered {{count}} pages in {{duration}}.', {
                count: sharedDiscoveredUrls.length,
                duration: sharedDiscoveryDurationLabel,
              })}
            </div>
          ) : isSharedDiscovering ? (
            <div className="flow-hint-text discovery-loading" style={{ marginBottom: '0.75rem', color: 'var(--flow-accent)', fontWeight: 600 }}>
              <span className="discovery-loading-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              {t('createBot.discoveringPagesFoundSoFar', 'Discovering pages... {{count}} found so far', {
                count: sharedDiscoveredUrls.length,
              })}
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

          {/* URL Input Container for all sources */}
          <div
            style={{
              border: '1px solid var(--flow-border)',
              borderRadius: 'var(--flow-radius)',
              padding: '1rem',
              background: 'var(--flow-surface)',
              marginBottom: '0.75rem',
            }}
          >
            {/* Main website URL */}
            <div style={{ marginBottom: '0.6rem' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', marginBottom: '0.3rem' }}>
                {t('createBot.websiteUrl', 'Website URL')}
              </label>
              <input
                type="url"
                value={sharedDiscoveryUrl}
                onChange={(e) => setSharedDiscoveryUrl(e.target.value)}
                placeholder={t('createBot.sharedDiscoveryUrlPlaceholder', 'https://example.com')}
                disabled={isSharedDiscovering}
                style={{ width: '100%' }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isSharedDiscovering) {
                    void handleSharedDiscoverUrls()
                  }
                }}
              />
            </div>

            {/* OR separator for restaurants */}
            {businessType === 'restaurant' && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.15rem 0', marginBottom: '0.6rem' }}>
                  <div style={{ flex: 1, height: '1px', background: 'var(--flow-border)' }} />
                  <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--flow-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    {t('common.or', 'or')}
                  </span>
                  <div style={{ flex: 1, height: '1px', background: 'var(--flow-border)' }} />
                </div>

                {/* Restaurant reservation: one profile per agent */}
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--flow-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>
                    {t('createBot.reservationPlatforms', 'Reservation platform')}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', marginBottom: '0.3rem' }}>
                        {t('botKnowledge.reservationPlatform', 'Platform')}
                      </label>
                      <select
                        value={reservationPlatform}
                        onChange={(e) => setReservationPlatform((e.target.value || '') as '' | 'tabelog' | 'hotpepper' | 'tablecheck')}
                        disabled={isSharedDiscovering}
                        style={{ width: '100%' }}
                      >
                        <option value="">{t('botKnowledge.noReservationPlatform', 'None')}</option>
                        <option value="tabelog">{t('botKnowledge.reservationPlatformTabelog', 'Tabelog')}</option>
                        <option value="hotpepper">{t('botKnowledge.reservationPlatformHotpepper', 'HotPepper')}</option>
                        <option value="tablecheck">{t('botKnowledge.reservationPlatformTablecheck', 'TableCheck')}</option>
                      </select>
                    </div>
                    {reservationPlatform === 'tabelog' && (
                      <div>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', marginBottom: '0.3rem' }}>
                          {t('botKnowledge.tabelogUrl', 'Tabelog URL')}
                        </label>
                        <input
                          type="url"
                          value={restaurantTabelogUrl}
                          onChange={(e) => setRestaurantTabelogUrl(e.target.value)}
                          placeholder="https://tabelog.com/tokyo/A1304/A130401/13224546/"
                          disabled={isSharedDiscovering}
                          style={{ width: '100%' }}
                        />
                      </div>
                    )}
                    {reservationPlatform === 'hotpepper' && (
                      <div>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', marginBottom: '0.3rem' }}>
                          {t('botKnowledge.hotPepperUrl', 'HotPepper URL')}
                        </label>
                        <input
                          type="url"
                          value={restaurantHotPepperUrl}
                          onChange={(e) => setRestaurantHotPepperUrl(e.target.value)}
                          placeholder="https://www.hotpepper.jp/strJ001234567/"
                          disabled={isSharedDiscovering}
                          style={{ width: '100%' }}
                        />
                      </div>
                    )}
                    {reservationPlatform === 'tablecheck' && (
                      <div>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--flow-muted)', marginBottom: '0.3rem' }}>
                          {t('botKnowledge.tableCheckUrl', 'TableCheck URL')}
                        </label>
                        <input
                          type="url"
                          value={restaurantTableCheckUrl}
                          onChange={(e) => setRestaurantTableCheckUrl(e.target.value)}
                          placeholder="https://www.tablecheck.com/en/shops/your-restaurant/reserve"
                          disabled={isSharedDiscovering}
                          style={{ width: '100%' }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Scan button below all URL fields */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '0.75rem' }}>
            {!isSharedDiscovering ? (
              <UiButton
                variant="primary"
                onClick={() => void handleSharedDiscoverUrls()}
                disabled={!sharedDiscoveryUrl.trim() && (businessType !== 'restaurant' || !reservationPlatform || (reservationPlatform === 'tabelog' && !restaurantTabelogUrl.trim()) || (reservationPlatform === 'hotpepper' && !restaurantHotPepperUrl.trim()) || (reservationPlatform === 'tablecheck' && !restaurantTableCheckUrl.trim()))}
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
                {t(
                  'createBot.automaticScanningBlockedUploadPdf',
                  'Automatic scanning is blocked for this website. Upload PDF pages instead.'
                )}
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
                      {t('createBot.pdfReadyCount', '{{count}} PDFs ready', { count: pdfFiles.length })}
                    </p>
                  </div>
                </div>
              )}
              <FileDropzone
                label={t('createBot.dropYourPdfsHere', 'Drop your PDFs here')}
                helperText={t('createBot.uploadUpToPdfFiles', 'Upload up to {{count}} PDF files.', { count: 20 })}
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
                  {t('createBot.selectedOfTotal', '{{selected}} of {{total}} selected', {
                    selected: sharedSelectedDiscoveredUrls.size,
                    total: sharedDiscoveredUrls.length,
                  })}
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
                    {t('createBot.scanningFoundSoFar', 'Scanning... ({{count}} found so far)', {
                      count: sharedDiscoveredUrls.length,
                    })}
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
                    {isSharedDiscovering
                      ? t('createBot.discovering', 'Discovering...')
                      : t('createBot.noDiscoveredPagesYet', 'No discovered pages yet.')}
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
                  {t('common.clear', 'Clear')}
                </UiButton>
              </div>
            </>
          )}
        </div>


        {localError && <div className={`alert ${localErrorType || 'error'}`}>{localError}</div>}

        <div className="flow-actions">
          <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
            {t('common.back', 'Back')}
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
              {t('common.continue', 'Continue')}
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
        <div className="card-title">{t('createBot.pickPagesToLearnFrom', 'Pick pages to learn from')}</div>
        <div className="card-subtitle">
          {isDiscovering ? (
            <span className="discovery-loading">
              <span className="discovery-loading-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              <span style={{ color: 'var(--flow-accent)', fontWeight: 500 }}>
                {t('createBot.scanningWebsiteFoundSoFar', 'Scanning your website... {{count}} found so far', {
                  count: discoveredUrls.length,
                })}
              </span>
            </span>
          ) : (
            <>
              {t('createBot.foundPagesChooseForLearning', 'We found {{count}} pages on {{url}}. Choose the ones your agent should learn from.', {
                count: discoveredUrls.length,
                url: normalizedWebsiteUrl,
              })}
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
          {selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0
            ? t('common.deselectAll', 'Deselect All')
            : t('common.selectAll', 'Select All')}
        </UiButton>
        <UiButton
          variant={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
          onClick={expandedCategories.size > 0 ? collapseAll : expandAll}
        >
          {expandedCategories.size > 0
            ? t('common.collapseAll', 'Collapse All')
            : t('common.expandAll', 'Expand All')}
        </UiButton>
        <span className="muted" style={{ marginLeft: 'auto' }}>
          {t('createBot.selectedOfTotal', '{{selected}} of {{total}} selected', {
            selected: selectedUrls.length,
            total: discoveredUrls.length,
          })}
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
            {t('createBot.scanningFoundSoFar', 'Scanning... ({{count}} found so far)', {
              count: discoveredUrls.length,
            })}
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
          <div style={{ color: 'var(--flow-muted)' }}>
            {isDiscovering
              ? t('createBot.discovering', 'Discovering...')
              : t('createBot.loadingCategories', 'Loading categories...')}
          </div>
        )}
      </div>

      <FileDropzone
        label={t('createBot.pdfFilesOptional', 'PDF files (optional)')}
        helperText={t('createBot.pdfFilesOptionalHelper', 'Drag & drop PDFs here. Your agent can learn from these too.')}
        files={pdfFiles}
        setFiles={setPdfFiles}
        accept="application/pdf"
        multiple
        maxFiles={20}
      />

      {localError && <div className={`alert ${localErrorType || 'error'}`}>{localError}</div>}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          {t('common.back', 'Back')}
        </UiButton>
        {isDiscovering ? (
          <UiButton variant="primary" onClick={stopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <StopIcon />
            {t('createBot.stop', 'Stop')}
          </UiButton>
        ) : (
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
            <UiButton variant="ghost" onClick={() => void handleSkip()}>
              {t('createBot.skip', 'Skip for now')}
            </UiButton>
            <UiButton variant="primary" onClick={handleContinue}>
              {t('common.continue', 'Continue')}
            </UiButton>
          </div>
        )}
      </div>
    </div>
  )
}
