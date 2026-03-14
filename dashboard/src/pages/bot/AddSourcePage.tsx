import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import {
  ScanSearch,
  FileText,
  CheckCircle2,
  ArrowLeft,
  Plus,
  X,
  FileIcon,
  Globe,
  Type,
} from 'lucide-react'
import { UiButton } from '../../components/ui'
import { FileDropzone } from '../../components/FileDropzone'
import { SegmentedTabs, type SegmentedTabOption } from '../../components/ui'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useTranslation } from 'react-i18next'
import { StopIcon } from '../createBot/DiscoveryIcons'
import {
  categorizeUrls,
  getAllExpandablePaths,
  getAllUrlsFromCategory,
  getCategoryDisplayPath,
  getCategoryUrlCount,
  type UrlCategory,
} from '../createBot/urlCategorizer'
import { AnimatedPage, GlassCard, GlassField } from '../../components/ui'
import { writeAdditionalSourcesRun } from './additionalSourcesRun'

type TabId = 'website' | 'pdf' | 'docs' | 'text' | 'custom'

type CustomTextEntry = { id: string; title: string; content: string }

function newAdditionalSourcesRunId(): string {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID()
    }
  } catch {
    // Fall through.
  }
  return `additional_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
}

export default function AddSourcePage() {
  const { t } = useTranslation()
  const { botId } = useParams()
  const navigate = useNavigate()
  const {
    selectedBot,
    discoverUrls,
    queueCrawlUrls,
    uploadPdfSources,
    uploadTextSources,
    uploadDocsSources,
    loadSources,
    loadJobs,
    loading,
    error,
    setError,
  } = useDashboardData()

  const [discoveryUrl, setDiscoveryUrl] = useState('')
  const [normalizedDiscoveryUrl, setNormalizedDiscoveryUrl] = useState('')
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedDiscoveredUrls, setSelectedDiscoveredUrls] = useState<Set<string>>(new Set())
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [discoveryError, setDiscoveryError] = useState<string | null>(null)
  const [discoveryErrorType, setDiscoveryErrorType] = useState<'error' | 'warning' | null>(null)
  const [discoveryDurationMs, setDiscoveryDurationMs] = useState<number | null>(null)
  const [discoveryTimedOutMessage, setDiscoveryTimedOutMessage] = useState<string | null>(null)
  const [showPdfFallback, setShowPdfFallback] = useState(false)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())
  const [singlePageUrl, setSinglePageUrl] = useState('')
  const [singlePageError, setSinglePageError] = useState<string | null>(null)
  const [addingSinglePage, setAddingSinglePage] = useState(false)
  const [pdfFiles, setPdfFiles] = useState<File[]>([])
  const [textDocFiles, setTextDocFiles] = useState<File[]>([])
  const [textContent, setTextContent] = useState('')
  const [customTextEntries, setCustomTextEntries] = useState<CustomTextEntry[]>([
    { id: '1', title: '', content: '' },
  ])
  const [activeTab, setActiveTab] = useState<TabId>('website')
  const [submitting, setSubmitting] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const startTimeRef = useRef<number | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const timedOutByTimerRef = useRef(false)
  const selectionTouchedRef = useRef(false)
  const discoveryMethod = 'auto'

  const urlCategories = useMemo(() => {
    if (!discoveredUrls.length || !normalizedDiscoveryUrl) return null
    return categorizeUrls(discoveredUrls, normalizedDiscoveryUrl)
  }, [discoveredUrls, normalizedDiscoveryUrl])

  useEffect(() => {
    if (urlCategories && expandedCategories.size === 0) {
      setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
    }
  }, [urlCategories])

  const discoveryDurationLabel =
    discoveryDurationMs != null && !isDiscovering
      ? (() => {
        const sec = Math.round(discoveryDurationMs / 1000)
        if (sec < 60) return `${sec}s`
        return `${Math.floor(sec / 60)}m ${sec % 60}s`
      })()
      : null

  const handleDiscover = useCallback(async () => {
    setDiscoveryError(null)
    setDiscoveryErrorType(null)
    setShowPdfFallback(false)
    const trimmed = discoveryUrl.trim()
    if (!trimmed) {
      setDiscoveryError('Enter a URL to discover pages')
      setDiscoveryErrorType('error')
      return
    }
    let normalized = ''
    try {
      const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
      const parsed = new URL(withProtocol)
      normalized = parsed.href
    } catch {
      setDiscoveryError('Enter a valid URL')
      setDiscoveryErrorType('error')
      return
    }

    setIsDiscovering(true)
    setDiscoveredUrls([])
    setSelectedDiscoveredUrls(new Set())
    setDiscoveryDurationMs(null)
    setDiscoveryTimedOutMessage(null)
    setNormalizedDiscoveryUrl(normalized)
    selectionTouchedRef.current = false
    startTimeRef.current = Date.now()

    const controller = new AbortController()
    abortRef.current = controller
    timedOutByTimerRef.current = false
    timerRef.current = setTimeout(() => {
      timerRef.current = null
      timedOutByTimerRef.current = true
      controller.abort()
    }, 90_000)

    let localCount = 0
    let hasShownError = false

    try {
      await discoverUrls(normalized, discoveryMethod, (evt) => {
        if (evt.type === 'discovered' && typeof evt.url === 'string') {
          const url = evt.url
          localCount++
          setDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
          if (!selectionTouchedRef.current) {
            setSelectedDiscoveredUrls((prev) => new Set([...prev, url]))
          }
          setDiscoveryError(null)
          setDiscoveryErrorType(null)
          setShowPdfFallback(false)
        }
        if (evt.type === 'error' && typeof evt.message === 'string') {
          hasShownError = true
          const reason = evt.failure_reason as string | undefined
          if (reason === 'robots_blocked') {
            setDiscoveryError('This website blocks automatic scanning.')
            setDiscoveryErrorType('error')
            setShowPdfFallback(true)
          } else {
            setDiscoveryError(evt.message)
            setDiscoveryErrorType('error')
          }
        }
        if (evt.type === 'warning' && typeof evt.message === 'string') {
          hasShownError = true
          setDiscoveryError(evt.message)
          setDiscoveryErrorType('warning')
        }
        if (evt.type === 'done') {
          if (timerRef.current) {
            clearTimeout(timerRef.current)
            timerRef.current = null
          }
          const start = startTimeRef.current
          if (start != null) setDiscoveryDurationMs(Date.now() - start)
          setIsDiscovering(false)
          const reason = (evt as { failure_reason?: string }).failure_reason
          const urls = (evt as { urls?: unknown[] }).urls || []
          if (reason === 'no_results' || (Array.isArray(urls) && urls.length === 0)) {
            hasShownError = true
            setDiscoveryError('Could not discover pages. Add sources manually or upload files below.')
            setDiscoveryErrorType('warning')
            setShowPdfFallback(true)
          }
        }
      }, controller.signal, { max_duration_sec: 90 })
    } catch (err) {
      const e = err as Error & { name?: string }
      if (e.name === 'AbortError') {
        const start = startTimeRef.current
        if (start != null) setDiscoveryDurationMs((prev) => (prev === null ? Date.now() - start : prev))
      } else {
        hasShownError = true
        setDiscoveryError(e.message || 'Discovery failed')
        setDiscoveryErrorType('error')
        setShowPdfFallback(true)
      }
    } finally {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
      setIsDiscovering(false)
      abortRef.current = null
      if (localCount === 0 && !hasShownError) {
        setDiscoveryError('Discovery completed but found no usable pages. Add sources manually or upload files below.')
        setDiscoveryErrorType('warning')
        setShowPdfFallback(true)
      }
    }
  }, [discoveryUrl, discoverUrls, discoveryMethod])

  const handleStopDiscovery = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    abortRef.current?.abort()
    setIsDiscovering(false)
  }, [])

  const handleAddSinglePage = useCallback(async () => {
    if (!selectedBot) return
    setSinglePageError(null)
    const trimmed = singlePageUrl.trim()
    if (!trimmed) {
      setSinglePageError('Enter a URL to add')
      return
    }

    let withProtocol: string
    try {
      withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
      new URL(withProtocol)
    } catch {
      setSinglePageError('Enter a valid URL')
      return
    }

    setAddingSinglePage(true)
    try {
      const jobId = await queueCrawlUrls(selectedBot.bot_id, [withProtocol])
      if (jobId) {
        setSinglePageUrl('')
        setSinglePageError(null)
        await loadJobs(selectedBot.bot_id)
      }
    } catch (err) {
      setSinglePageError((err as Error).message || 'Failed to add page')
    } finally {
      setAddingSinglePage(false)
    }
  }, [singlePageUrl, selectedBot, queueCrawlUrls, loadJobs])

  const handleToggleUrl = useCallback((url: string) => {
    selectionTouchedRef.current = true
    setSelectedDiscoveredUrls((prev) => {
      const next = new Set(prev)
      if (next.has(url)) next.delete(url)
      else next.add(url)
      return next
    })
  }, [])

  const toggleCategoryExpand = (path: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const expandAll = () => urlCategories && setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
  const collapseAll = () => setExpandedCategories(new Set())
  const handleSelectAll = () => {
    if (urlCategories) {
      const all = getAllUrlsFromCategory(urlCategories)
      setSelectedDiscoveredUrls(new Set(all))
    }
  }
  const handleDeselectAll = () => setSelectedDiscoveredUrls(new Set())

  const renderCategory = (category: UrlCategory): React.ReactNode => {
    const categoryUrls = getAllUrlsFromCategory(category)
    const urlCount = getCategoryUrlCount(category)
    const isExpanded = expandedCategories.has(category.path)
    const isSelected = categoryUrls.length > 0 && categoryUrls.every((u) => selectedDiscoveredUrls.has(u))
    const isPartial =
      categoryUrls.filter((u) => selectedDiscoveredUrls.has(u)).length > 0 &&
      categoryUrls.filter((u) => selectedDiscoveredUrls.has(u)).length < categoryUrls.length
    const hasExpandable = category.children.size > 0 || category.urls.length > 0

    return (
      <div key={category.path || 'root'} style={{ marginLeft: `${category.level * 20}px` }}>
        <div style={{ display: 'flex', alignItems: 'center', padding: '6px 0', cursor: 'pointer', userSelect: 'none' }}>
          {hasExpandable ? (
            <span
              onClick={(e) => {
                e.stopPropagation()
                toggleCategoryExpand(category.path)
              }}
              style={{
                marginRight: '8px',
                transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease',
                color: 'var(--flow-muted)',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </span>
          ) : (
            <span style={{ marginRight: '16px', width: '12px' }} />
          )}
          <input
            type="checkbox"
            checked={isSelected}
            ref={(el) => {
              if (el) el.indeterminate = isPartial
            }}
            onChange={() => {
              if (isSelected) {
                setSelectedDiscoveredUrls((prev) => {
                  const next = new Set(prev)
                  categoryUrls.forEach((u) => next.delete(u))
                  return next
                })
              } else {
                setSelectedDiscoveredUrls((prev) => new Set([...prev, ...categoryUrls]))
              }
            }}
            style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--flow-accent)' }}
          />
          <span
            onClick={() => hasExpandable && toggleCategoryExpand(category.path)}
            style={{ flex: 1, cursor: hasExpandable ? 'pointer' : 'default', fontWeight: 500, color: 'var(--flow-text)', fontSize: '0.9rem' }}
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
        {hasExpandable && isExpanded && (
          <div>
            {Array.from(category.children.values())
              .sort((a, b) => getCategoryUrlCount(b) - getCategoryUrlCount(a))
              .map((child) => renderCategory(child))}
            {category.urls.length > 0 && (
              <div style={{ marginLeft: '20px', paddingLeft: '20px' }}>
                {category.urls.map((url) => (
                  <label key={url} className="url-list-item" style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={selectedDiscoveredUrls.has(url)}
                      onChange={() => handleToggleUrl(url)}
                      style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--flow-accent)' }}
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

  const handleAddTextField = () => {
    setCustomTextEntries((prev) => [...prev, { id: Date.now().toString(), title: '', content: '' }])
  }
  const handleRemoveTextField = (id: string) => {
    if (customTextEntries.length > 1) setCustomTextEntries((prev) => prev.filter((f) => f.id !== id))
  }
  const handleUpdateTextField = (id: string, field: 'title' | 'content', value: string) => {
    setCustomTextEntries((prev) =>
      prev.map((f) => (f.id === id ? { ...f, [field]: value } : f))
    )
  }

  const hasSelectedUrls = selectedDiscoveredUrls.size > 0
  const hasPdfFiles = pdfFiles.length > 0
  const hasDocFiles = textDocFiles.length > 0
  const hasTextContent = textContent.trim().length > 0
  const hasCustomEntries = customTextEntries.some((e) => e.content.trim())
  const canSubmit =
    (hasSelectedUrls || hasPdfFiles || hasDocFiles || hasTextContent || hasCustomEntries) &&
    !submitting && !loading && !isDiscovering

  const handleSubmit = async () => {
    if (!botId || !selectedBot) return
    setSubmitting(true)
    setLocalError(null)
    setError(null)

    let anyAdded = false
    const submittedJobIds = new Set<string>()
    try {
      if (hasSelectedUrls) {
        const urls = Array.from(selectedDiscoveredUrls)
        const jobId = await queueCrawlUrls(selectedBot.bot_id, urls)
        if (jobId) {
          anyAdded = true
          submittedJobIds.add(String(jobId).trim())
          await loadJobs(selectedBot.bot_id)
        }
      }
      if (hasPdfFiles) {
        const resp = await uploadPdfSources(selectedBot.bot_id, pdfFiles, null)
        if (resp?.items?.length) {
          anyAdded = true
          for (const item of resp.items) {
            const jobId = String(item?.job_id || '').trim()
            if (jobId) submittedJobIds.add(jobId)
          }
          await loadSources(selectedBot.bot_id)
          await loadJobs(selectedBot.bot_id)
        }
      }
      if (hasDocFiles) {
        const resp = await uploadDocsSources(selectedBot.bot_id, textDocFiles)
        if (resp?.items?.length) {
          anyAdded = true
          for (const item of resp.items) {
            const jobId = String(item?.job_id || '').trim()
            if (jobId) submittedJobIds.add(jobId)
          }
        }
      }
      if (hasTextContent) {
        const resp = await uploadTextSources(selectedBot.bot_id, [{ content: textContent }])
        if (resp?.items?.length) {
          anyAdded = true
          for (const item of resp.items) {
            const jobId = String(item?.job_id || '').trim()
            if (jobId) submittedJobIds.add(jobId)
          }
        }
      }
      if (hasCustomEntries) {
        const entries = customTextEntries
          .filter((e) => e.content.trim())
          .map((e) => ({ title: e.title.trim() || undefined, content: e.content.trim() }))
        const resp = await uploadTextSources(selectedBot.bot_id, entries)
        if (resp?.items?.length) {
          anyAdded = true
          for (const item of resp.items) {
            const jobId = String(item?.job_id || '').trim()
            if (jobId) submittedJobIds.add(jobId)
          }
        }
      }
      if (anyAdded) {
        const jobIds = Array.from(submittedJobIds)
        if (jobIds.length > 0) {
          writeAdditionalSourcesRun({
            run_id: newAdditionalSourcesRunId(),
            bot_id: selectedBot.bot_id,
            job_ids: jobIds,
            created_at: Date.now(),
            total_sources: jobIds.length,
          })
        }
        navigate(`/bots/${botId}/knowledge`, { replace: true })
      } else {
        setLocalError('Add at least one source.')
      }
    } catch (err) {
      setLocalError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const tabs: SegmentedTabOption<TabId>[] = [
    { id: 'website', label: t('addSource.tabs.website', 'Website'), icon: <Globe size={16} /> },
    { id: 'pdf', label: t('addSource.tabs.pdf', 'PDF'), icon: <FileText size={16} /> },
    { id: 'docs', label: t('addSource.tabs.docs', 'Docs'), icon: <FileIcon size={16} /> },
    { id: 'text', label: t('addSource.tabs.text', 'Text'), icon: <Type size={16} /> },
    { id: 'custom', label: t('addSource.tabs.custom', 'Custom'), icon: <Plus size={16} /> },
  ]

  if (!botId || !selectedBot || selectedBot.bot_id !== botId) {
    return <div className="empty-panel">Loading…</div>
  }

  return (
    <AnimatedPage>
      <div className="flow-shell">
        <div className="flow-panel-body add-source-page-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <Link to={`/bots/${botId}/knowledge`} style={{ display: 'flex', alignItems: 'center', textDecoration: 'none', color: 'inherit' }}>
              <UiButton variant="ghost" style={{ padding: '0.4rem' }}>
                <ArrowLeft size={20} strokeWidth={2} />
              </UiButton>
            </Link>
            <h2 className="ui-section-header-title" style={{ margin: 0 }}>{t('addSource.title', 'Add source')}</h2>
          </div>

          <div className="add-source-tabs-wrap" style={{ marginBottom: '1.5rem' }}>
            <SegmentedTabs value={activeTab} onChange={setActiveTab} options={tabs} ariaLabel="Source type tabs" />
          </div>

          {activeTab === 'website' && (
            <>
              {/* Add Single Page Section */}
              <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem' }}>{t('addSource.addSinglePage', 'Add Single Page')}</div>
                <div className="flow-hint-text" style={{ marginBottom: '1.25rem' }}>
                  {t('addSource.addSinglePageSubtitle', 'Add a specific page URL to your knowledge base.')}
                </div>

                <div className="add-source-discovery-row" style={{ display: 'flex', gap: '12px', marginBottom: '0.75rem', alignItems: 'flex-start' }}>
                  <GlassField label={t('addSource.pageUrlLabel', 'Page URL')} style={{ flex: 1, minWidth: 0 }}>
                    <input
                      type="url"
                      value={singlePageUrl}
                      onChange={(e) => setSinglePageUrl(e.target.value)}
                      placeholder={t('addSource.pageUrlPlaceholder', 'https://example.com/page')}
                      disabled={addingSinglePage}
                      onKeyDown={(e) => e.key === 'Enter' && !addingSinglePage && void handleAddSinglePage()}
                    />
                  </GlassField>
                  <div className="add-source-discovery-action" style={{ paddingTop: '0.2rem' }}>
                    <UiButton
                      variant="primary"
                      onClick={() => void handleAddSinglePage()}
                      disabled={!singlePageUrl.trim() || addingSinglePage}
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
                    >
                      {addingSinglePage ? t('addSource.adding', 'Adding...') : t('addSource.add', 'Add')}
                    </UiButton>
                  </div>
                </div>

                {singlePageError && (
                  <div className="alert error" style={{ marginBottom: '0.75rem' }}>
                    {singlePageError}
                  </div>
                )}
              </GlassCard>

              {/* OR Divider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '1.5rem 0', opacity: 0.6 }}>
                <div style={{ flex: 1, height: '1px', background: 'var(--ui-flow-border)' }} />
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--ui-flow-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t('addSource.or', 'Or')}</div>
                <div style={{ flex: 1, height: '1px', background: 'var(--ui-flow-border)' }} />
              </div>

              {/* Scan Entire Website Section */}
              <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem' }}>{t('addSource.scanEntireWebsite', 'Scan Entire Website')}</div>
                <div className="flow-hint-text" style={{ marginBottom: '1.25rem' }}>
                  {t('addSource.scanEntireWebsiteSubtitle', 'Enter your website (or a section) to find related pages automatically.')}
                </div>

                {!isDiscovering && discoveredUrls.length > 0 && discoveryDurationLabel != null && (
                  <div className="flow-hint-text" style={{ marginBottom: '0.75rem', color: 'var(--flow-accent)', fontWeight: 600 }}>
                    Discovered {discoveredUrls.length} page{discoveredUrls.length !== 1 ? 's' : ''} in {discoveryDurationLabel}.
                  </div>
                )}
                {isDiscovering && (
                  <div className="flow-hint-text discovery-loading" style={{ marginBottom: '0.75rem', color: 'var(--flow-accent)', fontWeight: 600 }}>
                    <span className="discovery-loading-dots" aria-hidden>
                      <span /><span /><span />
                    </span>
                    Discovering pages... {discoveredUrls.length} found so far
                  </div>
                )}
                {discoveryTimedOutMessage && !isDiscovering && (
                  <div className="alert info" style={{ marginBottom: '0.75rem' }}>
                    {discoveryTimedOutMessage}
                  </div>
                )}

                <div className="add-source-discovery-row" style={{ display: 'flex', gap: '12px', marginBottom: '0.75rem', alignItems: 'flex-start' }}>
                  <GlassField label={t('addSource.websiteUrlLabel', 'Website URL')} style={{ flex: 1, minWidth: 0 }}>
                    <input
                      type="url"
                      value={discoveryUrl}
                      onChange={(e) => setDiscoveryUrl(e.target.value)}
                      placeholder={t('addSource.websiteUrlPlaceholder', 'https://example.com/your-section/')}
                      disabled={isDiscovering}
                      onKeyDown={(e) => e.key === 'Enter' && !isDiscovering && void handleDiscover()}
                    />
                  </GlassField>
                  <div className="add-source-discovery-action" style={{ paddingTop: '0.2rem' }}>
                    {!isDiscovering ? (
                      <UiButton variant="primary" onClick={() => void handleDiscover()} disabled={!discoveryUrl.trim()} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <ScanSearch size={16} />
                        {t('addSource.scan', 'Scan')}
                      </UiButton>
                    ) : (
                      <UiButton variant="secondary" onClick={handleStopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <StopIcon />
                        {t('addSource.stop', 'Stop')}
                      </UiButton>
                    )}
                  </div>
                </div>

                {discoveryError && (
                  <div className={`alert ${discoveryErrorType || 'error'}`} style={{ marginBottom: '0.75rem' }}>
                    {discoveryError}
                  </div>
                )}

                {showPdfFallback && !isDiscovering && (
                  <div className="alert warning" style={{ marginBottom: '0.75rem' }}>
                    {t('addSource.pdfFallbackShort', 'This website blocks automatic scanning. Add sources manually or upload files instead.')}
                  </div>
                )}

                {(isDiscovering || discoveredUrls.length > 0) && (
                  <>
                    <div className="flow-toolbar" style={{ marginBottom: '0.75rem' }}>
                      <UiButton
                        variant={selectedDiscoveredUrls.size === discoveredUrls.length && discoveredUrls.length > 0 ? 'ghost' : 'secondary'}
                        onClick={selectedDiscoveredUrls.size === discoveredUrls.length && discoveredUrls.length > 0 ? handleDeselectAll : handleSelectAll}
                      >
                        {selectedDiscoveredUrls.size === discoveredUrls.length && discoveredUrls.length > 0 ? t('addSource.deselectAll', 'Deselect all') : t('addSource.selectAll', 'Select all')}
                      </UiButton>
                      <UiButton
                        variant={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
                        onClick={expandedCategories.size > 0 ? collapseAll : expandAll}
                        disabled={!urlCategories}
                      >
                        {expandedCategories.size > 0 ? t('addSource.collapseAll', 'Collapse all') : t('addSource.expandAll', 'Expand all')}
                      </UiButton>
                      <span className="muted" style={{ marginLeft: 'auto' }}>
                        {t('addSource.selectedCount', '{{selected}} of {{total}} selected', { selected: selectedDiscoveredUrls.size, total: discoveredUrls.length })}
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
                      {isDiscovering && (
                        <div style={{ marginBottom: 12, color: 'var(--flow-muted)', fontSize: '0.85rem' }}>
                          {t('addSource.scanningSoFar', 'Scanning... ({{count}} found so far)', { count: discoveredUrls.length })}
                        </div>
                      )}
                      {urlCategories ? (
                        <div>
                          {Array.from(urlCategories.children.values())
                            .sort((a, b) => getCategoryUrlCount(b) - getCategoryUrlCount(a))
                            .map((cat) => renderCategory(cat))}
                          {urlCategories.urls.length > 0 && (
                            <div style={{ marginLeft: 0 }}>
                              {urlCategories.urls.map((url) => (
                                <label key={url} className="url-list-item" style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                                  <input
                                    type="checkbox"
                                    checked={selectedDiscoveredUrls.has(url)}
                                    onChange={() => handleToggleUrl(url)}
                                    style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--flow-accent)' }}
                                  />
                                  <span style={{ fontSize: '0.85rem', color: 'var(--flow-muted)' }}>{url}</span>
                                </label>
                              ))}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div style={{ color: 'var(--flow-muted)' }}>{isDiscovering ? t('addSource.discovering', 'Discovering...') : t('addSource.noDiscoveredPagesYet', 'No discovered pages yet.')}</div>
                      )}
                    </div>

                    <UiButton variant="secondary" onClick={() => { setDiscoveredUrls([]); setSelectedDiscoveredUrls(new Set()); setDiscoveryUrl(''); setDiscoveryDurationMs(null); setDiscoveryTimedOutMessage(null); setDiscoveryError(null); }} disabled={isDiscovering}>
                      {t('addSource.clear', 'Clear')}
                    </UiButton>
                  </>
                )}
              </GlassCard>
            </>
          )}

          {activeTab === 'pdf' && (
            <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              {pdfFiles.length > 0 && (
                <div
                  style={{
                    background: 'var(--flow-bg)',
                    borderRadius: 12,
                    padding: '1rem 1.25rem',
                    marginBottom: '1.25rem',
                    border: '1px solid var(--flow-border)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                  }}
                >
                  <CheckCircle2 size={24} color="var(--flow-accent)" strokeWidth={2.5} />
                  <div>
                    <p style={{ margin: 0, fontWeight: 700, color: 'var(--flow-heading)', fontSize: '1rem' }}>
                      {t('addSource.pdfsReady', '{{count}} PDF(s) ready', { count: pdfFiles.length })}
                    </p>
                    <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: 'var(--flow-muted)' }}>
                      {t('addSource.addedToKnowledgeBase', 'These will be added to your knowledge base')}
                    </p>
                  </div>
                </div>
              )}
              <FileDropzone
                label={t('addSource.dropPdfsHere', 'Drop your PDFs here')}
                helperText={t('addSource.dropPdfsHelper', 'Each PDF teaches your AI about that page. Upload up to 20 files.')}
                files={pdfFiles}
                setFiles={setPdfFiles}
                accept="application/pdf"
                multiple
                maxFiles={20}
              />
            </GlassCard>
          )}

          {activeTab === 'docs' && (
            <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <div className="card-title" style={{ marginBottom: '0.5rem' }}>{t('addSource.textDocuments', 'Text documents')}</div>
                <div className="card-subtitle">{t('addSource.textDocumentsSubtitle', 'Upload .txt, .md, .doc, .docx for your assistant to learn from.')}</div>
              </div>
              <FileDropzone
                label={t('addSource.dropTextFilesHere', 'Drop text files here')}
                helperText={t('addSource.dropTextFilesHelper', 'Upload .txt, .md, .doc, .docx (up to 20 files)')}
                files={textDocFiles}
                setFiles={setTextDocFiles}
                accept=".txt,.md,.doc,.docx,text/plain,text/markdown,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                multiple
                maxFiles={20}
              />
            </GlassCard>
          )}



          {activeTab === 'text' && (
            <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <div className="card-title" style={{ marginBottom: '0.5rem' }}>{t('addSource.plainText', 'Plain text')}</div>
                <div className="card-subtitle">{t('addSource.plainTextSubtitle', 'Paste or type text for your assistant to learn from.')}</div>
              </div>
              <textarea
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                placeholder={t('addSource.pasteOrTypePlaceholder', 'Paste or type your content here...')}
                rows={10}
                style={{ width: '100%', resize: 'vertical', fontFamily: 'inherit', padding: '0.75rem', border: '1px solid var(--flow-border)', borderRadius: 'var(--flow-radius)', background: 'var(--flow-surface)' }}
              />
            </GlassCard>
          )}

          {activeTab === 'custom' && (
            <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <div className="card-title" style={{ marginBottom: '0.5rem' }}>{t('addSource.customTextEntries', 'Custom text entries')}</div>
                <div className="card-subtitle">{t('addSource.customTextEntriesSubtitle', 'Create structured entries (FAQs, policies, hours, etc.).')}</div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {customTextEntries.map((field, idx) => (
                  <div
                    key={field.id}
                    style={{ background: 'var(--flow-surface)', border: '1px solid var(--flow-border)', borderRadius: 'var(--flow-radius)', padding: '1rem' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--flow-heading)' }}>{t('addSource.entryNumber', 'Entry #{{count}}', { count: idx + 1 })}</div>
                      {customTextEntries.length > 1 && (
                        <button
                          type="button"
                          onClick={() => handleRemoveTextField(field.id)}
                          style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '0.25rem', color: 'var(--flow-muted)' }}
                        >
                          <X size={18} />
                        </button>
                      )}
                    </div>
                    <div style={{ display: 'grid', gap: '1.25rem' }}>
                      <GlassField label={t('addSource.entryTitleLabel', 'Title')}>
                        <input
                          type="text"
                          value={field.title}
                          onChange={(e) => handleUpdateTextField(field.id, 'title', e.target.value)}
                          placeholder={t('addSource.entryTitlePlaceholder', 'Entry title (optional)')}
                        />
                      </GlassField>
                      <GlassField label={t('addSource.entryContentLabel', 'Content')}>
                        <textarea
                          value={field.content}
                          onChange={(e) => handleUpdateTextField(field.id, 'content', e.target.value)}
                          placeholder={t('addSource.entryContentPlaceholder', 'Type or paste content here...')}
                          rows={6}
                          style={{ minHeight: '120px' }}
                        />
                      </GlassField>
                    </div>
                  </div>
                ))}
              </div>
              <UiButton variant="secondary" onClick={handleAddTextField} style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Plus size={16} />
                {t('addSource.addAnotherEntry', 'Add Another Entry')}
              </UiButton>
            </GlassCard>
          )}

          {(localError || error) && (
            <div className="alert error" style={{ marginBottom: '1rem' }}>
              {localError || error}
            </div>
          )}

          <div className="flow-actions add-source-actions">
            <UiButton variant="secondary" onClick={() => navigate(`/bots/${botId}/knowledge`)}>{t('addSource.back', 'Back')}</UiButton>
            <div className="add-source-actions-right" style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
              <UiButton variant="ghost" onClick={() => navigate(`/bots/${botId}/knowledge`)}>{t('addSource.cancel', 'Cancel')}</UiButton>
              <UiButton variant="primary" onClick={() => void handleSubmit()} disabled={!canSubmit}>
                {submitting ? t('addSource.addingSources', 'Adding...') : t('addSource.addSources', 'Add sources')}
              </UiButton>
            </div>
          </div>
        </div>
      </div>
    </AnimatedPage >
  )
}
