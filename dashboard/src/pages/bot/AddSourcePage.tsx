import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import {
  ScanSearch,
  MousePointerClick,
  Printer,
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  ArrowLeft,
  Plus,
  X,
  FileIcon,
  Upload,
  Globe,
  Type,
} from 'lucide-react'
import { UiButton } from '../../components/ui'
import { FileDropzone } from '../../components/FileDropzone'
import { SegmentedTabs, type SegmentedTabOption } from '../../components/ui'
import { useDashboardData } from '../../hooks/useDashboardData'
import { StopIcon } from '../createBot/DiscoveryIcons'
import {
  categorizeUrls,
  getAllExpandablePaths,
  getAllUrlsFromCategory,
  getCategoryDisplayPath,
  getCategoryUrlCount,
  type UrlCategory,
} from '../createBot/urlCategorizer'
import { AnimatedPage, GlassCard } from '../../components/ui'

type TabId = 'website' | 'pdf' | 'docs' | 'drive' | 'text' | 'custom'

type CustomTextEntry = { id: string; title: string; content: string }

export default function AddSourcePage() {
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
          if (reason === 'no_results' || (Array.isArray(urls) && urls.length <= 1)) {
            hasShownError = true
            setDiscoveryError('Could not discover pages. Use the PDF upload below.')
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
      if (localCount <= 1 && !hasShownError) {
        setDiscoveryError('Discovery completed but found no usable pages. Please use the PDF upload below.')
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
    try {
      if (hasSelectedUrls) {
        const urls = Array.from(selectedDiscoveredUrls)
        const jobId = await queueCrawlUrls(selectedBot.bot_id, urls)
        if (jobId) {
          anyAdded = true
          await loadJobs(selectedBot.bot_id)
        }
      }
      if (hasPdfFiles) {
        const resp = await uploadPdfSources(selectedBot.bot_id, pdfFiles, null)
        if (resp?.items?.length) {
          anyAdded = true
          await loadSources(selectedBot.bot_id)
          await loadJobs(selectedBot.bot_id)
        }
      }
      if (hasDocFiles) {
        const resp = await uploadDocsSources(selectedBot.bot_id, textDocFiles)
        if (resp?.items?.length) anyAdded = true
      }
      if (hasTextContent) {
        const resp = await uploadTextSources(selectedBot.bot_id, [{ content: textContent }])
        if (resp?.items?.length) anyAdded = true
      }
      if (hasCustomEntries) {
        const entries = customTextEntries
          .filter((e) => e.content.trim())
          .map((e) => ({ title: e.title.trim() || undefined, content: e.content.trim() }))
        const resp = await uploadTextSources(selectedBot.bot_id, entries)
        if (resp?.items?.length) anyAdded = true
      }
      if (anyAdded) {
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
    { id: 'website', label: 'Website', icon: <Globe size={16} /> },
    { id: 'pdf', label: 'PDF', icon: <FileText size={16} /> },
    { id: 'docs', label: 'Docs', icon: <FileIcon size={16} /> },
    { id: 'drive', label: 'Drive', icon: <Upload size={16} /> },
    { id: 'text', label: 'Text', icon: <Type size={16} /> },
    { id: 'custom', label: 'Custom', icon: <Plus size={16} /> },
  ]

  if (!botId || !selectedBot || selectedBot.bot_id !== botId) {
    return <div className="empty-panel">Loading…</div>
  }

  return (
    <AnimatedPage>
      <div className="flow-shell">
        <div className="flow-panel-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <Link to={`/bots/${botId}/knowledge`} style={{ display: 'flex', alignItems: 'center', textDecoration: 'none', color: 'inherit' }}>
              <UiButton variant="ghost" style={{ padding: '0.4rem' }}>
                <ArrowLeft size={20} strokeWidth={2} />
              </UiButton>
            </Link>
            <h2 className="ui-section-header-title" style={{ margin: 0 }}>Add source</h2>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <SegmentedTabs value={activeTab} onChange={setActiveTab} options={tabs} ariaLabel="Source type tabs" />
          </div>

          {activeTab === 'website' && (
          <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.75rem' }}>Discover pages</div>

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
            {!isDiscovering && discoveredUrls.length === 0 && (
              <div className="flow-hint-text" style={{ marginBottom: '0.75rem' }}>
                Enter your website (or a section) to find related pages automatically.
              </div>
            )}
            {discoveryTimedOutMessage && !isDiscovering && (
              <div className="alert info" style={{ marginBottom: '0.75rem' }}>
                {discoveryTimedOutMessage}
              </div>
            )}

            <div style={{ display: 'flex', gap: '8px', marginBottom: '0.75rem' }}>
              <div style={{ flex: 1 }}>
                <input
                  type="url"
                  value={discoveryUrl}
                  onChange={(e) => setDiscoveryUrl(e.target.value)}
                  placeholder="https://example.com/your-section/"
                  disabled={isDiscovering}
                  style={{ width: '100%' }}
                  onKeyDown={(e) => e.key === 'Enter' && !isDiscovering && void handleDiscover()}
                />
              </div>
              {!isDiscovering ? (
                <UiButton variant="primary" onClick={() => void handleDiscover()} disabled={!discoveryUrl.trim()} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <ScanSearch size={16} />
                  Scan
                </UiButton>
              ) : (
                <UiButton variant="secondary" onClick={handleStopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <StopIcon />
                  Stop
                </UiButton>
              )}
            </div>

            {discoveryError && (
              <div className={`alert ${discoveryErrorType || 'error'}`} style={{ marginBottom: '0.75rem' }}>
                {discoveryError}
              </div>
            )}

            {showPdfFallback && !isDiscovering && (
              <div
                style={{
                  background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                  border: '2px solid #0ea5e9',
                  borderRadius: '16px',
                  padding: '2rem',
                  marginBottom: '1.5rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
                  <div style={{ width: 48, height: 48, borderRadius: 12, background: '#0ea5e9', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <AlertCircle size={28} color="white" strokeWidth={2.5} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>No problem! We have an easy solution</h3>
                    <p style={{ margin: '0.25rem 0 0', color: '#475569', fontSize: '0.95rem' }}>Follow these 3 simple steps to add your website pages</p>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ background: 'white', borderRadius: 12, padding: '1.25rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0' }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.75rem' }}>
                      <MousePointerClick size={22} color="white" strokeWidth={2.5} />
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#cbd5e1', marginBottom: '0.5rem' }}>STEP 1</div>
                    <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Open your webpage</h4>
                    <p style={{ margin: 0, fontSize: '0.875rem', color: '#64748b', lineHeight: 1.5 }}>Go to the important pages on your website (like Services, Prices, or Contact).</p>
                  </div>
                  <div style={{ background: 'white', borderRadius: 12, padding: '1.25rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0' }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.75rem' }}>
                      <Printer size={22} color="white" strokeWidth={2.5} />
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#cbd5e1', marginBottom: '0.5rem' }}>STEP 2</div>
                    <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Save as PDF</h4>
                    <p style={{ margin: 0, fontSize: '0.875rem', color: '#64748b', lineHeight: 1.5 }}>Right-click the page → <strong>Print</strong> → Choose <strong>&quot;Save as PDF&quot;</strong>.</p>
                  </div>
                  <div style={{ background: 'white', borderRadius: 12, padding: '1.25rem', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0' }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.75rem' }}>
                      <UploadCloud size={22} color="white" strokeWidth={2.5} />
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#cbd5e1', marginBottom: '0.5rem' }}>STEP 3</div>
                    <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Upload here</h4>
                    <p style={{ margin: 0, fontSize: '0.875rem', color: '#64748b', lineHeight: 1.5 }}>Drop your PDF in the box below. Your AI will learn from it!</p>
                  </div>
                </div>

                <div style={{ background: 'rgba(14, 165, 233, 0.1)', borderRadius: 12, padding: '1rem 1.25rem', border: '1px solid rgba(14, 165, 233, 0.3)' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <FileText size={20} color="#0ea5e9" strokeWidth={2} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div>
                      <p style={{ margin: 0, fontSize: '0.875rem', color: '#0f172a', fontWeight: 600 }}>Tip: Do this for every important page</p>
                      <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: '#475569' }}>Save your Services page, Prices, Hours, Contact info, and FAQs as PDFs and upload them in the PDF tab.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {(isDiscovering || discoveredUrls.length > 0) && (
              <>
                <div className="flow-toolbar" style={{ marginBottom: '0.75rem' }}>
                  <UiButton
                    variant={selectedDiscoveredUrls.size === discoveredUrls.length && discoveredUrls.length > 0 ? 'ghost' : 'secondary'}
                    onClick={selectedDiscoveredUrls.size === discoveredUrls.length && discoveredUrls.length > 0 ? handleDeselectAll : handleSelectAll}
                  >
                    {selectedDiscoveredUrls.size === discoveredUrls.length && discoveredUrls.length > 0 ? 'Deselect all' : 'Select all'}
                  </UiButton>
                  <UiButton
                    variant={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
                    onClick={expandedCategories.size > 0 ? collapseAll : expandAll}
                    disabled={!urlCategories}
                  >
                    {expandedCategories.size > 0 ? 'Collapse all' : 'Expand all'}
                  </UiButton>
                  <span className="muted" style={{ marginLeft: 'auto' }}>
                    {selectedDiscoveredUrls.size} of {discoveredUrls.length} selected
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
                      Scanning... ({discoveredUrls.length} found so far)
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
                    <div style={{ color: 'var(--flow-muted)' }}>{isDiscovering ? 'Discovering...' : 'No discovered pages yet.'}</div>
                  )}
                </div>

                <UiButton variant="secondary" onClick={() => { setDiscoveredUrls([]); setSelectedDiscoveredUrls(new Set()); setDiscoveryUrl(''); setDiscoveryDurationMs(null); setDiscoveryTimedOutMessage(null); setDiscoveryError(null); }} disabled={isDiscovering}>
                  Clear
                </UiButton>
              </>
            )}
          </GlassCard>
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
                    {pdfFiles.length} PDF{pdfFiles.length !== 1 ? 's' : ''} ready
                  </p>
                  <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: 'var(--flow-muted)' }}>
                    These will be added to your knowledge base
                  </p>
                </div>
              </div>
            )}

            <div style={{ marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700, color: 'var(--flow-heading)' }}>Add Website Pages as PDFs</h3>
                  <p style={{ margin: '0.25rem 0 0', color: 'var(--flow-muted)', fontSize: '0.95rem' }}>Follow these 3 easy steps</p>
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: '1rem' }}>
              <div className="flow-instruction-card" style={{ background: 'white', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--flow-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <MousePointerClick size={18} color="white" strokeWidth={2.5} />
                  </div>
                  <span className="flow-instruction-card-number" style={{ fontSize: '1.25rem', color: '#cbd5e1' }}>1</span>
                </div>
                <div className="flow-instruction-card-heading">Open the page</div>
                <div className="flow-instruction-card-body">Open important pages (services, prices, hours, booking, contact).</div>
              </div>
              <div className="flow-instruction-card" style={{ background: 'white', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--flow-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Printer size={18} color="white" strokeWidth={2.5} />
                  </div>
                  <span className="flow-instruction-card-number" style={{ fontSize: '1.25rem', color: '#cbd5e1' }}>2</span>
                </div>
                <div className="flow-instruction-card-heading">Save as PDF</div>
                <div className="flow-instruction-card-body">Right-click → Print → &quot;Save as PDF&quot;</div>
              </div>
              <div className="flow-instruction-card" style={{ background: 'white', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--flow-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <UploadCloud size={18} color="white" strokeWidth={2.5} />
                  </div>
                  <span className="flow-instruction-card-number" style={{ fontSize: '1.25rem', color: '#cbd5e1' }}>3</span>
                </div>
                <div className="flow-instruction-card-heading">Upload here</div>
                <div className="flow-instruction-card-body">Drop your PDFs below. Your AI will learn from them!</div>
              </div>
            </div>

            <FileDropzone
              label="Drop your PDFs here"
              helperText="Each PDF teaches your AI about that page. Upload up to 20 files."
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
              <div className="card-title" style={{ marginBottom: '0.5rem' }}>Text documents</div>
              <div className="card-subtitle">Upload .txt, .md, .doc, .docx for your assistant to learn from.</div>
            </div>
            <FileDropzone
              label="Drop text files here"
              helperText="Upload .txt, .md, .doc, .docx (up to 20 files)"
              files={textDocFiles}
              setFiles={setTextDocFiles}
              accept=".txt,.md,.doc,.docx,text/plain,text/markdown,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              multiple
              maxFiles={20}
            />
          </GlassCard>
          )}

          {activeTab === 'drive' && (
          <div style={{ textAlign: 'center', padding: '4rem 2rem', background: 'var(--flow-surface)', borderRadius: 'var(--flow-radius)', border: '1px dashed var(--flow-border)', marginBottom: '1.5rem' }}>
            <Upload size={48} color="var(--flow-muted)" style={{ marginBottom: '1rem' }} />
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--flow-heading)', marginBottom: '0.5rem' }}>Google Drive</div>
            <div style={{ fontSize: '0.95rem', color: 'var(--flow-muted)', marginBottom: '1.5rem' }}>Coming soon! Sync files from Google Drive.</div>
          </div>
          )}

          {activeTab === 'text' && (
          <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ marginBottom: '1rem' }}>
              <div className="card-title" style={{ marginBottom: '0.5rem' }}>Plain text</div>
              <div className="card-subtitle">Paste or type text for your assistant to learn from.</div>
            </div>
            <textarea
              value={textContent}
              onChange={(e) => setTextContent(e.target.value)}
              placeholder="Paste or type your content here..."
              rows={10}
              style={{ width: '100%', resize: 'vertical', fontFamily: 'inherit', padding: '0.75rem', border: '1px solid var(--flow-border)', borderRadius: 'var(--flow-radius)', background: 'var(--flow-surface)' }}
            />
          </GlassCard>
          )}

          {activeTab === 'custom' && (
          <GlassCard className="ui-glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ marginBottom: '1rem' }}>
              <div className="card-title" style={{ marginBottom: '0.5rem' }}>Custom text entries</div>
              <div className="card-subtitle">Create structured entries (FAQs, policies, hours, etc.).</div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {customTextEntries.map((field, idx) => (
                    <div
                      key={field.id}
                      style={{ background: 'var(--flow-surface)', border: '1px solid var(--flow-border)', borderRadius: 'var(--flow-radius)', padding: '1rem' }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                        <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--flow-heading)' }}>Entry #{idx + 1}</div>
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
                      <div style={{ marginBottom: '0.75rem' }}>
                        <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--flow-heading)' }}>Title</label>
                        <input
                          type="text"
                          value={field.title}
                          onChange={(e) => handleUpdateTextField(field.id, 'title', e.target.value)}
                          placeholder="e.g., Return Policy"
                          style={{ width: '100%' }}
                        />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--flow-heading)' }}>Content</label>
                        <textarea
                          value={field.content}
                          onChange={(e) => handleUpdateTextField(field.id, 'content', e.target.value)}
                          placeholder="Enter text..."
                          rows={4}
                          style={{ width: '100%', resize: 'vertical', fontFamily: 'inherit' }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
            <UiButton variant="secondary" onClick={handleAddTextField} style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Plus size={16} />
              Add Another Entry
            </UiButton>
          </GlassCard>
          )}

          {(localError || error) && (
            <div className="alert error" style={{ marginBottom: '1rem' }}>
              {localError || error}
            </div>
          )}

          <div className="flow-actions">
            <UiButton variant="secondary" onClick={() => navigate(`/bots/${botId}/knowledge`)}>Back</UiButton>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
              <UiButton variant="ghost" onClick={() => navigate(`/bots/${botId}/knowledge`)}>Cancel</UiButton>
              <UiButton variant="primary" onClick={() => void handleSubmit()} disabled={!canSubmit}>
                {submitting ? 'Adding…' : 'Add sources'}
              </UiButton>
            </div>
          </div>
        </div>
      </div>
    </AnimatedPage>
  )
}
