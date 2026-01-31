import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Trash2 } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import {
  categorizeUrls,
  getAllExpandablePaths,
  getAllUrlsFromCategory,
  getCategoryDisplayPath,
  getCategoryUrlCount,
  type UrlCategory,
} from '../createBot/urlCategorizer'

const SOURCES_PER_PAGE = 20

function formatRelativeTime(iso: string): string {
  const d = new Date(iso)
  const diff = Date.now() - d.getTime()
  const s = Math.floor(diff / 1000)
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  const days = Math.floor(h / 24)
  if (days > 0) return `${days} day${days === 1 ? '' : 's'} ago`
  if (h > 0) return `${h} hour${h === 1 ? '' : 's'} ago`
  if (m > 0) return `${m} min ago`
  if (s > 10) return `${s} sec ago`
  return 'Just now'
}

function statusLabel(stage: string): string {
  const s = (stage || '').toLowerCase()
  if (s === 'complete' || s === 'done') return '✓ Trained'
  if (s === 'crawling' || s === 'running' || s === 'pending' || s === 'queued') return 'Training'
  if (s === 'uploading' || s === 'importing' || s === 'import_submitted') return 'Importing'
  if (s === 'failed' || s === 'error') return 'Failed'
  if (s === 'cancelled') return 'Cancelled'
  return stage || '—'
}

function statusPillClass(stage: string): string {
  const s = (stage || '').toLowerCase()
  if (s === 'complete' || s === 'done') return 'pill trained'
  if (s === 'crawling' || s === 'running' || s === 'pending' || s === 'queued' || s === 'uploading' || s === 'importing' || s === 'import_submitted') return 'pill training'
  if (s === 'failed' || s === 'error') return 'pill failed'
  if (s === 'cancelled') return 'pill'
  return 'pill'
}

/** Rough progress 0–100 for training stage (for progress bar). */
function trainingProgressPercent(stage: string | undefined): number {
  const s = (stage || '').toLowerCase()
  if (s === 'done' || s === 'complete' || s === 'error' || s === 'failed' || s === 'cancelled' || s === 'import_submitted') return 100
  if (s === 'uploading' || s === 'importing') return 75
  if (s === 'crawling' || s === 'running' || s === 'pending') return 45
  if (s === 'queued') return 15
  return 10
}

function truncateName(url: string, maxLen = 42): string {
  try {
    const u = new URL(url)
    const path = u.pathname === '/' ? '' : u.pathname
    const full = u.hostname + path
    if (full.length <= maxLen) return full
    return full.slice(0, maxLen - 3) + '...'
  } catch {
    return url.length <= maxLen ? url : url.slice(0, maxLen - 3) + '...'
  }
}

export default function BotKnowledgeTab() {
  const { botId } = useParams()
  const {
    selectedBot,
    jobs,
    sources,
    loading,
    loadJobs,
    loadSources,
    deleteSource,
    queueCrawlUrls,
    discoverUrls,
    getJobStatus,
  } = useDashboardData()

  const [filterUrls, setFilterUrls] = useState('')
  const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(new Set())
  const [page, setPage] = useState(1)
  const [discoverInputUrl, setDiscoverInputUrl] = useState('')
  const [discoveryMethod, setDiscoveryMethod] = useState<'auto' | 'sitemap'>('auto')
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
  const [selectedDiscovered, setSelectedDiscovered] = useState<Set<string>>(new Set())
  const [reTraining, setReTraining] = useState(false)
  const [trainingDiscovered, setTrainingDiscovered] = useState(false)
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null)
  const [deletingSourceId, setDeletingSourceId] = useState<string | null>(null)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())
  const [discoverTrainingJobId, setDiscoverTrainingJobId] = useState<string | null>(null)
  const [discoverTrainingStatus, setDiscoverTrainingStatus] = useState<{
    stage?: string
    docs_count?: number
    last_error?: string
  } | null>(null)
  const [discoverTrainingUrlCount, setDiscoverTrainingUrlCount] = useState(0)
  const [discoverTrainingSuccess, setDiscoverTrainingSuccess] = useState(false)

  const discoverSuccessTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const DISCOVER_TERMINAL_STAGES = new Set(['done', 'error', 'cancelled', 'import_submitted'])
  const DISCOVER_SUCCESS_STAGES = new Set(['done', 'import_submitted'])
  useEffect(() => {
    if (!selectedBot || !discoverTrainingJobId) return
    let cancelled = false
    const poll = async () => {
      const status = await getJobStatus(selectedBot.bot_id, discoverTrainingJobId!)
      if (cancelled || !status) return
      setDiscoverTrainingStatus({ stage: status.stage, docs_count: status.docs_count, last_error: status.last_error })
      if (status.stage && DISCOVER_TERMINAL_STAGES.has(status.stage)) {
        void loadSources(selectedBot.bot_id)
        void loadJobs(selectedBot.bot_id)
        setDiscoverTrainingJobId(null)
        if (status.stage && DISCOVER_SUCCESS_STAGES.has(status.stage)) {
          setDiscoverTrainingSuccess(true)
          if (discoverSuccessTimeoutRef.current) window.clearTimeout(discoverSuccessTimeoutRef.current)
          discoverSuccessTimeoutRef.current = window.setTimeout(() => {
            setDiscoverTrainingStatus(null)
            setDiscoverTrainingUrlCount(0)
            setDiscoverTrainingSuccess(false)
            discoverSuccessTimeoutRef.current = null
          }, 3000)
        } else {
          setDiscoverTrainingStatus(null)
          setDiscoverTrainingUrlCount(0)
        }
      }
    }
    void poll()
    const timer = window.setInterval(poll, 4000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
      if (discoverSuccessTimeoutRef.current) {
        window.clearTimeout(discoverSuccessTimeoutRef.current)
        discoverSuccessTimeoutRef.current = null
      }
    }
  }, [selectedBot, discoverTrainingJobId, getJobStatus, loadJobs, loadSources])

  const normalizedDiscoverUrl = (discoverInputUrl || '').trim().replace(/\/+$/, '') || undefined
  const urlCategories = useMemo(() => {
    if (!discoveredUrls.length || !normalizedDiscoverUrl) return null
    return categorizeUrls(discoveredUrls, normalizedDiscoverUrl)
  }, [discoveredUrls, normalizedDiscoverUrl])

  const hasExpandedDefault = useRef(false)
  useEffect(() => {
    if (!discoveredUrls.length) {
      hasExpandedDefault.current = false
      return
    }
    if (urlCategories && !hasExpandedDefault.current) {
      setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
      hasExpandedDefault.current = true
    }
  }, [urlCategories, discoveredUrls.length])

  // Jobs and sources are already loaded by useDashboardData when selectedBotId changes; no need to refetch on tab mount.

  useEffect(() => {
    setPage(1)
  }, [filterUrls])

  const filteredJobs = filterUrls.trim()
    ? jobs.filter((j) => j.url.toLowerCase().includes(filterUrls.trim().toLowerCase()) || j.hostname.toLowerCase().includes(filterUrls.trim().toLowerCase()))
    : jobs

  const totalPages = Math.max(1, Math.ceil(filteredJobs.length / SOURCES_PER_PAGE))
  const currentPage = Math.min(page, totalPages)
  const pageStart = (currentPage - 1) * SOURCES_PER_PAGE
  const pageJobs = filteredJobs.slice(pageStart, pageStart + SOURCES_PER_PAGE)

  const allOnPageSelected = pageJobs.length > 0 && pageJobs.every((j) => selectedJobIds.has(j.job_id))

  useEffect(() => {
    if (page > totalPages && totalPages >= 1) setPage(totalPages)
  }, [page, totalPages])

  const togglePageSelection = useCallback(() => {
    if (allOnPageSelected) {
      setSelectedJobIds((prev) => {
        const next = new Set(prev)
        pageJobs.forEach((j) => next.delete(j.job_id))
        return next
      })
    } else {
      setSelectedJobIds((prev) => {
        const next = new Set(prev)
        pageJobs.forEach((j) => next.add(j.job_id))
        return next
      })
    }
  }, [allOnPageSelected, pageJobs])

  const toggleJob = useCallback((jobId: string) => {
    setSelectedJobIds((prev) => {
      const next = new Set(prev)
      if (next.has(jobId)) next.delete(jobId)
      else next.add(jobId)
      return next
    })
  }, [])

  const handleReTrainSelected = useCallback(async () => {
    if (!selectedBot || selectedJobIds.size === 0 || reTraining) return
    const urls = jobs.filter((j) => selectedJobIds.has(j.job_id)).map((j) => j.url)
    if (urls.length === 0) return
    setReTraining(true)
    try {
      await queueCrawlUrls(selectedBot.bot_id, urls)
      setSelectedJobIds(new Set())
      await loadJobs(selectedBot.bot_id)
    } finally {
      setReTraining(false)
    }
  }, [selectedBot, selectedJobIds, jobs, queueCrawlUrls, loadJobs, reTraining])

  const handleDiscover = useCallback(async () => {
    if (!discoverInputUrl.trim() || isDiscovering) return
    setIsDiscovering(true)
    setDiscoveredUrls([])
    setSelectedDiscovered(new Set())
    try {
      const result = await discoverUrls(discoverInputUrl.trim(), discoveryMethod, (evt) => {
        if (evt.type === 'discovered' && typeof evt.url === 'string') {
          const url = evt.url
          setDiscoveredUrls((prev) => (prev.includes(url) ? prev : [...prev, url]))
        }
      })
      setDiscoveredUrls(result.urls || [])
    } catch {
      setDiscoveredUrls([])
    } finally {
      setIsDiscovering(false)
    }
  }, [discoverInputUrl, discoveryMethod, isDiscovering, discoverUrls])

  const toggleDiscovered = useCallback((url: string) => {
    setSelectedDiscovered((prev) => {
      const next = new Set(prev)
      if (next.has(url)) next.delete(url)
      else next.add(url)
      return next
    })
  }, [])

  const allDiscoveredSelected = discoveredUrls.length > 0 && selectedDiscovered.size === discoveredUrls.length
  const toggleAllDiscovered = useCallback(() => {
    if (allDiscoveredSelected) setSelectedDiscovered(new Set())
    else setSelectedDiscovered(new Set(discoveredUrls))
  }, [allDiscoveredSelected, discoveredUrls])

  const expandAllCategories = useCallback(() => {
    if (urlCategories) setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
  }, [urlCategories])
  const collapseAllCategories = useCallback(() => setExpandedCategories(new Set()), [])

  const toggleCategoryExpand = useCallback((path: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const isCategorySelected = useCallback(
    (category: UrlCategory): boolean => {
      const categoryUrls = getAllUrlsFromCategory(category)
      return categoryUrls.length > 0 && categoryUrls.every((url) => selectedDiscovered.has(url))
    },
    [selectedDiscovered]
  )
  const isCategoryPartiallySelected = useCallback(
    (category: UrlCategory): boolean => {
      const categoryUrls = getAllUrlsFromCategory(category)
      const count = categoryUrls.filter((url) => selectedDiscovered.has(url)).length
      return count > 0 && count < categoryUrls.length
    },
    [selectedDiscovered]
  )
  const toggleCategory = useCallback((_path: string, categoryUrls: string[]) => {
    setSelectedDiscovered((prev) => {
      const next = new Set(prev)
      const allSelected = categoryUrls.length > 0 && categoryUrls.every((u) => next.has(u))
      if (allSelected) categoryUrls.forEach((u) => next.delete(u))
      else categoryUrls.forEach((u) => next.add(u))
      return next
    })
  }, [])

  const renderDiscoverCategory = useCallback(
    (category: UrlCategory): React.ReactNode => {
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
              padding: '8px 0',
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
                  width: '14px',
                  height: '14px',
                  transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: '#64748b',
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
                toggleCategory(category.path, categoryUrls)
              }}
              ref={(input) => {
                if (input) input.indeterminate = isPartial
              }}
              style={{ marginRight: '8px', cursor: 'pointer', accentColor: '#6366f1' }}
            />
            <span
              onClick={() => hasExpandableContent && toggleCategoryExpand(category.path)}
              style={{ flex: 1, cursor: hasExpandableContent ? 'pointer' : 'default' }}
            >
              {getCategoryDisplayPath(category)}
            </span>
            <span
              style={{
                marginLeft: '8px',
                padding: '2px 8px',
                borderRadius: '999px',
                background: 'rgba(99, 102, 241, 0.15)',
                color: '#4f46e5',
                fontSize: '13px',
                fontWeight: 500,
              }}
            >
              {urlCount} {urlCount === 1 ? 'page' : 'pages'}
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
                .map((child) => renderDiscoverCategory(child))}
              {category.urls.length > 0 && (
                <div style={{ marginLeft: '20px', paddingLeft: '20px' }}>
                  {category.urls.map((url) => (
                    <label
                      key={url}
                      className="url-list-item"
                      style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedDiscovered.has(url)}
                        onChange={() => toggleDiscovered(url)}
                        style={{ marginRight: '8px', cursor: 'pointer', accentColor: '#6366f1' }}
                      />
                      <span style={{ fontSize: '16px', color: '#334155' }}>{url}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )
    },
    [
      expandedCategories,
      isCategorySelected,
      isCategoryPartiallySelected,
      toggleCategoryExpand,
      toggleCategory,
      toggleDiscovered,
      selectedDiscovered,
    ]
  )

  const handleTrainDiscovered = useCallback(async () => {
    if (!selectedBot || selectedDiscovered.size === 0 || trainingDiscovered) return
    const urls = Array.from(selectedDiscovered)
    setTrainingDiscovered(true)
    try {
      const jobId = await queueCrawlUrls(selectedBot.bot_id, urls)
      if (jobId) {
        setDiscoverTrainingJobId(jobId)
        setDiscoverTrainingUrlCount(urls.length)
        setSelectedDiscovered(new Set())
        setDiscoveredUrls([])
        setDiscoverInputUrl('')
      } else {
        await loadJobs(selectedBot.bot_id)
      }
    } finally {
      setTrainingDiscovered(false)
    }
  }, [selectedBot, selectedDiscovered, queueCrawlUrls, loadJobs, trainingDiscovered])

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to manage knowledge.</div>
  }

  const handleDeleteSource = useCallback(
    async (sourceId: string) => {
      if (!selectedBot || deletingSourceId) return
      setDeletingSourceId(sourceId)
      try {
        await deleteSource(selectedBot.bot_id, sourceId)
      } finally {
        setDeletingSourceId(null)
      }
    },
    [selectedBot, deleteSource, deletingSourceId]
  )

  /** For Source column: URL or config summary (not display name). */
  function sourceUrlOrConfig(source: { type: string; config: Record<string, unknown> }): string {
    if (source.type === 'url' && typeof source.config?.url === 'string') return source.config.url
    if (source.type === 'drive' && typeof source.config?.folder_id === 'string') return `Drive folder: ${source.config.folder_id}`
    if (source.type === 'docs' && typeof source.config?.doc_id === 'string') return `Doc: ${source.config.doc_id}`
    return source.type || '—'
  }

  /** For Name column: display_name or fallback from URL (pathname/hostname) for URL sources. */
  function sourceDisplayName(source: { type: string; config: Record<string, unknown>; display_name?: string | null }): string {
    if (source.display_name?.trim()) return source.display_name.trim()
    if (source.type === 'url' && typeof source.config?.url === 'string') {
      try {
        const u = new URL(source.config.url)
        const path = u.pathname.replace(/\/+$/, '') || '/'
        if (path !== '/') return path.length > 80 ? path.slice(0, 77) + '...' : path
        return u.hostname || '—'
      } catch {
        return source.config.url.length > 60 ? source.config.url.slice(0, 57) + '...' : source.config.url
      }
    }
    return '—'
  }

  function sourceTypeLabel(type: string): string {
    const t = (type || '').toLowerCase()
    if (t === 'url') return 'URL'
    if (t === 'drive') return 'Drive'
    if (t === 'docs') return 'Google Docs'
    return type || '—'
  }

  return (
    <div className="card-grid">
      {/* Sources: main table — one row per source (URL, Drive, Docs, etc.) */}
      <section className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="card-title">Sources</div>
        <p className="card-subtitle" style={{ marginTop: 0, marginBottom: '1rem' }}>
          Every source (URL, Drive, Docs, etc.) this bot learns from. Add a URL or connect Drive/Docs. Training runs in the background.
        </p>
        <div className="knowledge-toolbar" style={{ marginBottom: '1rem' }}>
          <Link to={botId ? `/bots/${botId}/sources/new` : '#'} className="primary">
            + Add source
          </Link>
        </div>
        {sources.length > 0 ? (
          <div className="knowledge-table-wrap knowledge-table-wrap-scroll">
            <table className="knowledge-table">
              <thead>
                <tr>
                  <th style={{ width: '120px' }}>Type</th>
                  <th style={{ width: '160px' }}>Name</th>
                  <th>Source</th>
                  <th>Added</th>
                  <th style={{ width: '80px' }}></th>
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => (
                  <tr key={s.source_id}>
                    <td>
                      <span className="source-type-badge" data-type={s.type.toLowerCase()}>
                        {sourceTypeLabel(s.type)}
                      </span>
                    </td>
                    <td className="knowledge-name">{sourceDisplayName(s)}</td>
                    <td className="knowledge-name" style={{ wordBreak: 'break-all' }}>
                      {sourceUrlOrConfig(s)}
                    </td>
                    <td className="muted">{formatRelativeTime(s.updated_at)}</td>
                    <td>
                      <button
                        type="button"
                        onClick={() => handleDeleteSource(s.source_id)}
                        disabled={deletingSourceId === s.source_id}
                        aria-label={`Delete ${sourceDisplayName(s)}`}
                        style={{
                          padding: '0.25rem',
                          background: 'none',
                          border: 'none',
                          cursor: deletingSourceId === s.source_id ? 'not-allowed' : 'pointer',
                          color: deletingSourceId === s.source_id ? '#94a3b8' : '#dc2626',
                          opacity: deletingSourceId === s.source_id ? 0.7 : 1,
                        }}
                      >
                        {deletingSourceId === s.source_id ? (
                          <span style={{ fontSize: '0.875rem' }}>…</span>
                        ) : (
                          <Trash2 size={18} aria-hidden />
                        )}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty muted" style={{ padding: '1.5rem' }}>
            No sources yet. Add a URL or connect Drive/Google Docs to train this bot.
          </div>
        )}

        {/* Run history: collapsible, so main view is Sources only */}
        {jobs.length > 0 && (
          <details className="run-history-details" style={{ marginTop: '1.5rem' }}>
            <summary className="muted" style={{ cursor: 'pointer', fontSize: '0.875rem' }}>
              Run history ({jobs.length})
            </summary>
            <div className="knowledge-toolbar" style={{ marginTop: '0.75rem' }}>
              <input
                type="text"
                className="design-form-input"
                placeholder="Filter..."
                value={filterUrls}
                onChange={(e) => setFilterUrls(e.target.value)}
                style={{ maxWidth: '200px' }}
              />
            </div>
            <div className="knowledge-table-wrap" style={{ marginTop: '0.5rem' }}>
              <table className="knowledge-table">
                <thead>
                  <tr>
                    <th style={{ width: '40px' }}>
                      <input
                        type="checkbox"
                        checked={allOnPageSelected}
                        onChange={togglePageSelection}
                        aria-label="Select all on page"
                      />
                    </th>
                    <th>URL</th>
                    <th>Pages</th>
                    <th>Status</th>
                    <th>Last run</th>
                  </tr>
                </thead>
                <tbody>
                  {pageJobs.map((job) => {
                    const urls = job.crawled_urls ?? []
                    const hasUrls = urls.length > 0
                    const isExpanded = expandedJobId === job.job_id
                    return (
                      <Fragment key={job.job_id}>
                        <tr>
                          <td>
                            <input
                              type="checkbox"
                              checked={selectedJobIds.has(job.job_id)}
                              onChange={() => toggleJob(job.job_id)}
                              aria-label={`Select ${job.url}`}
                            />
                          </td>
                          <td className="knowledge-name">
                            {hasUrls ? (
                              <button
                                type="button"
                                className="ghost"
                                onClick={() => setExpandedJobId((id) => (id === job.job_id ? null : job.job_id))}
                                style={{ padding: 0, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                                aria-expanded={isExpanded}
                              >
                                <span aria-hidden>{isExpanded ? '▼' : '▶'}</span>
                                {truncateName(job.url)}
                              </button>
                            ) : (
                              truncateName(job.url)
                            )}
                          </td>
                          <td>
                            {hasUrls ? `${urls.length} URLs` : job.docs_count > 0 ? `${job.docs_count} pages` : job.pages_crawled > 0 ? `${job.pages_crawled} pages` : '—'}
                          </td>
                          <td>
                            <span className={statusPillClass(job.stage)}>{statusLabel(job.stage)}</span>
                          </td>
                          <td className="muted">{formatRelativeTime(job.updated_at)}</td>
                        </tr>
                        {isExpanded && hasUrls && (
                          <tr key={`${job.job_id}-urls`} className="knowledge-detail-row">
                            <td colSpan={5} style={{ padding: '0.5rem 1rem 1rem 2.5rem', verticalAlign: 'top', borderTop: 'none' }}>
                              <div className="muted" style={{ fontSize: '0.8125rem', marginBottom: '0.35rem' }}>
                                Pages crawled ({urls.length}):
                              </div>
                              <ul className="list" style={{ margin: 0, paddingLeft: '1.25rem', maxHeight: '160px', overflowY: 'auto' }}>
                                {urls.map((u) => (
                                  <li key={u} style={{ wordBreak: 'break-all' }}>
                                    <a href={u} target="_blank" rel="noopener noreferrer" className="muted" style={{ fontSize: '0.8125rem' }}>
                                      {u}
                                    </a>
                                  </li>
                                ))}
                              </ul>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {filteredJobs.length > 0 && (
              <div className="knowledge-footer" style={{ marginTop: '0.75rem' }}>
                <span className="muted">{selectedJobIds.size} selected</span>
                <button
                  type="button"
                  className="secondary"
                  onClick={handleReTrainSelected}
                  disabled={selectedJobIds.size === 0 || loading || reTraining}
                >
                  {reTraining ? 'Starting…' : 'Re-train selected'}
                </button>
                <div className="knowledge-pagination">
                  <span className="muted">Page {currentPage} of {totalPages}</span>
                  <button type="button" className="ghost" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={currentPage <= 1}>
                    Previous
                  </button>
                  <button type="button" className="ghost" onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={currentPage >= totalPages}>
                    Next
                  </button>
                </div>
              </div>
            )}
          </details>
        )}
      </section>

      {/* Add more pages — same UI as create-bot URL selection */}
      <section className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="card-title">Add more pages</div>
        <p className="card-subtitle" style={{ marginTop: 0 }}>
          {isDiscovering ? (
            <span className="discovery-loading" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="discovery-loading-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              <span style={{ color: '#6366f1', fontWeight: 500 }}>
                Discovering pages… {discoveredUrls.length} found so far
              </span>
            </span>
          ) : (
            <>
              Enter a website URL to discover pages. Choose the ones your bot should learn from.
              {discoveredUrls.length > 0 && (
                <span style={{ marginLeft: '8px', color: '#6366f1', fontWeight: 500 }}>
                  {discoveredUrls.length} page{discoveredUrls.length === 1 ? '' : 's'} found.
                </span>
              )}
            </>
          )}
        </p>
        <div className="design-form stack" style={{ marginBottom: '1rem' }}>
          <div className="row" style={{ flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
            <input
              type="url"
              value={discoverInputUrl}
              onChange={(e) => setDiscoverInputUrl(e.target.value)}
              placeholder="https://example.com"
              className="design-form-input"
              style={{ flex: 1, minWidth: '200px' }}
            />
            <select
              value={discoveryMethod}
              onChange={(e) => setDiscoveryMethod(e.target.value as 'auto' | 'sitemap')}
              className="design-form-input"
              style={{ minWidth: '140px' }}
            >
              <option value="auto">Automatic</option>
              <option value="sitemap">Sitemap only</option>
            </select>
            <button type="button" className="primary" onClick={handleDiscover} disabled={!discoverInputUrl.trim() || loading || isDiscovering}>
              {isDiscovering ? 'Discovering…' : 'Discover'}
            </button>
          </div>
        </div>

        {(discoverTrainingJobId || discoverTrainingSuccess) && (
          <div className="progress-card" style={{ marginBottom: '1rem', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
            {discoverTrainingSuccess ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#059669', fontWeight: 500 }}>
                <span aria-hidden style={{ fontSize: '1.25rem' }}>✓</span>
                <span>Added to bot knowledge</span>
              </div>
            ) : (
              <>
                <div className="progress-label" style={{ color: '#334155' }}>
                  Training in progress… {discoverTrainingUrlCount} page{discoverTrainingUrlCount === 1 ? '' : 's'}
                </div>
                <div className="progress-track" style={{ marginTop: '0.5rem' }}>
                  <div
                    className="progress-fill"
                    style={{ width: `${trainingProgressPercent(discoverTrainingStatus?.stage)}%` }}
                  />
                </div>
                <div className="muted" style={{ fontSize: '0.875rem', marginTop: '0.35rem' }}>
                  {discoverTrainingStatus?.stage ? statusLabel(discoverTrainingStatus.stage) : 'Starting…'}
                  {discoverTrainingStatus?.docs_count != null && discoverTrainingStatus.docs_count > 0 && (
                    <> · {discoverTrainingStatus.docs_count} doc{discoverTrainingStatus.docs_count === 1 ? '' : 's'}</>
                  )}
                </div>
                {discoverTrainingStatus?.last_error && (
                  <div className="alert error" style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                    {discoverTrainingStatus.last_error}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {discoveredUrls.length > 0 && !isDiscovering && !discoverTrainingJobId && !discoverTrainingSuccess && (
          <>
            <div className="flow-toolbar" style={{ marginBottom: '0.75rem' }}>
              <button
                className={allDiscoveredSelected ? 'ghost' : 'secondary'}
                onClick={toggleAllDiscovered}
              >
                {allDiscoveredSelected ? 'Deselect all' : 'Select all'}
              </button>
              <button
                className={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
                onClick={expandedCategories.size > 0 ? collapseAllCategories : expandAllCategories}
              >
                {expandedCategories.size > 0 ? 'Collapse all' : 'Expand all'}
              </button>
              <div className="muted">{selectedDiscovered.size} selected</div>
            </div>
            <div className="url-list" style={{ maxHeight: '500px', overflowY: 'auto', border: '1px solid #e0e0e0', borderRadius: '4px', padding: '12px' }}>
              {urlCategories ? (
                <div>
                  {Array.from(urlCategories.children.values())
                    .sort((a, b) => {
                      const countA = getCategoryUrlCount(a)
                      const countB = getCategoryUrlCount(b)
                      if (countA !== countB) return countB - countA
                      return a.name.localeCompare(b.name)
                    })
                    .map((category) => renderDiscoverCategory(category))}
                  {urlCategories.urls.length > 0 && (
                    <div style={{ marginLeft: 0 }}>
                      {urlCategories.urls.map((url) => (
                        <label
                          key={url}
                          className="url-list-item"
                          style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                        >
                          <input
                            type="checkbox"
                            checked={selectedDiscovered.has(url)}
                            onChange={() => toggleDiscovered(url)}
                            style={{ marginRight: '8px', cursor: 'pointer', accentColor: '#6366f1' }}
                          />
                          <span style={{ fontSize: '16px', color: '#334155' }}>{url}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="muted">Loading categories…</div>
              )}
            </div>
            <div className="flow-actions" style={{ marginTop: '1rem' }}>
              <button
                type="button"
                className="primary"
                onClick={handleTrainDiscovered}
                disabled={selectedDiscovered.size === 0 || loading || trainingDiscovered}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
              >
                {trainingDiscovered ? 'Starting…' : '▷ Start training'}
              </button>
            </div>
          </>
        )}
      </section>

    </div>
  )
}
