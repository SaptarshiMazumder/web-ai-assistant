import React, { useEffect, useState, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlowIcon } from '../../components/FlowIcon'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon, StopIcon } from './DiscoveryIcons'
import StarBorder from '../../components/StarBorder'
import { categorizeUrls, getAllUrlsFromCategory, getCategoryUrlCount, getCategoryDisplayPath, getAllExpandablePaths, type UrlCategory } from './urlCategorizer'
import { FileDropzone } from '../../components/FileDropzone'

export default function CreateBotUrlsPage() {
  const navigate = useNavigate()
  const { step2, flow } = useCreateBotFlow()
  const {
    discoveredUrls,
    selectedUrls,
    normalizedWebsiteUrl,
    contentHosting,
    pdfFiles,
    setPdfFiles,
    isDiscovering,
    isStartingTraining,
    discoveryDurationMs,
    discoveryTimedOutMessage,
    continueWithoutSources,
    toggleUrl,
    toggleCategory,
    selectAll,
    deselectAll,
    stopDiscovery,
    localError,
    startTraining,
  } = step2

  const hasAnySources = selectedUrls.length > 0 || pdfFiles.length > 0

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

  const urlCategories = useMemo(() => {
    if (!discoveredUrls.length || !normalizedWebsiteUrl) return null
    return categorizeUrls(discoveredUrls, normalizedWebsiteUrl)
  }, [discoveredUrls, normalizedWebsiteUrl])

  useEffect(() => {
    if (!contentHosting) {
      if (flow.prevPath) navigate(flow.prevPath)
      return
    }
    if (contentHosting === 'own' && !discoveredUrls.length && !isDiscovering) {
      if (flow.prevPath) navigate(flow.prevPath)
    }
  }, [discoveredUrls.length, isDiscovering, navigate, flow.firstPath, contentHosting])

  const hasExpandedDefault = useRef(false)
  useEffect(() => {
    if (urlCategories && !hasExpandedDefault.current) {
      setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
      hasExpandedDefault.current = true
    }
  }, [urlCategories])

  const expandAll = () => {
    if (urlCategories) setExpandedCategories(new Set(getAllExpandablePaths(urlCategories)))
  }
  const collapseAll = () => setExpandedCategories(new Set())

  const handleStartTraining = async () => {
    console.log('[Create Bot] Start Training clicked')
    const botId = await startTraining()
    if (botId && flow.nextPath) {
      navigate(flow.nextPath)
    }
  }

  const handleSkip = async () => {
    const botId = await continueWithoutSources()
    if (botId && flow.nextPath) {
      navigate(flow.nextPath)
    }
  }

  /* ── Shared-hosting sub-view ───────────────────────────────────────────── */
  if (contentHosting === 'shared') {
    return (
      <div className="flow-panel-body">
        <div>
          <div className="card-title">Add your business content</div>
          <div className="card-subtitle">
            Save each important page as a PDF, then upload it here. Follow these three steps:
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '12px' }}>
          <div className="flow-instruction-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <FlowIcon name="ads_click" filled style={{ color: 'var(--flow-accent)', fontSize: '22px' }} />
              <div className="flow-instruction-card-number">1</div>
            </div>
            <div className="flow-instruction-card-heading">Open the page</div>
            <div className="flow-instruction-card-body">
              Go to one important page at a time (services, prices, hours, booking, contact).
            </div>
          </div>

          <div className="flow-instruction-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <FlowIcon name="print" filled style={{ color: 'var(--flow-accent)', fontSize: '22px' }} />
              <div className="flow-instruction-card-number">2</div>
            </div>
            <div className="flow-instruction-card-heading">Print as PDF</div>
            <div className="flow-instruction-card-body">
              Right-click the page, choose <b>Print</b>, then <b>Save as PDF</b>.
            </div>
          </div>

          <div className="flow-instruction-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <FlowIcon name="cloud_upload" filled style={{ color: 'var(--flow-accent)', fontSize: '22px' }} />
              <div className="flow-instruction-card-number">3</div>
            </div>
            <div className="flow-instruction-card-heading">Upload here</div>
            <div className="flow-instruction-card-body">
              Drop the saved PDF below. Your agent will learn from it.
            </div>
          </div>
        </div>

        <FileDropzone
          label="Upload PDFs"
          helperText="Drag & drop PDFs here."
          files={pdfFiles}
          setFiles={setPdfFiles}
          accept="application/pdf"
          multiple
          maxFiles={20}
        />

        {localError && <div className="alert error">{localError}</div>}

        <div className="flow-actions">
          <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
            Back
          </button>
          <button type="button" className="primary" onClick={() => flow.nextPath && navigate(flow.nextPath)}>
            Continue
          </button>
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
            style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--flow-accent)' }}
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
              <div style={{ marginLeft: '20px', paddingLeft: '20px' }}>
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
                      checked={selectedUrls.includes(url)}
                      onChange={() => toggleUrl(url)}
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

      <FileDropzone
        label="PDF files (optional)"
        helperText="Drag & drop PDFs here. Your agent can learn from these too."
        files={pdfFiles}
        setFiles={setPdfFiles}
        accept="application/pdf"
        multiple
        maxFiles={20}
      />

      <div className="flow-toolbar">
        <button
          className={selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0 ? 'ghost' : 'secondary'}
          onClick={selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0 ? deselectAll : selectAll}
        >
          {selectedUrls.length === discoveredUrls.length && discoveredUrls.length > 0 ? 'Deselect all' : 'Select all'}
        </button>
        <button
          className={expandedCategories.size > 0 ? 'ghost' : 'secondary'}
          onClick={expandedCategories.size > 0 ? collapseAll : expandAll}
        >
          {expandedCategories.size > 0 ? 'Collapse all' : 'Expand all'}
        </button>
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
                      style={{ marginRight: '8px', cursor: 'pointer', accentColor: 'var(--flow-accent)' }}
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

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        {isDiscovering ? (
          <button type="button" className="primary" onClick={stopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <StopIcon />
            Stop
          </button>
        ) : (
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
            <button type="button" className="ghost" onClick={() => void handleSkip()} disabled={isStartingTraining}>
              Skip for now
            </button>
            {hasAnySources && (
              <StarBorder
                as="button"
                type="button"
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
        )}
      </div>
    </div>
  )
}
