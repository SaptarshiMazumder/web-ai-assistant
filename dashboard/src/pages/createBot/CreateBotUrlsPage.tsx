import React, { useEffect, useState, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { MousePointerClick, Printer, UploadCloud } from 'lucide-react'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon, StopIcon } from './DiscoveryIcons'
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

  if (contentHosting === 'shared') {
    return (
      <div className="flow-panel-body">
        <div>
          <div className="card-title">Add info for your helper</div>
          <div className="card-subtitle">Save your important website pages as PDFs, then upload them here.</div>
        </div>

        <div className="muted" style={{ marginTop: '8px' }}>
          Do this for <b>every page</b> on your website that has helpful info about your business (services, prices, hours, booking, contact, location, FAQs).
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '12px' }}>
          <div style={{ border: '1px solid #e2e8f0', borderRadius: '16px', padding: '14px', background: '#fff', boxShadow: '0 10px 28px rgba(15,23,42,0.06)' }}>
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="icon-pill" style={{ background: 'rgba(99,102,241,0.10)', color: '#4f46e5' }}>
                <MousePointerClick size={16} aria-hidden />
              </div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>1</div>
            </div>
            <div style={{ marginTop: '10px', fontWeight: 700, color: '#0f172a' }}>Open the page on your website</div>
            <div className="muted" style={{ marginTop: '6px' }}>
              Go to one important page at a time (services, prices, hours, booking, contact).
            </div>
          </div>

          <div style={{ border: '1px solid #e2e8f0', borderRadius: '16px', padding: '14px', background: '#fff', boxShadow: '0 10px 28px rgba(15,23,42,0.06)' }}>
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="icon-pill" style={{ background: 'rgba(34,197,94,0.12)', color: '#166534' }}>
                <Printer size={16} aria-hidden />
              </div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>2</div>
            </div>
            <div style={{ marginTop: '10px', fontWeight: 700, color: '#0f172a' }}>Print → Save as PDF</div>
            <div className="muted" style={{ marginTop: '6px' }}>
              Right click the page → <b>Print</b> → choose <b>Save as PDF</b> (or “Microsoft Print to PDF”).
            </div>
          </div>

          <div style={{ border: '1px solid #e2e8f0', borderRadius: '16px', padding: '14px', background: '#fff', boxShadow: '0 10px 28px rgba(15,23,42,0.06)' }}>
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="icon-pill" style={{ background: 'rgba(14,165,233,0.12)', color: '#075985' }}>
                <UploadCloud size={16} aria-hidden />
              </div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>3</div>
            </div>
            <div style={{ marginTop: '10px', fontWeight: 700, color: '#0f172a' }}>Upload the PDF here</div>
            <div className="muted" style={{ marginTop: '6px' }}>
              Drop the saved PDF below. Your helper will learn from what’s inside.
            </div>
          </div>
        </div>

        <div style={{ marginTop: '10px' }}>
          <FileDropzone
            label="Upload PDFs"
            helperText="Drag & drop PDFs here."
            files={pdfFiles}
            setFiles={setPdfFiles}
            accept="application/pdf"
            multiple
            maxFiles={20}
          />
        </div>

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
              if (input) {
                input.indeterminate = isPartial
              }
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
  }

  return (
    <div className="flow-panel-body">
      <div>
        {discoveryTimedOutMessage && !isDiscovering && (
          <div className="alert info" style={{ marginBottom: '12px' }}>
            {discoveryTimedOutMessage}
          </div>
        )}
        <div className="card-title">Pick pages from your website</div>
        <div className="card-subtitle">
          {isDiscovering ? (
            <span className="discovery-loading">
              <span className="discovery-loading-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              <span style={{ color: '#6366f1', fontWeight: 500 }}>
                Looking for pages on your website… {discoveredUrls.length} found so far
              </span>
            </span>
          ) : (
            <>
              We found <span style={{ color: '#6366f1', fontWeight: 600 }}>{discoveredUrls.length}</span> pages on {normalizedWebsiteUrl}. Choose the ones your helper should learn from.
              {discoveryDurationLabel != null && (
                <span style={{ marginLeft: '8px', color: '#6366f1', fontWeight: 500 }}>
                  This took {discoveryDurationLabel}.
                </span>
              )}
            </>
          )}
        </div>
      </div>

      <div style={{ marginTop: '10px' }}>
        <FileDropzone
          label="PDF files (optional)"
          helperText="Drag & drop PDFs here. Your helper can learn from these too."
          files={pdfFiles}
          setFiles={setPdfFiles}
          accept="application/pdf"
          multiple
          maxFiles={20}
        />
      </div>

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
        <div className="muted">{selectedUrls.length} selected</div>
      </div>

      <div className="url-list" style={{ maxHeight: '500px', overflowY: 'auto', border: '1px solid #e0e0e0', borderRadius: '4px', padding: '12px' }}>
        {isDiscovering && (
          <div style={{ marginBottom: '12px', color: '#666', fontSize: '14px' }}>
            Looking for pages… ({discoveredUrls.length} found so far)
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
                      style={{ marginRight: '8px', cursor: 'pointer', accentColor: '#6366f1' }}
                    />
                    <span style={{ fontSize: '16px', color: '#334155' }}>{url}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div>{isDiscovering ? 'Discovering…' : 'Loading categories…'}</div>
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
          <div className="row" style={{ gap: '0.75rem', flexWrap: 'wrap' }}>
            <button type="button" className="ghost" onClick={() => void handleSkip()} disabled={isStartingTraining}>
              Skip for now
            </button>
            {hasAnySources && (
              <button
                type="button"
                className="primary"
                onClick={handleStartTraining}
                disabled={isStartingTraining}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
                aria-disabled={isStartingTraining}
              >
                <PlayIcon />
                {isStartingTraining ? 'Starting…' : 'Start training'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
