import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useTranslation } from 'react-i18next'
import {
  Plus,
  Trash2,
  Loader2,
  Image,
  Pencil,
  X,
  Check,
  Upload,
  Sparkles,
  CheckSquare,
  Square,
} from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton, GlassCard, GlassField } from '../../components/ui'

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin
const ASSET_LIMIT_FALLBACK = 50

type AssetRecord = {
  asset_id: string
  bot_id: string
  org_id: string
  name: string
  description: string
  image_url: string
  link_url: string | null
  keywords: string[]
  is_active: boolean
  created_at: string
  updated_at: string
}

type AssetListResponse = {
  assets: AssetRecord[]
  count?: number
  limit?: number
}

type IndexJobRecord = {
  job_id: string
  url: string
  crawled_urls?: string[]
}

type AutoExtractResponse = {
  ok: boolean
  job_id?: string
  assets_extracted: number
  assets_count?: number
  assets_limit?: number
  pages_considered?: number
}

type ExtractionStatusResponse = {
  job_id: string
  status: string
  assets_discovered: number
  assets_downloaded: number
  assets_created: number
  assets_total: number
  limit: number
  error?: string
}

export default function BotImageAssetsTab() {
  const { t } = useTranslation()
  const { botId } = useParams()
  const { getAccessTokenSilently } = useAuth0()

  const [assets, setAssets] = useState<AssetRecord[]>([])
  const [assetCount, setAssetCount] = useState(0)
  const [assetLimit, setAssetLimit] = useState(ASSET_LIMIT_FALLBACK)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Add form state
  const [showAdd, setShowAdd] = useState(false)
  const [addName, setAddName] = useState('')
  const [addDescription, setAddDescription] = useState('')
  const [addLinkUrl, setAddLinkUrl] = useState('')
  const [addKeywords, setAddKeywords] = useState('')
  const [addFile, setAddFile] = useState<File | null>(null)
  const [addPreview, setAddPreview] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editLinkUrl, setEditLinkUrl] = useState('')
  const [editKeywords, setEditKeywords] = useState('')
  const [editFile, setEditFile] = useState<File | null>(null)
  const [editSaving, setEditSaving] = useState(false)

  const [deleting, setDeleting] = useState<string | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [extractResult, setExtractResult] = useState<string | null>(null)
  const [showExtractSettings, setShowExtractSettings] = useState(false)
  const [loadingExtractPages, setLoadingExtractPages] = useState(false)
  const [extractPages, setExtractPages] = useState<string[]>([])
  const [selectedExtractPages, setSelectedExtractPages] = useState<Set<string>>(new Set())
  const [extractPageFilter, setExtractPageFilter] = useState('')

  // Async extraction state
  const [extractStats, setExtractStats] = useState<ExtractionStatusResponse | null>(null)

  // Multi-select state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const hasReachedAssetLimit = assetCount >= assetLimit

  const authedFetch = useCallback(
    async (path: string, init?: RequestInit): Promise<Response> => {
      const token = await getAccessTokenSilently()
      return fetch(`${API_BASE}${path}`, {
        ...init,
        headers: {
          ...(init?.headers || {}),
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      })
    },
    [getAccessTokenSilently]
  )

  const loadAssets = useCallback(async () => {
    if (!botId) return
    setLoading(true)
    setError(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/image-assets`)
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as AssetListResponse
      const nextAssets = data.assets || []
      setAssets(nextAssets)
      setAssetCount(typeof data.count === 'number' ? data.count : nextAssets.length)
      setAssetLimit(typeof data.limit === 'number' ? data.limit : ASSET_LIMIT_FALLBACK)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [botId, authedFetch])

  useEffect(() => {
    void loadAssets()
  }, [loadAssets])

  useEffect(() => {
    setSelectedIds((prev) => {
      const valid = new Set(assets.map((a) => a.asset_id))
      const next = new Set<string>()
      prev.forEach((id) => {
        if (valid.has(id)) next.add(id)
      })
      return next
    })
  }, [assets])

  const loadExtractPages = useCallback(async () => {
    if (!botId) return
    setLoadingExtractPages(true)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/jobs`)
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as { jobs: IndexJobRecord[] }
      const seen = new Set<string>()
      const urls: string[] = []

      for (const job of data.jobs || []) {
        const crawlUrls = Array.isArray(job.crawled_urls) ? job.crawled_urls : []
        for (const raw of crawlUrls) {
          const cleaned = (raw || '').trim()
          if (!cleaned || seen.has(cleaned)) continue
          seen.add(cleaned)
          urls.push(cleaned)
        }

        const fallback = (job.url || '').trim()
        if (fallback && !seen.has(fallback)) {
          seen.add(fallback)
          urls.push(fallback)
        }
      }

      setExtractPages(urls)
      setSelectedExtractPages((prev) => {
        if (!prev.size) return new Set(urls)
        const next = new Set<string>()
        prev.forEach((url) => {
          if (seen.has(url)) next.add(url)
        })
        return next
      })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoadingExtractPages(false)
    }
  }, [authedFetch, botId])

  useEffect(() => {
    let intervalId: number | undefined

    const checkStatus = async () => {
      if (!botId) return
      try {
        const resp = await authedFetch(`/v1/org/bots/${botId}/image-assets/extract-status`)
        if (resp.ok) {
          const data = (await resp.json()) as ExtractionStatusResponse
          setExtractStats(data)

          if (data.status === 'queued' || data.status === 'running') {
            setExtracting(true)
          } else if (data.status === 'done') {
            if (extracting) {
              // Job just finished while we were watching
              setExtractResult(`Extracted ${data.assets_created} new assets.`)
              void loadAssets()
            }
            setExtracting(false)
          } else if (data.status === 'error') {
            if (extracting) {
              setError(data.error || 'Extraction failed')
            }
            setExtracting(false)
          }
        }
      } catch (e) {
        console.error('Failed to check extraction status', e)
      }
    }

    // Check on mount to resume if needed
    void checkStatus()

    if (extracting) {
      intervalId = window.setInterval(checkStatus, 3000)
    }

    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [botId, extracting, authedFetch, loadAssets])

  const toggleExtractSettings = async () => {
    const nextOpen = !showExtractSettings
    setShowExtractSettings(nextOpen)
    if (nextOpen && extractPages.length === 0) {
      await loadExtractPages()
    }
  }


  // File preview
  const handleFileChange = (file: File | null, setFileFn: (f: File | null) => void, setPreviewFn?: (url: string | null) => void) => {
    setFileFn(file)
    if (setPreviewFn) {
      if (file) {
        const reader = new FileReader()
        reader.onload = () => setPreviewFn(reader.result as string)
        reader.readAsDataURL(file)
      } else {
        setPreviewFn(null)
      }
    }
  }

  const handleAdd = async () => {
    if (!botId || !addFile || !addName.trim()) return
    if (hasReachedAssetLimit) {
      setError(`Asset limit reached (${assetLimit}). Delete assets or raise the limit before adding new ones.`)
      return
    }
    setSaving(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', addFile)
      form.append('name', addName.trim())
      form.append('description', addDescription.trim())
      if (addLinkUrl.trim()) form.append('link_url', addLinkUrl.trim())
      if (addKeywords.trim()) form.append('keywords', addKeywords.trim())

      const resp = await authedFetch(`/v1/org/bots/${botId}/image-assets`, {
        method: 'POST',
        body: form,
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      setShowAdd(false)
      setAddName('')
      setAddDescription('')
      setAddLinkUrl('')
      setAddKeywords('')
      setAddFile(null)
      setAddPreview(null)
      await loadAssets()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const startEdit = (a: AssetRecord) => {
    setEditingId(a.asset_id)
    setEditName(a.name)
    setEditDescription(a.description)
    setEditLinkUrl(a.link_url || '')
    setEditKeywords((a.keywords || []).join(', '))
    setEditFile(null)
  }

  const handleEdit = async () => {
    if (!botId || !editingId || !editName.trim()) return
    setEditSaving(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('name', editName.trim())
      form.append('description', editDescription.trim())
      form.append('link_url', editLinkUrl.trim())
      form.append('keywords', editKeywords.trim())
      if (editFile) form.append('file', editFile)

      const resp = await authedFetch(`/v1/org/bots/${botId}/image-assets/${editingId}`, {
        method: 'PUT',
        body: form,
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      setEditingId(null)
      await loadAssets()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setEditSaving(false)
    }
  }

  const handleDelete = async (assetId: string) => {
    if (!botId) return
    setDeleting(assetId)
    setError(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/image-assets/${assetId}`, {
        method: 'DELETE',
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      await loadAssets()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setDeleting(null)
    }
  }

  const handleAutoExtract = async () => {
    if (!botId) return
    if (hasReachedAssetLimit) {
      setExtractResult(`Asset limit reached (${assetCount}/${assetLimit}).`)
      return
    }
    if (extractPages.length > 0 && selectedExtractPages.size === 0) {
      setError('Select at least one page for extraction.')
      return
    }
    setExtracting(true)
    setError(null)
    setExtractResult(null)
    try {
      const payload = {
        page_urls: extractPages.length > 0 ? Array.from(selectedExtractPages) : [],
      }
      const resp = await authedFetch(`/v1/org/bots/${botId}/image-assets/auto-extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as AutoExtractResponse

      if (data.job_id) {
        // Polling effect will pick this up
      } else {
        // Fallback for sync return (shouldn't happen with new backend but safe to keep)
        setExtracting(false)
        if (typeof data.assets_count === 'number') setAssetCount(data.assets_count)
        if (typeof data.assets_limit === 'number') setAssetLimit(data.assets_limit)
        await loadAssets()
      }

    } catch (err) {
      setError((err as Error).message)
      setExtracting(false)
    }
  }

  const toggleSelect = (assetId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(assetId)) next.delete(assetId)
      else next.add(assetId)
      return next
    })
  }

  const handleSelectAll = () => {
    if (selectedIds.size === assets.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(assets.map((a) => a.asset_id)))
    }
  }

  const handleDeleteSelected = async () => {
    if (!botId || selectedIds.size === 0) return
    setBulkDeleting(true)
    setError(null)
    try {
      await Promise.all(
        [...selectedIds].map((id) =>
          authedFetch(`/v1/org/bots/${botId}/image-assets/${id}`, { method: 'DELETE' })
        )
      )
      setSelectedIds(new Set())
      await loadAssets()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBulkDeleting(false)
    }
  }

  return (
    <AnimatedPage>
      <div style={{ maxWidth: 800, margin: '0 auto' }}>
        <SectionHeader
          title={t('botImageAssets.title', 'Image Assets')}
          subtitle={t('botImageAssets.subtitle', 'Products, services, and offerings your AI agent can show in conversations. Auto-extracted from training data or uploaded manually.')}
        />

        {error && (
          <div
            style={{
              padding: '0.75rem 1rem',
              borderRadius: 10,
              background: '#fef2f2',
              color: '#dc2626',
              marginBottom: '1rem',
              fontSize: '0.9rem',
            }}
          >
            {error}
          </div>
        )}

        {!showAdd && (
          <>
            <div
              style={{
                marginBottom: '0.75rem',
                padding: '0.6rem 0.85rem',
                borderRadius: 10,
                background: hasReachedAssetLimit ? '#fff7ed' : '#f8fafc',
                border: hasReachedAssetLimit ? '1px solid #fdba74' : '1px solid #e2e8f0',
                fontSize: '0.9rem',
                color: '#334155',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                flexWrap: 'wrap',
              }}
            >
              <strong>
                {assetCount}/{assetLimit}
              </strong>
              <span>{t('botImageAssets.assetsUsed', 'assets used')}</span>
              <span style={{ color: hasReachedAssetLimit ? '#c2410c' : '#64748b' }}>
                {hasReachedAssetLimit ? t('botImageAssets.limitReached', 'Limit reached') : t('botImageAssets.slotsLeft', '{{slots}} slots left', { slots: assetLimit - assetCount })}
              </span>
            </div>

            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
              <UiButton
                variant="primary"
                onClick={() => setShowAdd(true)}
                disabled={hasReachedAssetLimit}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <Plus size={16} />
                {t('botImageAssets.addAsset', 'Add Asset')}
              </UiButton>
              <UiButton
                variant="secondary"
                onClick={() => void toggleExtractSettings()}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <Sparkles size={16} />
                {showExtractSettings ? t('botImageAssets.hidePageSelection', 'Hide Page Selection') : t('botImageAssets.autoExtractFromPages', 'Auto extract from Pages')}
              </UiButton>
            </div>
          </>
        )}

        {showExtractSettings && (
          <GlassCard style={{ marginBottom: '1rem', padding: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
              <strong style={{ fontSize: '0.95rem' }}>{t('botImageAssets.extractionSourcePages', 'Extraction Source Pages')}</strong>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={() => void loadExtractPages()}
                  disabled={loadingExtractPages}
                  style={{
                    border: '1px solid #cbd5e1',
                    borderRadius: 8,
                    background: '#fff',
                    padding: '0.25rem 0.6rem',
                    fontSize: '0.8rem',
                    color: '#475569',
                    cursor: loadingExtractPages ? 'not-allowed' : 'pointer',
                  }}
                >
                  {loadingExtractPages ? t('botImageAssets.refreshing', 'Refreshing...') : t('botImageAssets.refreshPages', 'Refresh Pages')}
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedExtractPages(new Set(extractPages))}
                  disabled={extractPages.length === 0}
                  style={{
                    border: '1px solid #cbd5e1',
                    borderRadius: 8,
                    background: '#fff',
                    padding: '0.25rem 0.6rem',
                    fontSize: '0.8rem',
                    color: '#475569',
                    cursor: extractPages.length === 0 ? 'not-allowed' : 'pointer',
                  }}
                >
                  {t('botImageAssets.selectAll', 'Select All')}
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedExtractPages(new Set())}
                  disabled={extractPages.length === 0}
                  style={{
                    border: '1px solid #cbd5e1',
                    borderRadius: 8,
                    background: '#fff',
                    padding: '0.25rem 0.6rem',
                    fontSize: '0.8rem',
                    color: '#475569',
                    cursor: extractPages.length === 0 ? 'not-allowed' : 'pointer',
                  }}
                >
                  {t('botImageAssets.clear', 'Clear')}
                </button>
              </div>
            </div>

            <input
              type="text"
              value={extractPageFilter}
              onChange={(e) => setExtractPageFilter(e.target.value)}
              placeholder={t('botImageAssets.filterPages', 'Filter pages...')}
              style={{
                width: '100%',
                marginBottom: '0.75rem',
                border: '1px solid #dbe3ee',
                borderRadius: 8,
                padding: '0.45rem 0.65rem',
                fontSize: '0.85rem',
              }}
            />

            {loadingExtractPages ? (
              <div style={{ color: '#64748b', fontSize: '0.85rem', padding: '0.35rem 0' }}>{t('botImageAssets.loadingPages', 'Loading pages...')}</div>
            ) : extractPages.length === 0 ? (
              <div style={{ color: '#64748b', fontSize: '0.85rem', padding: '0.35rem 0' }}>
                {t('botImageAssets.noCrawledPages', 'No crawled pages found yet. Train URLs first to target specific pages.')}
              </div>
            ) : (
              <div
                style={{
                  border: '1px solid #e2e8f0',
                  borderRadius: 10,
                  maxHeight: 220,
                  overflowY: 'auto',
                  padding: '0.35rem',
                  display: 'grid',
                  gap: '0.35rem',
                }}
              >
                {/* We map extractPages, filtering by extractPageFilter */}
                {extractPages
                  .filter(u => !extractPageFilter || u.toLowerCase().includes(extractPageFilter.toLowerCase()))
                  .map((url) => {
                    const checked = selectedExtractPages.has(url)
                    return (
                      <label
                        key={url}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          padding: '0.35rem 0.5rem',
                          borderRadius: 6,
                          cursor: 'pointer',
                          fontSize: '0.85rem',
                          color: '#334155',
                          background: checked ? 'rgba(240,149,87,0.08)' : 'transparent',
                          border: checked ? '1px solid rgba(240,149,87,0.3)' : '1px solid transparent',
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => {
                            const next = new Set(selectedExtractPages)
                            if (checked) next.delete(url)
                            else next.add(url)
                            setSelectedExtractPages(next)
                          }}
                          style={{ accentColor: 'var(--ui-flow-accent-secondary)' }}
                        />
                        <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {url}
                        </span>
                      </label>
                    )
                  })}
              </div>
            )}

            {/* Extract button at bottom of modal */}
            {extractPages.length > 0 && (
              <div style={{ marginTop: '1rem', display: 'flex', gap: '0.75rem' }}>
                <UiButton
                  variant="primary"
                  onClick={() => void handleAutoExtract()}
                  disabled={extracting || hasReachedAssetLimit || selectedExtractPages.size === 0}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  {extracting ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
                  {extracting ? t('botImageAssets.extractingImages', 'Extracting...') : t('botImageAssets.extractImages', 'Extract Images')}
                </UiButton>
              </div>
            )}
          </GlassCard>
        )}

        {/* Extraction Progress - Persistent */}
        {extracting && extractStats && (
          <GlassCard style={{ marginBottom: '1rem', padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <Loader2 size={16} className="spin" />
              <strong style={{ fontSize: '0.9rem' }}>
                {extractStats.status === 'queued' ? t('botImageAssets.extractionQueued', 'Extraction Queued...') : t('botImageAssets.extractingAssets', 'Extracting Assets...')}
              </strong>
            </div>
            <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, width: '100%', overflow: 'hidden' }}>
              <div style={{
                background: 'var(--ui-flow-accent-secondary)',
                height: '100%',
                width: `${Math.min(100, Math.max(5, (extractStats.assets_discovered > 0 ? (extractStats.assets_downloaded / extractStats.assets_discovered) * 100 : 0)))}%`,
                transition: 'width 0.5s ease-out'
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: '0.8rem', color: '#64748b' }}>
              <span>{t('botImageAssets.found', 'Found: {{count}}', { count: extractStats.assets_discovered })}</span>
              <span>{t('botImageAssets.downloaded', 'Downloaded: {{downloaded}} / {{total}}', { downloaded: extractStats.assets_downloaded, total: extractStats.assets_discovered })}</span>
              <span>{t('botImageAssets.limit', 'Limit: {{limit}}', { limit: extractStats.limit })}</span>
            </div>
          </GlassCard>
        )}



        {/* Extract result message */}
        {
          extractResult && (
            <div
              style={{
                padding: '0.75rem 1rem',
                borderRadius: 10,
                background: '#f0fdf4',
                color: '#16a34a',
                marginBottom: '1rem',
                fontSize: '0.9rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <span>{extractResult}</span>
              <button
                onClick={() => setExtractResult(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#16a34a', padding: 2 }}
              >
                <X size={14} />
              </button>
            </div>
          )
        }

        {/* Bulk-selection toolbar — only shown when assets exist and none in edit mode */}
        {
          !loading && assets.length > 0 && !showAdd && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                marginBottom: '1rem',
                padding: '0.5rem 0.75rem',
                borderRadius: 10,
                background: selectedIds.size > 0 ? 'rgba(239,68,68,0.07)' : 'transparent',
                border: selectedIds.size > 0 ? '1px solid rgba(239,68,68,0.2)' : '1px solid transparent',
                transition: 'all 0.2s',
                flexWrap: 'wrap',
              }}
            >
              {/* Select-all toggle */}
              <button
                onClick={handleSelectAll}
                title={selectedIds.size === assets.length ? 'Deselect all' : 'Select all'}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: '#64748b',
                  fontSize: '0.85rem',
                  padding: '4px 6px',
                  borderRadius: 6,
                }}
              >
                {selectedIds.size === assets.length && assets.length > 0 ? (
                  <CheckSquare size={16} color="#e4587a" />
                ) : (
                  <Square size={16} />
                )}
                {selectedIds.size === assets.length && assets.length > 0 ? t('botImageAssets.deselectAll', 'Deselect All') : t('botImageAssets.selectAll', 'Select All')}
              </button>

              {selectedIds.size > 0 && (
                <>
                  <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
                    {t('botImageAssets.selectedCount', '{{count}} selected', { count: selectedIds.size })}
                  </span>
                  <button
                    onClick={() => void handleDeleteSelected()}
                    disabled={bulkDeleting}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 6,
                      marginLeft: 'auto',
                      padding: '0.45rem 1rem',
                      borderRadius: 8,
                      border: 'none',
                      background: '#ef4444',
                      color: '#fff',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      cursor: bulkDeleting ? 'not-allowed' : 'pointer',
                      opacity: bulkDeleting ? 0.7 : 1,
                      transition: 'opacity 0.15s',
                    }}
                  >
                    {bulkDeleting ? <Loader2 size={15} className="spin" /> : <Trash2 size={15} />}
                    {bulkDeleting ? t('botImageAssets.deleting', 'Deleting...') : t('botImageAssets.deleteSelected', 'Delete Selected ({{count}})', { count: selectedIds.size })}
                  </button>
                  <button
                    onClick={() => setSelectedIds(new Set())}
                    title={t('botImageAssets.clearSelection', 'Clear selection')}
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      color: '#94a3b8',
                      padding: 4,
                    }}
                  >
                    <X size={15} />
                  </button>
                </>
              )}
            </div>
          )
        }

        {/* Add Asset Form */}
        {
          showAdd && (
            <GlassCard style={{ marginBottom: '1.5rem', padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>{t('botImageAssets.newAsset', 'New Asset')}</h3>
                <button
                  onClick={() => {
                    setShowAdd(false)
                    setAddFile(null)
                    setAddPreview(null)
                  }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', padding: 4 }}
                >
                  <X size={18} />
                </button>
              </div>

              <div style={{ display: 'grid', gap: '1rem' }}>
                {/* Image upload */}
                <div>
                  <label
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 8,
                      padding: addPreview ? 0 : '2rem',
                      border: '2px dashed #e2e8f0',
                      borderRadius: 12,
                      cursor: 'pointer',
                      textAlign: 'center',
                      overflow: 'hidden',
                      background: '#fafafa',
                      transition: 'border-color 0.2s',
                    }}
                  >
                    {addPreview ? (
                      <img
                        src={addPreview}
                        alt="Preview"
                        style={{ width: '100%', maxHeight: 220, objectFit: 'cover' }}
                      />
                    ) : (
                      <>
                        <Upload size={28} color="#94a3b8" />
                        <span style={{ fontSize: '0.9rem', color: '#64748b' }}>
                          {t('botImageAssets.uploadImagePrompt', 'Click or drag to upload image')}
                        </span>
                        <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                          {t('botImageAssets.supportedFormats', 'JPEG, PNG, GIF, WebP, SVG')}
                        </span>
                      </>
                    )}
                    <input
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      onChange={(e) =>
                        handleFileChange(
                          e.target.files?.[0] || null,
                          setAddFile,
                          setAddPreview
                        )
                      }
                    />
                  </label>
                </div>

                <GlassField label={t('botImageAssets.nameLabel', 'Name')}>
                  <input
                    type="text"
                    value={addName}
                    onChange={(e) => setAddName(e.target.value)}
                    placeholder={t('botImageAssets.namePlaceholder', 'e.g. Deluxe Ocean Room')}
                  />
                </GlassField>

                <GlassField label={t('botImageAssets.descriptionLabel', 'Description (for AI context)')}>
                  <textarea
                    value={addDescription}
                    onChange={(e) => setAddDescription(e.target.value)}
                    placeholder={t('botImageAssets.descriptionPlaceholder', 'Describe this asset so the AI knows when to show it...')}
                    rows={3}
                    style={{ fontFamily: 'inherit' }}
                  />
                </GlassField>

                <GlassField label={t('botImageAssets.linkUrlLabel', 'Link URL (optional)')}>
                  <input
                    type="url"
                    value={addLinkUrl}
                    onChange={(e) => setAddLinkUrl(e.target.value)}
                    placeholder={t('botImageAssets.linkUrlPlaceholder', 'https://example.com/booking')}
                  />
                </GlassField>

                <GlassField label={t('botImageAssets.keywordsLabel', 'Keywords (optional, comma-separated)')}>
                  <input
                    type="text"
                    value={addKeywords}
                    onChange={(e) => setAddKeywords(e.target.value)}
                    placeholder={t('botImageAssets.keywordsPlaceholder', 'e.g. ocean, room, luxury')}
                  />
                </GlassField>

                <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
                  <UiButton
                    variant="secondary"
                    onClick={() => {
                      setShowAdd(false)
                      setAddFile(null)
                      setAddPreview(null)
                    }}
                  >
                    {t('botImageAssets.cancel', 'Cancel')}
                  </UiButton>
                  <UiButton
                    variant="primary"
                    onClick={() => void handleAdd()}
                    disabled={!addFile || !addName.trim() || saving || hasReachedAssetLimit}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
                  >
                    {saving ? <Loader2 size={16} className="spin" /> : <Check size={16} />}
                    {saving ? t('botImageAssets.uploading', 'Uploading...') : t('botImageAssets.saveAsset', 'Save Asset')}
                  </UiButton>
                </div>
              </div>
            </GlassCard>
          )
        }

        {/* Loading */}
        {
          loading && (
            <div style={{ textAlign: 'center', padding: '3rem 0', color: '#94a3b8' }}>
              <Loader2 size={28} className="spin" />
              <p style={{ marginTop: '0.5rem' }}>{t('botImageAssets.loadingAssets', 'Loading assets...')}</p>
            </div>
          )
        }

        {/* Empty state */}
        {
          !loading && assets.length === 0 && (
            <GlassCard style={{ padding: '3rem', textAlign: 'center' }}>
              <Image size={48} color="#cbd5e1" style={{ marginBottom: '1rem' }} />
              <h3 style={{ margin: '0 0 0.5rem', fontWeight: 600, color: '#334155' }}>
                {t('botImageAssets.noAssetsYet', 'No assets yet')}
              </h3>
              <p style={{ margin: 0, color: '#94a3b8', maxWidth: 440, marginInline: 'auto' }} dangerouslySetInnerHTML={{ __html: t('botImageAssets.noAssetsDescription', 'Click <strong>Auto-extract from URLs</strong> to automatically detect products, services, and offerings from your training data. You can also upload assets manually.') }} />
            </GlassCard>
          )
        }

        {/* Asset list */}
        {
          !loading && assets.length > 0 && (
            <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
              {assets
                .filter((a) => a.image_url && a.image_url.trim() !== '')
                .map((a) => {
                  const isEditing = editingId === a.asset_id
                  const isDeleting = deleting === a.asset_id

                  if (isEditing) {
                    return (
                      <GlassCard key={a.asset_id} style={{ padding: 0, overflow: 'hidden' }}>
                        {/* Image fills top without padding */}
                        {a.image_url ? (
                          <img
                            src={`${API_BASE}${a.image_url}`}
                            alt={a.name}
                            style={{
                              width: '100%',
                              height: 160,
                              objectFit: 'cover',
                            }}
                          />
                        ) : (
                          <div
                            style={{
                              width: '100%',
                              height: 160,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              background: '#f1f5f9',
                              color: '#94a3b8',
                            }}
                          >
                            <Image size={32} />
                          </div>
                        )}

                        {/* Content with padding */}
                        <div style={{ padding: '1rem', display: 'grid', gap: '0.75rem' }}>
                          <label
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                              fontSize: '0.82rem',
                              color: '#64748b',
                              cursor: 'pointer',
                            }}
                          >
                            <Upload size={14} />
                            {t('botImageAssets.replaceImage', 'Replace image')}
                            <input
                              type="file"
                              accept="image/*"
                              style={{ display: 'none' }}
                              onChange={(e) => setEditFile(e.target.files?.[0] || null)}
                            />
                          </label>

                          <GlassField label={t('botImageAssets.nameLabel', 'Name')}>
                            <input
                              type="text"
                              value={editName}
                              onChange={(e) => setEditName(e.target.value)}
                            />
                          </GlassField>

                          <GlassField label={t('botImageAssets.descriptionLabel', 'Description (for AI context)')}>
                            <textarea
                              value={editDescription}
                              onChange={(e) => setEditDescription(e.target.value)}
                              rows={2}
                              style={{ fontFamily: 'inherit' }}
                            />
                          </GlassField>

                          <GlassField label={t('botImageAssets.linkUrlLabel', 'Link URL (optional)')}>
                            <input
                              type="url"
                              value={editLinkUrl}
                              onChange={(e) => setEditLinkUrl(e.target.value)}
                            />
                          </GlassField>

                          <GlassField label={t('botImageAssets.keywordsLabel', 'Keywords (optional, comma-separated)')}>
                            <input
                              type="text"
                              value={editKeywords}
                              onChange={(e) => setEditKeywords(e.target.value)}
                            />
                          </GlassField>

                          <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                            <UiButton variant="secondary" onClick={() => setEditingId(null)}>
                              {t('botImageAssets.cancel', 'Cancel')}
                            </UiButton>
                            <UiButton
                              variant="primary"
                              onClick={() => void handleEdit()}
                              disabled={!editName.trim() || editSaving}
                              style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                            >
                              {editSaving ? <Loader2 size={14} className="spin" /> : <Check size={14} />}
                              {t('botImageAssets.save', 'Save')}
                            </UiButton>
                          </div>
                        </div>
                      </GlassCard>
                    )
                  }

                  const isSelected = selectedIds.has(a.asset_id)

                  return (
                    <GlassCard
                      key={a.asset_id}
                      style={{
                        padding: 0,
                        overflow: 'hidden',
                        opacity: a.is_active ? 1 : 0.5,
                        outline: isSelected ? '2px solid #e4587a' : '2px solid transparent',
                        outlineOffset: -2,
                        transition: 'outline 0.15s',
                        position: 'relative',
                      }}
                    >
                      {/* Checkbox overlay */}
                      <button
                        onClick={() => toggleSelect(a.asset_id)}
                        title={isSelected ? 'Deselect' : 'Select'}
                        style={{
                          position: 'absolute',
                          top: 8,
                          left: 8,
                          zIndex: 10,
                          background: isSelected ? '#e4587a' : 'rgba(255,255,255,0.85)',
                          border: isSelected ? '2px solid #e4587a' : '2px solid #cbd5e1',
                          borderRadius: 5,
                          width: 22,
                          height: 22,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          padding: 0,
                          boxShadow: '0 1px 4px rgba(0,0,0,0.12)',
                          transition: 'all 0.15s',
                        }}
                      >
                        {isSelected && <Check size={13} color="#fff" strokeWidth={3} />}
                      </button>

                      {a.image_url ? (
                        <img
                          src={`${API_BASE}${a.image_url}`}
                          alt={a.name}
                          style={{
                            width: '100%',
                            height: 180,
                            objectFit: 'cover',
                            display: 'block',
                          }}
                        />
                      ) : (
                        <div
                          style={{
                            width: '100%',
                            height: 120,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)',
                            color: '#94a3b8',
                          }}
                        >
                          <Image size={36} />
                        </div>
                      )}
                      <div style={{ padding: '0.75rem 1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div>
                            <h4 style={{ margin: '0 0 0.25rem', fontSize: '1rem', fontWeight: 600 }}>
                              {a.name}
                            </h4>
                            {a.description && (
                              <p
                                style={{
                                  margin: 0,
                                  fontSize: '0.82rem',
                                  color: '#64748b',
                                  lineHeight: 1.4,
                                  maxHeight: '2.8em',
                                  overflow: 'hidden',
                                }}
                              >
                                {a.description}
                              </p>
                            )}
                            {a.keywords && a.keywords.length > 0 && (
                              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 6 }}>
                                {a.keywords.map((kw) => (
                                  <span
                                    key={kw}
                                    style={{
                                      fontSize: '0.72rem',
                                      padding: '2px 8px',
                                      borderRadius: 999,
                                      background: '#f1f5f9',
                                      color: '#64748b',
                                    }}
                                  >
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                          <div style={{ display: 'flex', gap: 4, flexShrink: 0 }}>
                            <button
                              onClick={() => startEdit(a)}
                              title="Edit"
                              style={{
                                background: 'none',
                                border: 'none',
                                cursor: 'pointer',
                                color: '#94a3b8',
                                padding: 4,
                              }}
                            >
                              <Pencil size={16} />
                            </button>
                            <button
                              onClick={() => void handleDelete(a.asset_id)}
                              title="Delete"
                              disabled={isDeleting}
                              style={{
                                background: 'none',
                                border: 'none',
                                cursor: 'pointer',
                                color: '#ef4444',
                                padding: 4,
                              }}
                            >
                              {isDeleting ? <Loader2 size={16} className="spin" /> : <Trash2 size={16} />}
                            </button>
                          </div>
                        </div>
                        {a.link_url && (
                          <a
                            href={a.link_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{
                              display: 'inline-block',
                              marginTop: 6,
                              fontSize: '0.82rem',
                              color: 'var(--app-accent, #e4587a)',
                            }}
                          >
                            {a.link_url.replace(/^https?:\/\//, '').slice(0, 40)}
                            {a.link_url.replace(/^https?:\/\//, '').length > 40 ? '...' : ''}
                          </a>
                        )}
                      </div>
                    </GlassCard>
                  )
                })}
            </div>
          )
        }
      </div >
    </AnimatedPage >
  )
}
