import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import {
  Plus,
  Trash2,
  Loader2,
  Image,
  Pencil,
  X,
  Check,
  Upload,
} from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton, GlassCard, GlassField } from '../../components/ui'

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

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

export default function BotBusinessAssetsTab() {
  const { botId } = useParams()
  const { getAccessTokenSilently } = useAuth0()

  const [assets, setAssets] = useState<AssetRecord[]>([])
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
      const resp = await authedFetch(`/v1/org/bots/${botId}/assets`)
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as { assets: AssetRecord[] }
      setAssets(data.assets || [])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [botId, authedFetch])

  useEffect(() => {
    void loadAssets()
  }, [loadAssets])

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
    setSaving(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', addFile)
      form.append('name', addName.trim())
      form.append('description', addDescription.trim())
      if (addLinkUrl.trim()) form.append('link_url', addLinkUrl.trim())
      if (addKeywords.trim()) form.append('keywords', addKeywords.trim())

      const resp = await authedFetch(`/v1/org/bots/${botId}/assets`, {
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

      const resp = await authedFetch(`/v1/org/bots/${botId}/assets/${editingId}`, {
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
      const resp = await authedFetch(`/v1/org/bots/${botId}/assets/${assetId}`, {
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

  return (
    <AnimatedPage>
      <div style={{ maxWidth: 800, margin: '0 auto' }}>
        <SectionHeader
          title="Business Assets"
          subtitle="Upload images with titles and links that your AI agent can show in chat conversations."
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

        {/* Add Asset Button */}
        {!showAdd && (
          <UiButton
            variant="primary"
            onClick={() => setShowAdd(true)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              marginBottom: '1.5rem',
            }}
          >
            <Plus size={16} />
            Add Asset
          </UiButton>
        )}

        {/* Add Asset Form */}
        {showAdd && (
          <GlassCard style={{ marginBottom: '1.5rem', padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>New Asset</h3>
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
                        Click or drag to upload image
                      </span>
                      <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                        JPEG, PNG, GIF, WebP, SVG
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

              <GlassField label="Name">
                <input
                  type="text"
                  value={addName}
                  onChange={(e) => setAddName(e.target.value)}
                  placeholder="e.g. Deluxe Ocean Room"
                />
              </GlassField>

              <GlassField label="Description (for AI context)">
                <textarea
                  value={addDescription}
                  onChange={(e) => setAddDescription(e.target.value)}
                  placeholder="Describe this asset so the AI knows when to show it..."
                  rows={3}
                  style={{ fontFamily: 'inherit' }}
                />
              </GlassField>

              <GlassField label="Link URL (optional)">
                <input
                  type="url"
                  value={addLinkUrl}
                  onChange={(e) => setAddLinkUrl(e.target.value)}
                  placeholder="https://example.com/booking"
                />
              </GlassField>

              <GlassField label="Keywords (optional, comma-separated)">
                <input
                  type="text"
                  value={addKeywords}
                  onChange={(e) => setAddKeywords(e.target.value)}
                  placeholder="e.g. ocean, room, luxury"
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
                  Cancel
                </UiButton>
                <UiButton
                  variant="primary"
                  onClick={() => void handleAdd()}
                  disabled={!addFile || !addName.trim() || saving}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
                >
                  {saving ? <Loader2 size={16} className="spin" /> : <Check size={16} />}
                  {saving ? 'Uploading...' : 'Save Asset'}
                </UiButton>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '3rem 0', color: '#94a3b8' }}>
            <Loader2 size={28} className="spin" />
            <p style={{ marginTop: '0.5rem' }}>Loading assets...</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && assets.length === 0 && (
          <GlassCard style={{ padding: '3rem', textAlign: 'center' }}>
            <Image size={48} color="#cbd5e1" style={{ marginBottom: '1rem' }} />
            <h3 style={{ margin: '0 0 0.5rem', fontWeight: 600, color: '#334155' }}>
              No assets yet
            </h3>
            <p style={{ margin: 0, color: '#94a3b8', maxWidth: 400, marginInline: 'auto' }}>
              Upload product images, menus, room photos, or any visual assets. Your AI agent will
              automatically show them in conversations when relevant.
            </p>
          </GlassCard>
        )}

        {/* Asset list */}
        {!loading && assets.length > 0 && (
          <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
            {assets.map((a) => {
              const isEditing = editingId === a.asset_id
              const isDeleting = deleting === a.asset_id

              if (isEditing) {
                return (
                  <GlassCard key={a.asset_id} style={{ padding: '1rem' }}>
                    <div style={{ display: 'grid', gap: '0.75rem' }}>
                      {/* Image preview */}
                      <img
                        src={`${API_BASE}${a.image_url}`}
                        alt={a.name}
                        style={{
                          width: '100%',
                          height: 160,
                          objectFit: 'cover',
                          borderRadius: 8,
                        }}
                      />
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
                        Replace image
                        <input
                          type="file"
                          accept="image/*"
                          style={{ display: 'none' }}
                          onChange={(e) => setEditFile(e.target.files?.[0] || null)}
                        />
                      </label>

                      <GlassField label="Name">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                        />
                      </GlassField>

                      <GlassField label="Description">
                        <textarea
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          rows={2}
                          style={{ fontFamily: 'inherit' }}
                        />
                      </GlassField>

                      <GlassField label="Link URL">
                        <input
                          type="url"
                          value={editLinkUrl}
                          onChange={(e) => setEditLinkUrl(e.target.value)}
                        />
                      </GlassField>

                      <GlassField label="Keywords">
                        <input
                          type="text"
                          value={editKeywords}
                          onChange={(e) => setEditKeywords(e.target.value)}
                        />
                      </GlassField>

                      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                        <UiButton variant="secondary" onClick={() => setEditingId(null)}>
                          Cancel
                        </UiButton>
                        <UiButton
                          variant="primary"
                          onClick={() => void handleEdit()}
                          disabled={!editName.trim() || editSaving}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                        >
                          {editSaving ? <Loader2 size={14} className="spin" /> : <Check size={14} />}
                          Save
                        </UiButton>
                      </div>
                    </div>
                  </GlassCard>
                )
              }

              return (
                <GlassCard
                  key={a.asset_id}
                  style={{
                    padding: 0,
                    overflow: 'hidden',
                    opacity: a.is_active ? 1 : 0.5,
                  }}
                >
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
        )}
      </div>
    </AnimatedPage>
  )
}
