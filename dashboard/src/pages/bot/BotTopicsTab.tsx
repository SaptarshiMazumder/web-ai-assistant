import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { RefreshCw, Trash2, X, Check, Plus, FolderPlus, ExternalLink } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useDashboardData, type ExtractedTopic } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'
import { useDialog } from '../../contexts/DialogContext'

const CATEGORY_COLOR_STORAGE_PREFIX = 'webai.topicCategoryColors.'

const CATEGORY_OPTIONS = [
  { value: 'product', label: 'Product' },
  { value: 'pricing', label: 'Pricing' },
  { value: 'shipping', label: 'Shipping' },
  { value: 'support', label: 'Support' },
  { value: 'policy', label: 'Policy' },
  { value: 'location', label: 'Location' },
  { value: 'hours', label: 'Hours' },
  { value: 'contact', label: 'Contact' },
  { value: 'faq', label: 'FAQ' },
  { value: 'event', label: 'Event' },
  { value: 'other', label: 'Other' },
]

const DYNAMIC_CATEGORY_PALETTE = [
  '#0ea5e9',
  '#22c55e',
  '#f97316',
  '#a855f7',
  '#14b8a6',
  '#e11d48',
  '#eab308',
  '#6366f1',
  '#84cc16',
  '#64748b',
]

function hashCategoryToColor(category: string): string {
  let hash = 0
  for (let i = 0; i < category.length; i += 1) {
    hash = (hash * 31 + category.charCodeAt(i)) >>> 0
  }
  return DYNAMIC_CATEGORY_PALETTE[hash % DYNAMIC_CATEGORY_PALETTE.length]
}

function getCategoryColor(category: string | null | undefined): string {
  const colors: Record<string, string> = {
    product: '#6366f1',
    pricing: '#22c55e',
    shipping: '#f59e0b',
    support: '#3b82f6',
    policy: '#8b5cf6',
    location: '#ec4899',
    hours: '#14b8a6',
    contact: '#f97316',
    faq: '#06b6d4',
    event: '#a855f7',
    other: '#64748b',
  }
  const key = (category || 'other').toLowerCase()
  return colors[key] || hashCategoryToColor(key)
}

function normalizeHexColor(value: string): string | null {
  if (/^#[0-9a-fA-F]{6}$/.test(value)) return value.toLowerCase()
  return null
}

function categoryLabel(category: string): string {
  if (!category) return 'Other'
  const opt = CATEGORY_OPTIONS.find((o) => o.value === category.toLowerCase())
  return opt ? opt.label : category.charAt(0).toUpperCase() + category.slice(1).toLowerCase()
}

export default function BotTopicsTab() {
  const { botId } = useParams()
  const { t } = useTranslation()
  const dialog = useDialog()
  const {
    selectedBot,
    loading,
    getExtractedTopics,
    extractTopics,
    updateExtractedTopic,
    createExtractedTopic,
    deleteExtractedTopic,
  } = useDashboardData()

  const [topics, setTopics] = useState<ExtractedTopic[]>([])
  const [loadingTopics, setLoadingTopics] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showInactive, setShowInactive] = useState(false)
  /** Empty category boxes (no topics yet). Persisted only in session; gone on refresh until user adds a topic. */
  const [emptyCategories, setEmptyCategories] = useState<string[]>([])
  const [categoryColors, setCategoryColors] = useState<Record<string, string>>({})

  const loadTopics = useCallback(async () => {
    if (!botId) return
    setLoadingTopics(true)
    setError(null)
    try {
      const data = await getExtractedTopics(botId, !showInactive)
      setTopics(data?.topics || [])
    } catch (err) {
      setError('Failed to load topics')
      console.error(err)
    } finally {
      setLoadingTopics(false)
    }
  }, [botId, getExtractedTopics, showInactive])

  useEffect(() => {
    void loadTopics()
  }, [loadTopics])

  useEffect(() => {
    if (!botId) return
    const storageKey = `${CATEGORY_COLOR_STORAGE_PREFIX}${botId}`
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) {
      setCategoryColors({})
      return
    }
    try {
      const parsed = JSON.parse(raw) as Record<string, string>
      const cleaned: Record<string, string> = {}
      for (const [key, value] of Object.entries(parsed || {})) {
        const normalized = normalizeHexColor(value)
        if (normalized) cleaned[key.toLowerCase()] = normalized
      }
      setCategoryColors(cleaned)
    } catch {
      setCategoryColors({})
    }
  }, [botId])

  useEffect(() => {
    if (!botId) return
    const storageKey = `${CATEGORY_COLOR_STORAGE_PREFIX}${botId}`
    window.localStorage.setItem(storageKey, JSON.stringify(categoryColors))
  }, [botId, categoryColors])

  const handleExtract = async (clearExisting: boolean = false) => {
    if (!botId || extracting) return
    setExtracting(true)
    setError(null)
    try {
      await extractTopics(botId, clearExisting)
      await loadTopics()
    } catch (err) {
      setError('Failed to extract topics')
      console.error(err)
    } finally {
      setExtracting(false)
    }
  }

  const handleAddBox = async () => {
    const name = await dialog.prompt({
      title: t('botTopics.addCategoryTitle', 'Add category'),
      label: t('botTopics.categoryNameLabel', 'Category name'),
      placeholder: t('botTopics.categoryNamePlaceholder', 'e.g. Pricing, Location'),
      confirmLabel: t('common.add', 'Add'),
      cancelLabel: t('common.cancel', 'Cancel'),
      required: true,
    })
    const trimmed = name?.trim()
    if (!trimmed) return
    setEmptyCategories((prev) => (prev.includes(trimmed) ? prev : [...prev, trimmed]))
  }

  const handleAddTopic = async (category: string) => {
    if (!botId) return
    const name = await dialog.prompt({
      title: t('botTopics.addTopicTitle', 'Add topic'),
      label: t('botTopics.topicNameLabel', 'Topic name'),
      placeholder: t('botTopics.topicNamePlaceholder', 'Topic name'),
      confirmLabel: t('common.add', 'Add'),
      cancelLabel: t('common.cancel', 'Cancel'),
      required: true,
    })
    const trimmedName = name?.trim()
    if (!trimmedName) return
    setError(null)
    try {
      const created = await createExtractedTopic(botId, trimmedName, category)
      if (created) {
        setEmptyCategories((prev) => prev.filter((c) => c !== category))
        await loadTopics()
      } else {
        setError('Failed to add topic (maybe duplicate?)')
      }
    } catch (err) {
      setError('Failed to add topic')
      console.error(err)
    }
  }

  const handleCategoryColorChange = (category: string, color: string) => {
    const normalized = normalizeHexColor(color)
    if (!normalized) return
    const key = (category || 'other').toLowerCase()
    setCategoryColors((prev) => ({ ...prev, [key]: normalized }))
  }

  const handleToggleActive = async (topic: ExtractedTopic) => {
    if (!botId) return
    try {
      await updateExtractedTopic(botId, topic.topic_id, { is_active: !topic.is_active })
      await loadTopics()
    } catch (err) {
      setError('Failed to update topic')
      console.error(err)
    }
  }

  const handleDelete = async (topicId: string) => {
    if (!botId) return
    const confirmed = await dialog.confirm({
      title: t('botTopics.deleteTopicConfirm', 'Are you sure you want to delete this topic?'),
      confirmLabel: t('common.delete', 'Delete'),
      cancelLabel: t('common.cancel', 'Cancel'),
      tone: 'danger',
    })
    if (!confirmed) return
    try {
      await deleteExtractedTopic(botId, topicId)
      await loadTopics()
    } catch (err) {
      setError('Failed to delete topic')
      console.error(err)
    }
  }

  if (!botId) {
    return <div className="empty-panel">Select a bot to manage topics.</div>
  }

  if (loading && !selectedBot) {
    return <div className="empty-panel">Loading...</div>
  }

  if (selectedBot?.bot_id !== botId) {
    return <div className="empty-panel">Loading...</div>
  }

  // Group topics by category (lowercase for grouping)
  const byCategory = new Map<string, ExtractedTopic[]>()
  for (const t of topics) {
    const cat = (t.category || 'other').toLowerCase()
    if (!byCategory.has(cat)) byCategory.set(cat, [])
    byCategory.get(cat)!.push(t)
  }
  // All category names: from data + empty boxes
  const allCategoryNames = Array.from(
    new Set([...byCategory.keys(), ...emptyCategories.map((c) => c.toLowerCase())])
  ).sort((a, b) => a.localeCompare(b))

  const activeCount = topics.filter((t) => t.is_active).length
  const inactiveCount = topics.filter((t) => !t.is_active).length

  return (
    <AnimatedPage className="flow-panel-body">
      <SectionHeader
        eyebrow="Taxonomy"
        title="Topic clusters"
        subtitle="Curate grouped intent themes and keep your bot routing precise."
      />
      <GlassCard>
        <div className="topics-header">
          <div>
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FolderPlus size={16} style={{ color: 'var(--ui-flow-accent)' }} />
              Manage Topics
            </div>
            <p className="card-subtitle" style={{ marginTop: '0.25rem' }}>
              Topics are grouped by category. Add categories, and add topics inside each category box.
            </p>
          </div>
          <div className="topics-actions">
            <UiButton
              variant="secondary"
              onClick={handleAddBox}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <FolderPlus size={16} />
              Add Category
            </UiButton>
            <UiButton
              variant="secondary"
              onClick={() => void handleExtract(false)}
              disabled={extracting}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <RefreshCw size={16} className={extracting ? 'spin' : ''} />
              {extracting ? 'Extracting...' : 'Re-extract Topics'}
            </UiButton>
          </div>
        </div>

        {error && (
          <div className="topics-error">
            {error}
            <UiButton variant="ghost" onClick={() => setError(null)} style={{ padding: '0.3rem' }}>
              <X size={14} />
            </UiButton>
          </div>
        )}

        <div className="topics-filter">
          <label className="topics-filter-toggle">
            <input
              type="checkbox"
              checked={showInactive}
              onChange={(e) => setShowInactive(e.target.checked)}
            />
            <span>Show inactive topics</span>
          </label>
          <span className="topics-count">
            {activeCount} active{showInactive && `, ${inactiveCount} inactive`}
          </span>
        </div>

        {loadingTopics ? (
          <div className="topics-loading">Loading topics...</div>
        ) : allCategoryNames.length === 0 ? (
          <div className="topics-empty">
            <p>No topics yet.</p>
            <p className="muted">
              Click &quot;Add box&quot; to create a category, then add topics inside it.
              Or use &quot;Re-extract Topics&quot; to pull topics from your website.
            </p>
          </div>
        ) : (
          <div className="topics-boxes">
            {allCategoryNames.map((catKey) => {
              const boxTopics = byCategory.get(catKey) || []
              const isEmpty = boxTopics.length === 0
              const displayName = categoryLabel(catKey)
              const color = categoryColors[catKey] || getCategoryColor(catKey)

              return (
                <div key={catKey} className="topic-box">
                  <div className="topic-box-header">
                    <div className="topic-box-heading">
                      <label className="topic-box-color" title={`Change color for ${displayName}`}>
                        <input
                          type="color"
                          className="topic-box-color-input"
                          value={color}
                          onChange={(event) => handleCategoryColorChange(catKey, event.target.value)}
                          aria-label={`Color for ${displayName}`}
                        />
                        <span className="topic-box-color-swatch" style={{ backgroundColor: color }} />
                      </label>
                      <span className="topic-box-title">{displayName}</span>
                    </div>
                    <button
                      type="button"
                      className="topic-box-add"
                      onClick={() => void handleAddTopic(catKey)}
                      title={`Add topic to ${displayName}`}
                    >
                      <Plus size={14} />
                      Add topic
                    </button>
                  </div>
                  <div className="topic-box-body">
                    {isEmpty ? (
                      <p className="topic-box-empty">No topics yet. Click &quot;Add topic&quot; above.</p>
                    ) : (
                      <div className="topic-box-badges">
                        {boxTopics.map((topic) => (
                          <div
                            key={topic.topic_id}
                            className={`topic-bubble-wrap ${!topic.is_active ? 'topic-bubble-wrap--inactive' : ''}`}
                          >
                            <span
                              className="topic-badge"
                              style={{
                                backgroundColor: color + '20',
                                color,
                              }}
                            >
                              {topic.topic}
                              {topic.origin && topic.origin !== 'extracted' && (
                                <span className="topic-origin-badge topic-origin-badge--inline">
                                  {topic.origin === 'url_bank' ? 'link' : topic.origin}
                                </span>
                              )}
                            </span>
                            <div className="topic-bubble-actions">
                              {topic.source_url && (
                                <a
                                  href={topic.source_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="topic-source-link"
                                  title="View source"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <ExternalLink size={11} />
                                </a>
                              )}
                              <button
                                type="button"
                                className={`topic-toggle ${topic.is_active ? 'active' : ''}`}
                                onClick={() => void handleToggleActive(topic)}
                                title={topic.is_active ? 'Deactivate' : 'Activate'}
                              >
                                {topic.is_active ? <Check size={12} /> : <X size={12} />}
                              </button>
                              <button
                                type="button"
                                className="topic-delete"
                                onClick={() => void handleDelete(topic.topic_id)}
                                title="Delete"
                              >
                                <Trash2 size={12} />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </GlassCard>
    </AnimatedPage>
  )
}
