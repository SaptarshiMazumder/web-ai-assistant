import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { RefreshCw, Trash2, X, Check, Plus, FolderPlus } from 'lucide-react'
import { useDashboardData, type ExtractedTopic } from '../../hooks/useDashboardData'

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
  return colors[key] || colors.other
}

function categoryLabel(category: string): string {
  if (!category) return 'Other'
  const opt = CATEGORY_OPTIONS.find((o) => o.value === category.toLowerCase())
  return opt ? opt.label : category.charAt(0).toUpperCase() + category.slice(1).toLowerCase()
}

export default function BotTopicsTab() {
  const { botId } = useParams()
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

  const handleAddBox = () => {
    const name = window.prompt('Category name for the new box (e.g. Pricing, Location):')
    if (!name?.trim()) return
    const trimmed = name.trim()
    setEmptyCategories((prev) => (prev.includes(trimmed) ? prev : [...prev, trimmed]))
  }

  const handleAddTopic = async (category: string) => {
    if (!botId) return
    const name = window.prompt('Topic name:')
    if (!name?.trim()) return
    setError(null)
    try {
      const created = await createExtractedTopic(botId, name.trim(), category)
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
    if (!botId || !window.confirm('Are you sure you want to delete this topic?')) return
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
    <div className="flow-panel-body">
      <section className="card">
        <div className="topics-header">
          <div>
            <div className="card-title">Extracted Topics</div>
            <p className="card-subtitle" style={{ marginTop: '0.25rem' }}>
              Topics are grouped by category. Add boxes (categories) and add topics inside each box.
            </p>
          </div>
          <div className="topics-actions">
            <button
              type="button"
              className="secondary"
              onClick={handleAddBox}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <FolderPlus size={16} />
              Add box
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => void handleExtract(false)}
              disabled={extracting}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <RefreshCw size={16} className={extracting ? 'spin' : ''} />
              {extracting ? 'Extracting...' : 'Re-extract Topics'}
            </button>
          </div>
        </div>

        {error && (
          <div className="topics-error">
            {error}
            <button type="button" className="ghost" onClick={() => setError(null)}>
              <X size={14} />
            </button>
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
              const color = getCategoryColor(catKey)

              return (
                <div key={catKey} className="topic-box">
                  <div className="topic-box-header" style={{ borderLeftColor: color }}>
                    <span className="topic-box-title">{displayName}</span>
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
                            </span>
                            <div className="topic-bubble-actions">
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
      </section>
    </div>
  )
}
