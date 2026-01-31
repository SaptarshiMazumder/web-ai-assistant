import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function AddSourcePage() {
  const { botId } = useParams()
  const navigate = useNavigate()
  const { selectedBot, createSource, startCrawlForSource, loadSources, loadJobs, loading, error, setError } = useDashboardData()

  const [sourceType, setSourceType] = useState<'url' | 'drive' | 'docs'>('url')
  const [url, setUrl] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!botId || !selectedBot) return
    if (sourceType === 'url') {
      const u = url.trim()
      if (!u) {
        setLocalError('Enter a URL')
        return
      }
      setSubmitting(true)
      setLocalError(null)
      setError(null)
      try {
        const source = await createSource(selectedBot.bot_id, 'url', { url: u }, displayName.trim() || null)
        if (!source) {
          setLocalError('Failed to add source')
          return
        }
        await startCrawlForSource(selectedBot.bot_id, source.source_id)
        await loadSources(selectedBot.bot_id)
        await loadJobs(selectedBot.bot_id)
        navigate('../../knowledge', { relative: 'path' })
      } catch (err) {
        setLocalError((err as Error).message)
      } finally {
        setSubmitting(false)
      }
    } else {
      setLocalError('Drive and Google Docs coming soon. Use URL for now.')
    }
  }

  const handleBack = () => {
    navigate('../../knowledge', { relative: 'path' })
  }

  if (!botId || !selectedBot || selectedBot.bot_id !== botId) {
    return <div className="empty-panel">Loading…</div>
  }

  return (
    <div className="card" style={{ maxWidth: '560px', marginTop: '1rem' }}>
      <div className="row" style={{ alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
        <button type="button" className="ghost" onClick={handleBack} aria-label="Back to Knowledge">
          <ArrowLeft size={20} strokeWidth={2} />
        </button>
        <div>
          <h2 className="card-title" style={{ margin: 0 }}>Add source</h2>
          <p className="card-subtitle" style={{ margin: '0.25rem 0 0' }}>
            Add a URL, Drive folder, or Google Doc. We’ll crawl it and add it to this bot’s RAG.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="stack" style={{ gap: '1.25rem' }}>
        <div>
          <label className="design-form-label">Source type</label>
          <div className="design-form-radio-group" style={{ marginTop: '0.5rem' }}>
            <label className="design-form-radio-card">
              <input
                type="radio"
                name="sourceType"
                value="url"
                checked={sourceType === 'url'}
                onChange={() => setSourceType('url')}
              />
              <span>URL</span>
            </label>
            <label className="design-form-radio-card">
              <input
                type="radio"
                name="sourceType"
                value="drive"
                checked={sourceType === 'drive'}
                onChange={() => setSourceType('drive')}
              />
              <span>Google Drive</span>
            </label>
            <label className="design-form-radio-card">
              <input
                type="radio"
                name="sourceType"
                value="docs"
                checked={sourceType === 'docs'}
                onChange={() => setSourceType('docs')}
              />
              <span>Google Docs</span>
            </label>
          </div>
        </div>

        {sourceType === 'url' && (
          <>
            <div>
              <label className="design-form-label" htmlFor="add-source-url">URL</label>
              <input
                id="add-source-url"
                type="url"
                className="design-form-input"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                required
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label className="design-form-label" htmlFor="add-source-display">Display name (optional)</label>
              <input
                id="add-source-display"
                type="text"
                className="design-form-input"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="e.g. Homepage"
                style={{ width: '100%' }}
              />
            </div>
          </>
        )}

        {(sourceType === 'drive' || sourceType === 'docs') && (
          <div className="muted" style={{ padding: '1rem', background: 'var(--surface)', borderRadius: '10px' }}>
            Google Drive and Google Docs integration is coming soon. Use a URL for now.
          </div>
        )}

        {(localError || error) && (
          <div className="alert error">
            {localError || error}
          </div>
        )}

        <div className="row" style={{ gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            type="submit"
            className="primary"
            disabled={loading || submitting || (sourceType === 'url' && !url.trim())}
          >
            {submitting ? 'Adding & training…' : 'Add & train'}
          </button>
          <button type="button" className="ghost" onClick={handleBack}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}
