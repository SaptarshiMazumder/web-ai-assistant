import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, MousePointerClick, Printer, UploadCloud } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { FileDropzone } from '../../components/FileDropzone'

export default function AddSourcePage() {
  const { botId } = useParams()
  const navigate = useNavigate()
  const {
    selectedBot,
    selectedBotWidgetConfig,
    createSource,
    uploadPdfSources,
    startCrawlForSource,
    loadSources,
    loadJobs,
    loading,
    error,
    setError,
  } = useDashboardData()

  const defaultHosting = useMemo<'own' | 'shared' | null>(() => {
    const cfg = selectedBotWidgetConfig
    if (!cfg || typeof cfg !== 'object') return null
    const h = (cfg as Record<string, unknown>).contentHosting
    return h === 'own' || h === 'shared' ? h : null
  }, [selectedBotWidgetConfig])

  const [contentHosting, setContentHosting] = useState<'own' | 'shared' | null>(defaultHosting)
  const [url, setUrl] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [language, setLanguage] = useState<'auto' | 'ja'>('auto')
  const [pdfFiles, setPdfFiles] = useState<File[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  useEffect(() => {
    // Only set a default once; don't override an explicit user choice.
    setContentHosting((prev) => prev ?? defaultHosting)
  }, [defaultHosting])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!botId || !selectedBot) return
    if (!contentHosting) {
      setLocalError('Pick one option to continue.')
      return
    }
    if (contentHosting === 'own') {
      const u = url.trim()
      if (!u) {
        setLocalError('Enter a website link')
        return
      }
      setSubmitting(true)
      setLocalError(null)
      setError(null)
      try {
        const config: Record<string, unknown> = { url: u }
        if (language && language !== 'auto') config.language = language
        const source = await createSource(selectedBot.bot_id, 'url', config, displayName.trim() || null)
        if (!source) {
          setLocalError('Could not add source')
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
    } else if (contentHosting === 'shared') {
      if (!pdfFiles.length) {
        setLocalError('Add at least one PDF')
        return
      }
      setSubmitting(true)
      setLocalError(null)
      setError(null)
      try {
        const resp = await uploadPdfSources(selectedBot.bot_id, pdfFiles, displayName.trim() || null)
        if (!resp || !resp.items?.length) {
          setLocalError('Could not upload PDF')
          return
        }
        await loadSources(selectedBot.bot_id)
        await loadJobs(selectedBot.bot_id)
        navigate('../../knowledge', { relative: 'path' })
      } catch (err) {
        setLocalError((err as Error).message)
      } finally {
        setSubmitting(false)
      }
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
          <h2 className="card-title" style={{ margin: 0 }}>Add sources</h2>
          <p className="card-subtitle" style={{ margin: '0.25rem 0 0' }}>
            Add website pages or PDFs so your AI agent can learn your business info.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="stack" style={{ gap: '1.25rem' }}>
        <div>
          <div className="card-title">Where is your website?</div>
          <div className="card-subtitle">Pick what best describes your business.</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', marginTop: '0.75rem' }}>
            <button
              type="button"
              onClick={() => {
                setContentHosting('own')
                setLocalError(null)
              }}
              style={{
                textAlign: 'left',
                padding: '1.25rem',
                borderRadius: '16px',
                border: `2px solid ${contentHosting === 'own' ? '#6366f1' : '#e2e8f0'}`,
                background: contentHosting === 'own' ? 'rgba(99,102,241,0.06)' : '#fff',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '1.05rem', color: '#0f172a' }}>
                Yes — I have my own website
              </div>
            </button>

            <button
              type="button"
              onClick={() => {
                setContentHosting('shared')
                setLocalError(null)
              }}
              style={{
                textAlign: 'left',
                padding: '1.25rem',
                borderRadius: '16px',
                border: `2px solid ${contentHosting === 'shared' ? '#6366f1' : '#e2e8f0'}`,
                background: contentHosting === 'shared' ? 'rgba(99,102,241,0.06)' : '#fff',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '1.05rem', color: '#0f172a' }}>
                Not really — I use a website service
              </div>
            </button>
          </div>
        </div>

        {contentHosting === 'own' && (
          <>
            <div>
              <label className="design-form-label" htmlFor="add-source-url">Website link</label>
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
            <div>
              <label className="design-form-label" htmlFor="add-source-language">Language (optional)</label>
              <select
                id="add-source-language"
                className="design-form-input"
                value={language}
                onChange={(e) => setLanguage(e.target.value as 'auto' | 'ja')}
                style={{ width: '100%' }}
              >
                <option value="auto">Auto (detect)</option>
                <option value="ja">Japanese</option>
              </select>
              <div className="muted" style={{ marginTop: '0.35rem', fontSize: '0.85rem' }}>
                If text appears garbled, set Japanese to force charset-based decoding.
              </div>
            </div>
          </>
        )}

        {contentHosting === 'shared' && (
          <>
            <div>
              <div className="card-title">Website contents (PDFs)</div>
              <div className="card-subtitle">
                Save your important business pages as PDFs, then upload them here.
              </div>
              <div className="muted" style={{ marginTop: '8px' }}>
                Do this for <b>every page</b> that has helpful business info (services, prices, hours, booking, contact, location, FAQs).
              </div>
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
                  Drop the saved PDF below. Your agent will learn from what’s inside.
                </div>
              </div>
            </div>

            <FileDropzone
              label="PDF files"
              helperText="Drag & drop PDFs here."
              files={pdfFiles}
              setFiles={setPdfFiles}
              accept="application/pdf"
              multiple
              maxFiles={20}
            />
            <div>
              <label className="design-form-label" htmlFor="add-source-display-pdf">Display name (optional)</label>
              <input
                id="add-source-display-pdf"
                type="text"
                className="design-form-input"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="e.g. Company brochure"
                style={{ width: '100%' }}
              />
            </div>
          </>
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
            disabled={
              loading ||
              submitting ||
              !contentHosting ||
              (contentHosting === 'own' && !url.trim()) ||
              (contentHosting === 'shared' && pdfFiles.length === 0)
            }
          >
            {submitting ? 'Adding…' : 'Add sources'}
          </button>
          <button type="button" className="ghost" onClick={handleBack}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}
