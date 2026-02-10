import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Check, Copy, PartyPopper } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotEmbedPage() {
  const navigate = useNavigate()
  const { buildEmbedSnippet, copySnippet } = useDashboardData()
  const { step3, flow } = useCreateBotFlow()
  const { botId } = step3
  const [copied, setCopied] = useState(false)

  const snippet = useMemo(() => buildEmbedSnippet(), [buildEmbedSnippet])

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  const handleCopy = async () => {
    await copySnippet(snippet)
    setCopied(true)
    setTimeout(() => setCopied(false), 2500)
  }

  const handleFinish = () => {
    if (botId) {
      navigate(`/bots/${botId}/overview`)
    } else {
      navigate('/bots')
    }
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Add the chat to your website</div>
        <div className="card-subtitle">
          Copy this code snippet and share it with the person who manages your website. They'll paste it before the closing <code style={{
            background: 'var(--flow-accent-soft)',
            padding: '0.15rem 0.4rem',
            borderRadius: '4px',
            fontSize: '0.85rem',
            color: 'var(--flow-accent)',
          }}>&lt;/body&gt;</code> tag.
        </div>
      </div>

      {/* Snippet block */}
      <div style={{ position: 'relative' }}>
        <pre className="snippet" style={{ margin: 0 }}>
          {snippet}
        </pre>
        <button
          type="button"
          onClick={() => void handleCopy()}
          disabled={!snippet}
          style={{
            position: 'absolute',
            top: '0.75rem',
            right: '0.75rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.75rem',
            borderRadius: '8px',
            border: '1px solid rgba(255,255,255,0.12)',
            background: copied ? 'rgba(34, 197, 94, 0.2)' : 'rgba(255,255,255,0.08)',
            color: copied ? '#4ade80' : '#a5a3c0',
            fontSize: '0.8rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
        >
          {copied ? (
            <>
              <Check size={14} strokeWidth={2.5} />
              Copied
            </>
          ) : (
            <>
              <Copy size={14} strokeWidth={2} />
              Copy
            </>
          )}
        </button>
      </div>

      <div className="muted" style={{ fontSize: '0.85rem' }}>
        If you update the widget design later, changes will appear on your website automatically.
      </div>

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        <button
          type="button"
          className="primary"
          onClick={handleFinish}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <PartyPopper size={18} strokeWidth={2} aria-hidden />
          Finish setup
        </button>
      </div>
    </div>
  )
}
