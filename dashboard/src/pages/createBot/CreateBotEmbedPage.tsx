import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlowIcon } from '../../components/FlowIcon'
import { UiButton } from '../../components/ui'
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

      {/* Snippet block with external copy icon */}
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
        <pre className="snippet" style={{ margin: 0, flex: 1, minWidth: 0 }}>
          {snippet}
        </pre>
        <button
          type="button"
          onClick={() => void handleCopy()}
          disabled={!snippet}
          title={copied ? 'Copied!' : 'Copy to clipboard'}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            border: '1px solid var(--flow-border, #f2d8d2)',
            background: copied ? 'var(--flow-accent-soft, #fff1ef)' : 'var(--flow-surface, #ffffff)',
            color: copied ? 'var(--flow-accent, #e4587a)' : 'var(--flow-muted, #7e5a70)',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            flexShrink: 0,
          }}
        >
          <FlowIcon name={copied ? 'check' : 'content_copy'} size="sm" />
        </button>
      </div>

      <div className="muted" style={{ fontSize: '0.85rem' }}>
        If you update the widget design later, changes will appear on your website automatically.
      </div>

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        <UiButton
          variant="primary"
          onClick={handleFinish}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <FlowIcon name="celebration" filled size="sm" />
          Finish setup
        </UiButton>
      </div>
    </div>
  )
}
