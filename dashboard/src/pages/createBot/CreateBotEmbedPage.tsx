import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Check } from 'lucide-react'
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
    // Do not call resetFlow() here: it clears botId and triggers the useEffect above to redirect to flow.firstPath.
    // Navigating away unmounts CreateBotLayout so create-bot state is discarded anyway.
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Add chat to your website</div>
        <div className="card-subtitle">
          If you have a web person, send them this code. They will add it to your website to show the chat.
        </div>
      </div>

      <div>
        <pre className="snippet" style={{ marginTop: '12px', marginBottom: '12px' }}>
          {snippet}
        </pre>
        <button type="button" className="secondary" onClick={() => void handleCopy()} disabled={!snippet} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          {copied ? (
            <>
              <Check size={18} strokeWidth={2.5} aria-hidden />
              <span>Copied</span>
            </>
          ) : (
            'Copy snippet'
          )}
        </button>
      </div>

      <div className="muted" style={{ marginTop: '16px', fontSize: '14px' }}>
        If you change the design later, your website will show the new look the next time it loads.
      </div>

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        <button type="button" className="primary" onClick={handleFinish} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <Check size={18} strokeWidth={2.5} aria-hidden />
          <span>Finish</span>
        </button>
      </div>
    </div>
  )
}
