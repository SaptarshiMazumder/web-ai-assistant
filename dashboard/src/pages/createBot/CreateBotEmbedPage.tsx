import { useEffect, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotEmbedPage() {
  const navigate = useNavigate()
  const { buildEmbedSnippet, copySnippet } = useDashboardData()
  const { step3, step4, flow, resetFlow } = useCreateBotFlow()
  const { botId } = step3

  const snippet = useMemo(
    () =>
      buildEmbedSnippet({
        position: step4.widgetPosition,
        primaryColor: step4.widgetPrimaryColor,
        title: step4.widgetTitle,
        size: step4.widgetSize,
      }),
    [buildEmbedSnippet, step4.widgetPosition, step4.widgetPrimaryColor, step4.widgetTitle, step4.widgetSize]
  )

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  const handleCopy = () => {
    void copySnippet(snippet)
  }

  const handleFinish = () => {
    resetFlow()
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Add the script to your website</div>
        <div className="card-subtitle">
          Paste this snippet before the closing <code>&lt;/body&gt;</code> tag on your site to show the chat widget.
        </div>
      </div>

      <div>
        <pre className="snippet" style={{ marginTop: '12px', marginBottom: '12px' }}>
          {snippet}
        </pre>
        <button type="button" className="secondary" onClick={handleCopy} disabled={!snippet}>
          Copy snippet
        </button>
      </div>

      <div className="muted" style={{ marginTop: '16px', fontSize: '14px' }}>
        The widget will use the design you chose (position, color, title). You can change these later in the bot overview.
      </div>

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        {botId ? (
          <Link className="primary" to={`/bots/${botId}/overview`} onClick={handleFinish}>
            Go to bot overview
          </Link>
        ) : (
          <Link className="primary" to="/bots" onClick={handleFinish}>
            Go to bots
          </Link>
        )}
        <Link className="ghost" to="/bots" onClick={handleFinish}>
          Back to bots
        </Link>
      </div>
    </div>
  )
}
