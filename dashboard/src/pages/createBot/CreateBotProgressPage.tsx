import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotProgressPage() {
  const navigate = useNavigate()
  const { step3, flow } = useCreateBotFlow()
  const {
    trainingStage,
    trainingProgress,
    trainingPagesCrawled,
    trainingDocsCount,
    trainingStageName,
    botId,
    jobId,
    localError,
    resetFlow,
  } = step3

  useEffect(() => {
    if (trainingStage === 'idle') {
      navigate(flow.firstPath)
    }
  }, [trainingStage, navigate, flow.firstPath])

  const handleFinish = () => {
    resetFlow()
  }

  const stageLabels: Record<string, string> = {
    queued: 'Queued',
    crawling: 'Crawling pages',
    uploading: 'Uploading to storage',
    importing: 'Importing to knowledge base',
    import_submitted: 'Import submitted',
    done: 'Complete',
    error: 'Error',
  }

  const currentStageLabel =
    !jobId && trainingStage === 'training'
      ? 'Starting crawl…'
      : stageLabels[trainingStageName] || trainingStageName || 'Processing'

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Training your bot</div>
        <div className="card-subtitle">
          {trainingStage === 'complete'
            ? 'Training completed successfully!'
            : 'We are crawling the selected pages and preparing your chatbot knowledge base.'}
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="progress-card">
        <div className="progress-label">
          {currentStageLabel}
          {trainingPagesCrawled > 0 && ` • ${trainingPagesCrawled} pages crawled`}
          {trainingDocsCount > 0 && ` • ${trainingDocsCount} documents`}
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${trainingProgress}%` }} />
        </div>
        <div className="muted">{trainingProgress}% complete</div>
      </div>

      <div className="flow-actions">
        {trainingStage === 'complete' ? (
          <>
            {flow.nextPath ? (
              <button type="button" className="primary" onClick={() => navigate(flow.nextPath!)}>
                Continue
              </button>
            ) : null}
            {botId ? (
              <Link className="secondary" to={`/bots/${botId}/overview`} onClick={handleFinish}>
                Go to bot overview
              </Link>
            ) : (
              <Link className="secondary" to="/bots" onClick={handleFinish}>
                Go to bots
              </Link>
            )}
            <Link className="ghost" to="/bots" onClick={handleFinish}>
              Back to bots
            </Link>
          </>
        ) : (
          <div className="muted">This usually takes a few minutes. You can leave this tab open.</div>
        )}
      </div>
    </div>
  )
}
