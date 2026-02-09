import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'
import { getTrainingStageLabel } from './trainingProgressLabels'

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

  // Auto-advance to step 4 (Design widget) ~5s after crawl job starts so user can design while training runs in background
  useEffect(() => {
    if (!jobId || trainingStage !== 'training' || !flow.nextPath) return
    const t = window.setTimeout(() => {
      navigate(flow.nextPath!)
    }, 5000)
    return () => window.clearTimeout(t)
  }, [jobId, trainingStage, flow.nextPath, navigate])

  // If user skipped sources, there's no jobId; let them move on immediately.
  useEffect(() => {
    if (trainingStage !== 'complete' || jobId || !flow.nextPath) return
    const t = window.setTimeout(() => {
      navigate(flow.nextPath!)
    }, 800)
    return () => window.clearTimeout(t)
  }, [trainingStage, jobId, flow.nextPath, navigate])

  const handleFinish = () => {
    resetFlow()
  }

  const currentStageLabel = getTrainingStageLabel(jobId, trainingStage, trainingStageName)

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Training your bot</div>
        <div className="card-subtitle">
          {trainingStageName === 'skipped'
            ? 'No sources added, skipping training. You can add sources later in Knowledge.'
            : trainingStage === 'complete'
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

      {trainingStageName === 'skipped' && botId && (
        <div className="alert info">
          You can add sources later from <b>Bot → Knowledge</b>.
        </div>
      )}

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
          <div className="muted">
            Training runs in the background. We&apos;ll take you to design your widget in a few seconds—you don&apos;t need to wait here.
          </div>
        )}
      </div>
    </div>
  )
}
