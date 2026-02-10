import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'
import { getTrainingStageLabel, TRAINING_STAGE_LABELS } from './trainingProgressLabels'

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
    pdfJobIds,
    pdfJobs,
    localError,
    resetFlow,
  } = step3

  useEffect(() => {
    if (trainingStage === 'idle') {
      navigate(flow.firstPath)
    }
  }, [trainingStage, navigate, flow.firstPath])

  // Auto-advance to step 4 (Design widget) ~5s after learning starts so user can design while setup runs in background
  useEffect(() => {
    if ((!jobId && pdfJobIds.length === 0) || trainingStage !== 'training' || !flow.nextPath) return
    const t = window.setTimeout(() => {
      navigate(flow.nextPath!)
    }, 5000)
    return () => window.clearTimeout(t)
  }, [jobId, pdfJobIds.length, trainingStage, flow.nextPath, navigate])

  // If user skipped sources, there's no jobId; let them move on immediately.
  useEffect(() => {
    if (trainingStage !== 'complete' || trainingStageName !== 'skipped' || jobId || pdfJobIds.length > 0 || !flow.nextPath) return
    const t = window.setTimeout(() => {
      navigate(flow.nextPath!)
    }, 800)
    return () => window.clearTimeout(t)
  }, [trainingStage, trainingStageName, jobId, pdfJobIds.length, flow.nextPath, navigate])

  const handleFinish = () => {
    resetFlow()
  }

  const currentStageLabel =
    !jobId && pdfJobs.length > 0
      ? 'Preparing your PDF files…'
      : getTrainingStageLabel(jobId, trainingStage, trainingStageName)

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Getting your agent ready</div>
        <div className="card-subtitle">
          {trainingStageName === 'skipped'
            ? 'You didn’t add anything yet. You can do this later from your bot settings.'
            : trainingStage === 'complete'
            ? 'All set!'
            : 'We’re reading what you added and getting your agent ready.'}
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="progress-card">
        <div className="progress-label">
          {currentStageLabel}
          {trainingPagesCrawled > 0 && ` • ${trainingPagesCrawled} pages read`}
          {trainingDocsCount > 0 && ` • ${trainingDocsCount} documents`}
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${trainingProgress}%` }} />
        </div>
        <div className="muted">{trainingProgress}% complete</div>
      </div>

      {pdfJobs.length > 0 && (
        <div className="card" style={{ padding: '1rem' }}>
          <div className="card-title" style={{ marginBottom: '0.25rem' }}>PDF files</div>
          <div className="muted" style={{ marginBottom: '0.75rem' }}>We’re preparing each file you uploaded.</div>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            {pdfJobs.map((j) => (
              <div key={j.job_id} className="row" style={{ justifyContent: 'space-between', gap: '0.75rem' }}>
                <div className="muted" style={{ fontFamily: 'monospace' }}>{j.job_id}</div>
                <div>
                  <span className="pill" style={{ padding: '0.2rem 0.5rem', borderRadius: 999, background: '#e2e8f0', color: '#334155' }}>
                    {TRAINING_STAGE_LABELS[(j.stage || 'queued').toString()] || (j.stage || 'queued').toString()}
                  </span>
                  {typeof j.docs_count === 'number' && (
                    <span className="muted" style={{ marginLeft: '8px' }}>{j.docs_count} docs</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {trainingStageName === 'skipped' && botId && (
        <div className="alert info">
          You can add pages and PDFs later from <b>Bot → Knowledge</b>.
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
            This runs in the background. In a few seconds we&apos;ll take you to the design step—you don&apos;t need to wait here.
          </div>
        )}
      </div>
    </div>
  )
}
