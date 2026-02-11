import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { UiButton, UiCard } from '../../components/ui'
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

  // Auto-advance to next step ~5s after learning starts so user can continue setup while training runs in background
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
      ? 'Preparing your PDF files...'
      : getTrainingStageLabel(jobId, trainingStage, trainingStageName)

  const isComplete = trainingStage === 'complete'

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">
          {trainingStageName === 'skipped'
            ? 'No sources added yet'
            : isComplete
            ? 'Your agent is ready'
            : 'Training your agent'}
        </div>
        <div className="card-subtitle">
          {trainingStageName === 'skipped'
            ? 'You can add pages and PDFs later from your bot settings.'
            : isComplete
            ? 'All sources have been processed. Your agent is ready to chat.'
            : 'We\'re reading your content and teaching your agent. This won\'t take long.'}
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      {/* Progress visualization */}
      <div className="progress-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="progress-label">{currentStageLabel}</div>
          <span style={{
            fontSize: '0.85rem',
            fontWeight: 700,
            color: 'var(--flow-accent)',
          }}>
            {trainingProgress}%
          </span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${trainingProgress}%` }} />
        </div>
        {(trainingPagesCrawled > 0 || trainingDocsCount > 0) && (
          <div className="muted" style={{ fontSize: '0.8rem', display: 'flex', gap: '1rem' }}>
            {trainingPagesCrawled > 0 && <span>{trainingPagesCrawled} pages read</span>}
            {trainingDocsCount > 0 && <span>{trainingDocsCount} documents</span>}
          </div>
        )}
      </div>

      {/* PDF job statuses */}
      {pdfJobs.length > 0 && (
        <UiCard style={{ padding: '1.25rem', boxShadow: 'none' }}>
          <div style={{ fontWeight: 600, color: 'var(--flow-text)', marginBottom: '0.5rem', fontSize: '0.95rem' }}>
            PDF files
          </div>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            {pdfJobs.map((j) => (
              <div key={j.job_id} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.5rem 0',
                borderBottom: '1px solid var(--flow-border)',
              }}>
                <div className="muted" style={{ fontSize: '0.8rem' }}>{j.job_id}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{
                    padding: '0.2rem 0.6rem',
                    borderRadius: 999,
                    background: 'var(--flow-accent-soft)',
                    color: 'var(--flow-accent)',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                  }}>
                    {TRAINING_STAGE_LABELS[(j.stage || 'queued').toString()] || (j.stage || 'queued').toString()}
                  </span>
                  {typeof j.docs_count === 'number' && (
                    <span className="muted" style={{ fontSize: '0.8rem' }}>{j.docs_count} docs</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </UiCard>
      )}

      {trainingStageName === 'skipped' && botId && (
        <div className="alert info">
          You can add pages and PDFs later from <b>Bot &rarr; Knowledge</b>.
        </div>
      )}

      <div className="flow-actions">
        {isComplete ? (
          <>
            {flow.nextPath && (
              <UiButton variant="primary" onClick={() => navigate(flow.nextPath!)}>
                Continue
              </UiButton>
            )}
            {botId ? (
              <Link className="secondary" to={`/bots/${botId}/overview`} onClick={handleFinish}
                style={{ display: 'inline-flex', alignItems: 'center', padding: '0.65rem 1.25rem', borderRadius: 10, border: '1.5px solid var(--flow-border)', textDecoration: 'none', fontWeight: 500, fontSize: '0.9rem', color: 'var(--flow-text)' }}>
                Go to bot overview
              </Link>
            ) : (
              <Link className="secondary" to="/bots" onClick={handleFinish}
                style={{ display: 'inline-flex', alignItems: 'center', padding: '0.65rem 1.25rem', borderRadius: 10, border: '1.5px solid var(--flow-border)', textDecoration: 'none', fontWeight: 500, fontSize: '0.9rem', color: 'var(--flow-text)' }}>
                Go to bots
              </Link>
            )}
          </>
        ) : (
          <div className="muted" style={{ fontSize: '0.85rem' }}>
            This runs in the background. We&apos;ll take you to the design step shortly.
          </div>
        )}
      </div>
    </div>
  )
}
