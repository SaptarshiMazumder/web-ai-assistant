import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { UiButton, UiCard } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'
import { getTrainingStageLabel, TRAINING_STAGE_LABELS } from './trainingProgressLabels'

export default function CreateBotProgressPage() {
  const navigate = useNavigate()
  const { step2, step3, flow } = useCreateBotFlow()
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
  const { contentHosting, selectedUrls, trainingUrls, pdfFiles, textDocFiles, customTextEntries, startTraining, continueWithoutSources, isStartingTraining } = step2
  const sourceUrls = contentHosting === 'own' ? selectedUrls : trainingUrls
  const usedCustomEntries = customTextEntries.filter((entry) => entry.title.trim() || entry.content.trim())
  const hasAnySources = sourceUrls.length > 0 || pdfFiles.length > 0 || textDocFiles.length > 0 || usedCustomEntries.length > 0

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
  const isIdle = trainingStage === 'idle'

  const handleStartTraining = async () => {
    await startTraining()
  }

  const handleSkipTraining = async () => {
    await continueWithoutSources()
  }

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

      {isIdle ? (
        <>
          <UiCard style={{ padding: '1.25rem', boxShadow: 'none' }}>
            <div style={{ fontWeight: 700, color: 'var(--flow-heading)', marginBottom: '0.75rem' }}>
              Sources ready for training
            </div>
            {hasAnySources ? (
              <div style={{ display: 'grid', gap: '0.6rem' }}>
                {sourceUrls.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    Website pages: <b>{sourceUrls.length}</b>
                  </div>
                )}
                {pdfFiles.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    PDF files: <b>{pdfFiles.length}</b>
                  </div>
                )}
                {textDocFiles.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    Text docs: <b>{textDocFiles.length}</b>
                  </div>
                )}
                {usedCustomEntries.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    Custom text entries: <b>{usedCustomEntries.length}</b>
                  </div>
                )}
                <div style={{ marginTop: '0.35rem', maxHeight: 180, overflow: 'auto' }}>
                  {sourceUrls.slice(0, 8).map((url) => (
                    <div key={url} className="muted" style={{ fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                      {url}
                    </div>
                  ))}
                  {pdfFiles.slice(0, 8).map((file) => (
                    <div key={file.name} className="muted" style={{ fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                      {file.name}
                    </div>
                  ))}
                  {textDocFiles.slice(0, 8).map((file) => (
                    <div key={file.name} className="muted" style={{ fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                      {file.name}
                    </div>
                  ))}
                  {usedCustomEntries.slice(0, 8).map((entry) => (
                    <div key={entry.id} className="muted" style={{ fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                      {entry.title || `Custom entry ${entry.id}`}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="muted" style={{ fontSize: '0.9rem' }}>
                No sources added yet.
              </div>
            )}
          </UiCard>

          <div className="flow-actions">
            <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)} disabled={isStartingTraining}>
              Back
            </UiButton>
            <div style={{ display: 'flex', gap: '0.75rem', marginLeft: 'auto' }}>
              {hasAnySources ? (
                <UiButton variant="primary" onClick={() => void handleStartTraining()} disabled={isStartingTraining}>
                  {isStartingTraining ? 'Starting...' : 'Start training'}
                </UiButton>
              ) : (
                <UiButton variant="ghost" onClick={() => void handleSkipTraining()} disabled={isStartingTraining}>
                  {isStartingTraining ? 'Skipping...' : 'Skip training for now'}
                </UiButton>
              )}
            </div>
          </div>
        </>
      ) : (
        <>
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
        </>
      )}
    </div>
  )
}
