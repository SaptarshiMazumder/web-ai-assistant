import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { UiButton, UiCard } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'
import { getTrainingStageLabel, getTrainingStageLabelByName } from './trainingProgressLabels'

export default function CreateBotProgressPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { step2, step3, flow } = useCreateBotFlow()
  const {
    trainingStage,
    trainingProgress,
    trainingPagesCrawled,
    trainingDocsCount,
    trainingStageName,
    trainingStageMessage,
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
    if ((!jobId && pdfJobIds.length === 0) || trainingStage !== 'training' || !flow.nextPath || !!localError) return
    const t = window.setTimeout(() => {
      navigate(flow.nextPath!)
    }, 5000)
    return () => window.clearTimeout(t)
  }, [jobId, pdfJobIds.length, trainingStage, flow.nextPath, navigate, localError])

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
    if (botId) {
      navigate(`/bots/${botId}/overview`)
      return
    }
    navigate('/bots')
  }

  const currentStageLabel =
    !jobId && pdfJobs.length > 0
      ? t('createBot.preparingPdfFiles', 'Preparing your PDF files...')
      : getTrainingStageLabel(jobId, trainingStage, trainingStageName, t, trainingStageMessage)

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
            ? t('createBot.noSourcesAdded', 'No sources added yet')
            : isComplete
              ? t('createBot.agentReady', 'Your agent is ready')
              : t('createBot.trainingAgent', 'Training your agent')}
        </div>
        <div className="card-subtitle">
          {trainingStageName === 'skipped'
            ? t('createBot.youCanAddPagesLater', 'You can add pages and PDFs later from your bot settings.')
            : isComplete
              ? t('createBot.allSourcesProcessed', 'All sources have been processed. Your agent is ready to chat.')
              : t('createBot.wereReadingYourContent', 'We\'re reading your content and teaching your agent. This won\'t take long.')}
        </div>
      </div>

      {localError && <div className="alert error">{localError}</div>}

      {isIdle ? (
        <>
          <UiCard style={{ padding: '1.25rem', boxShadow: 'none' }}>
            <div style={{ fontWeight: 700, color: 'var(--flow-heading)', marginBottom: '0.75rem' }}>
              {t('createBot.sourcesReadyForTraining', 'Sources ready for training')}
            </div>
            {hasAnySources ? (
              <div style={{ display: 'grid', gap: '0.6rem' }}>
                {sourceUrls.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    {t('createBot.websitePages', 'Website pages:')} <b>{sourceUrls.length}</b>
                  </div>
                )}
                {pdfFiles.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    {t('createBot.pdfFiles', 'PDF files:')} <b>{pdfFiles.length}</b>
                  </div>
                )}
                {textDocFiles.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    {t('createBot.textDocs', 'Text docs:')} <b>{textDocFiles.length}</b>
                  </div>
                )}
                {usedCustomEntries.length > 0 && (
                  <div className="muted" style={{ fontSize: '0.9rem' }}>
                    {t('createBot.customTextEntries', 'Custom text entries:')} <b>{usedCustomEntries.length}</b>
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
                      {entry.title || t('createBot.customEntryWithId', 'Custom entry {{id}}', { id: entry.id })}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="muted" style={{ fontSize: '0.9rem' }}>
                {t('createBot.noSourcesAddedYet', 'No sources added yet.')}
              </div>
            )}
          </UiCard>

          <div className="flow-actions">
            <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)} disabled={isStartingTraining}>
              {t('common.back', 'Back')}
            </UiButton>
            {hasAnySources ? (
              <UiButton
                variant="primary"
                onClick={() => void handleStartTraining()}
                disabled={isStartingTraining}
                style={{ marginLeft: 'auto' }}
              >
                {isStartingTraining ? t('createBot.starting', 'Starting...') : t('createBot.startTraining', 'Start training')}
              </UiButton>
            ) : (
              <UiButton
                variant="ghost"
                onClick={() => void handleSkipTraining()}
                disabled={isStartingTraining}
                style={{ marginLeft: 'auto' }}
              >
                {isStartingTraining ? t('createBot.skipping', 'Skipping...') : t('createBot.skipTrainingForNow', 'Skip training for now')}
              </UiButton>
            )}
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
                {trainingPagesCrawled > 0 && <span>{t('createBot.pagesRead', '{{count}} pages read', { count: trainingPagesCrawled })}</span>}
                {trainingDocsCount > 0 && <span>{t('createBot.documents', '{{count}} documents', { count: trainingDocsCount })}</span>}
              </div>
            )}
          </div>

          {/* PDF job statuses */}
          {pdfJobs.length > 0 && (
            <UiCard style={{ padding: '1.25rem', boxShadow: 'none' }}>
              <div style={{ fontWeight: 600, color: 'var(--flow-text)', marginBottom: '0.5rem', fontSize: '0.95rem' }}>
                {t('createBot.pdfFilesLabel', 'PDF files')}
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
                        {getTrainingStageLabelByName((j.stage || 'queued').toString(), t)}
                      </span>
                      {typeof j.docs_count === 'number' && (
                        <span className="muted" style={{ fontSize: '0.8rem' }}>{t('createBot.docsCount', '{{count}} docs', { count: j.docs_count })}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </UiCard>
          )}

          {trainingStageName === 'skipped' && botId && (
            <div className="alert info">
              {t('createBot.addSourcesLaterHint', 'You can add pages and PDFs later from {{path}}.', {
                path: 'Bot -> Knowledge',
              })}
            </div>
          )}

          <div className="flow-actions">
            <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)} style={{ marginRight: 'auto' }}>
              {t('common.back', 'Back')}
            </UiButton>
            {isComplete ? (
              flow.nextPath ? (
                <UiButton variant="primary" onClick={() => navigate(flow.nextPath!)} style={{ marginLeft: 'auto' }}>
                  {t('common.continue', 'Continue')}
                </UiButton>
              ) : (
                <UiButton variant="primary" onClick={handleFinish} style={{ marginLeft: 'auto' }}>
                  {t('createBot.finishSetup', 'Finish setup')}
                </UiButton>
              )
            ) : flow.nextPath ? (
              <UiButton variant="primary" onClick={() => navigate(flow.nextPath!)} style={{ marginLeft: 'auto' }}>
                {t('common.continue', 'Continue')}
              </UiButton>
            ) : (
              <UiButton variant="primary" disabled style={{ marginLeft: 'auto' }}>
                {t('common.continue', 'Continue')}
              </UiButton>
            )}
          </div>
          {!isComplete && (
            <div className="muted" style={{ fontSize: '0.85rem' }}>
              {t('createBot.thisRunsInBackground', 'This runs in the background. We\'ll take you to the design step shortly.')}
            </div>
          )}
        </>
      )}
    </div>
  )
}
