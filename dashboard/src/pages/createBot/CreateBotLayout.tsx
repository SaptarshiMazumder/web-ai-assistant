import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { FlowIcon } from '../../components/FlowIcon'
import { useDashboardData } from '../../hooks/useDashboardData'
import { CreateBotProvider, useCreateBotFlow } from './CreateBotContext'
import { getCreateBotStepIndex, getCreateBotSteps, getCreateBotPrevPath } from './flowConfig'
import { TrainingProgressCircle } from './TrainingProgressCircle'


function FlowStepsWithProgress() {
  const location = useLocation()
  const { t } = useTranslation()
  const { step3 } = useCreateBotFlow()
  const steps = getCreateBotSteps()
  const activeStep = getCreateBotStepIndex(location.pathname, steps)
  const activeId = steps[activeStep]?.id
  const {
    trainingStage,
    trainingProgress,
    trainingPagesCrawled,
    trainingDocsCount,
    trainingStageName,
    jobId,
    pdfJobIds,
  } = step3
  const hasBackgroundTraining = trainingStage === 'training' || !!jobId || pdfJobIds.length > 0
  const showProgress = (activeId === 'topics' || activeId === 'widget' || activeId === 'embed') && hasBackgroundTraining

  return (
    <aside className="flow-steps">
      <div className="flow-steps-list">
        {steps.map((step, index) => {
          const isCurrent = index === activeStep
          return (
            <div key={step.id} className={`flow-step ${isCurrent ? 'active' : ''}`}>
              <div className="flow-step-number">
                {step.iconUrl ? (
                  <img className="flow-step-icon" src={step.iconUrl} alt="" aria-hidden="true" />
                ) : (
                  <FlowIcon
                    name={step.icon as import('../../components/FlowIcon').FlowIconName}
                    filled
                    size="md"
                    style={{ fontSize: 28 }}
                  />
                )}
              </div>
              <div>
                <div className="flow-step-title">{t(step.labelKey, step.label)}</div>
                <div className="flow-step-desc">{t(step.descriptionKey, step.description)}</div>
              </div>
            </div>
          )
        })}
      </div>
      {showProgress && (
        <div className="flow-steps-progress flow-steps-progress--enter">
          <TrainingProgressCircle
            progress={trainingProgress}
            jobId={jobId}
            trainingStage={trainingStage}
            trainingStageName={trainingStageName}
            trainingPagesCrawled={trainingPagesCrawled}
            trainingDocsCount={trainingDocsCount}
          />
        </div>
      )}
    </aside>
  )
}

export default function CreateBotLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const { t } = useTranslation()
  const { error } = useDashboardData()

  return (
    <CreateBotProvider>
      <div className="flow-shell">
        <header className="flow-header">
          <div>
            <div className="flow-eyebrow">{t('createBot.flowHeaderEyebrow', 'Set up')}</div>
            <div className="flow-title">{t('createBot.flowHeaderTitle', 'Create AI Agent')}</div>
          </div>
          <button
            type="button"
            className="flow-close"
            onClick={() => navigate('/bots')}
            aria-label={t('common.close', 'Close')}
          >
            <FlowIcon name="close" size="sm" />
          </button>
        </header>

        <div className="flow-grid">
          <FlowStepsWithProgress />
          <section className="flow-panel">
            {error && <div className="alert error">{error}</div>}
            {(() => {
              const prevPath = getCreateBotPrevPath(location.pathname)
              return prevPath ? (
                <button
                  type="button"
                  className="flow-mobile-back"
                  onClick={() => navigate(prevPath)}
                  aria-label={t('common.goBack', 'Go back')}
                >
                  <FlowIcon name="arrow_back" size="md" />
                </button>
              ) : null
            })()}
            <div key={location.pathname} className="flow-panel-animate">
              <Outlet />
            </div>
          </section>
        </div>
      </div>
    </CreateBotProvider>
  )
}
