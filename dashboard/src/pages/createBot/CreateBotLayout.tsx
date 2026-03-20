import { useLocation, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'
import { FlowIcon } from '../../components/FlowIcon'
import { useDashboardData } from '../../hooks/useDashboardData'
import { CreateBotProvider, useCreateBotFlow } from './CreateBotContext'
import { TrainingProgressCircle } from './TrainingProgressCircle'
import CreateBotScreenHost from './CreateBotScreenHost'

function MobileStepIndicator() {
  const { flow } = useCreateBotFlow()
  const steps = flow.stepGroups
  const activeStep = flow.activeStepIndex

  if (!steps.length) return null

  const progress = steps.length > 1 ? (activeStep / (steps.length - 1)) * 100 : 0

  return (
    <div className="flow-progress-bar">
      <motion.div
        className="flow-progress-bar-fill"
        animate={{ width: `${progress}%` }}
        initial={false}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      />
    </div>
  )
}

function FlowStepsWithProgress() {
  const { step3, flow } = useCreateBotFlow()
  const steps = flow.stepGroups
  const activeStep = flow.activeStepIndex
  const activeId = steps[activeStep]?.id
  const {
    trainingStage,
    trainingProgress,
    trainingPagesCrawled,
    trainingDocsCount,
    trainingStageName,
    trainingStageMessage,
    jobId,
    pdfJobIds,
  } = step3
  const hasBackgroundTraining = trainingStage === 'training' || !!jobId || pdfJobIds.length > 0
  const showProgress = (activeId === 'widget' || activeId === 'embed') && hasBackgroundTraining

  return (
    <aside className="flow-steps">
      {/* Mobile: step indicator */}
      <div className="flow-mobile-header-controls">
        <MobileStepIndicator />
      </div>
      {/* Desktop: full step list */}
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
                <div className="flow-step-title">{step.label}</div>
                <div className="flow-step-desc">{step.description}</div>
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
            trainingStageMessage={trainingStageMessage}
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
            <div key={location.pathname} className="flow-panel-animate">
              <CreateBotScreenHost />
            </div>
          </section>
        </div>
      </div>
    </CreateBotProvider>
  )
}
