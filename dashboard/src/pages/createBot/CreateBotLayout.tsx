import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { FlowIcon } from '../../components/FlowIcon'
import { useDashboardData } from '../../hooks/useDashboardData'
import { CreateBotProvider, useCreateBotFlow } from './CreateBotContext'
import { getCreateBotStepIndex, getCreateBotSteps } from './flowConfig'
import { TrainingProgressCircle } from './TrainingProgressCircle'

function FlowStepsWithProgress() {
  const location = useLocation()
  const { step2, step3 } = useCreateBotFlow()
  const steps = getCreateBotSteps(step2.contentHosting)
  const activeStep = getCreateBotStepIndex(location.pathname, steps)
  const activeId = steps[activeStep]?.id
  const showProgress = activeId === 'widget' || activeId === 'embed'
  const {
    trainingStage,
    trainingProgress,
    trainingPagesCrawled,
    trainingDocsCount,
    trainingStageName,
    jobId,
  } = step3

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
                <div className="flow-step-title">{step.label}</div>
                <div className="flow-step-desc">{step.description}</div>
              </div>
            </div>
          )
        })}
      </div>
      {showProgress && (
        <div className="flow-steps-progress">
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
  const { error, loading } = useDashboardData()

  return (
    <CreateBotProvider>
      <div className="flow-shell">
        <header className="flow-header">
          <div>
            <div className="flow-eyebrow">Set up</div>
            <div className="flow-title">Create AI Agent</div>
          </div>
          <button type="button" className="flow-close" onClick={() => navigate('/bots')} aria-label="Close">
            <FlowIcon name="close" size="sm" />
          </button>
        </header>

        <div className="flow-grid">
          <FlowStepsWithProgress />
          <section className="flow-panel">
            {error && <div className="alert error">{error}</div>}
            {loading && <div className="alert info">Working...</div>}
            <div key={location.pathname} className="flow-panel-animate">
              <Outlet />
            </div>
          </section>
        </div>
      </div>
    </CreateBotProvider>
  )
}
