import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { X } from 'lucide-react'
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
          const isPast = index < activeStep
          return (
            <div key={step.id} className={`flow-step ${isCurrent ? 'active' : ''}`}>
              <div className="flow-step-number">
                {isPast ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  index + 1
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
  const { error, loading } = useDashboardData()

  return (
    <CreateBotProvider>
      <div className="flow-shell">
        <header className="flow-header">
          <div>
            <div className="flow-eyebrow">Set up</div>
            <div className="flow-title">Create your AI agent</div>
          </div>
          <button type="button" className="flow-close" onClick={() => navigate('/bots')} aria-label="Close">
            <X size={18} strokeWidth={2} aria-hidden />
          </button>
        </header>

        <div className="flow-grid">
          <FlowStepsWithProgress />
          <section className="flow-panel">
            {error && <div className="alert error">{error}</div>}
            {loading && <div className="alert info">Working...</div>}
            <Outlet />
          </section>
        </div>
      </div>
    </CreateBotProvider>
  )
}
