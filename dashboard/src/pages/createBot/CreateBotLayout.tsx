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
        {steps.map((step, index) => (
          <div key={step.id} className={`flow-step ${index === activeStep ? 'active' : ''}`}>
            <div className="flow-step-number">{index + 1}</div>
            <div>
              <div className="flow-step-title">{step.label}</div>
              <div className="flow-step-desc">{step.description}</div>
            </div>
          </div>
        ))}
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
            <div className="flow-title">Create your AI helper</div>
          </div>
          <button type="button" className="ghost flow-back flow-close" onClick={() => navigate('/bots')} aria-label="Close">
            <X size={20} strokeWidth={2} aria-hidden />
          </button>
        </header>

        <div className="flow-grid">
          <FlowStepsWithProgress />
          <section className="flow-panel">
            {error && <div className="alert error">{error}</div>}
            {loading && <div className="alert">Working...</div>}
            <Outlet />
          </section>
        </div>
      </div>
    </CreateBotProvider>
  )
}
