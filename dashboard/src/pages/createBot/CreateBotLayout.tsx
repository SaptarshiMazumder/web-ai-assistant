import { Link, Outlet, useLocation } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import { CreateBotProvider } from './CreateBotContext'

const steps = [
  {
    id: 'details',
    label: 'Name + Website',
    description: 'Give your bot a name and the site to learn from.',
    path: '/create-bot',
  },
  {
    id: 'urls',
    label: 'Select URLs',
    description: 'Choose which pages should be included.',
    path: '/create-bot/urls',
  },
  {
    id: 'training',
    label: 'Training',
    description: 'We will start processing your sources.',
    path: '/create-bot/progress',
  },
]

export default function CreateBotLayout() {
  const { error, loading } = useDashboardData()
  const location = useLocation()
  const activeIndex = steps.findIndex((step) => step.path === location.pathname)
  const activeStep = activeIndex >= 0 ? activeIndex : 0

  return (
    <CreateBotProvider>
      <div className="flow-shell">
        <header className="flow-header">
          <div>
            <div className="flow-eyebrow">Create bot</div>
            <div className="flow-title">Set up a new chatbot agent</div>
          </div>
          <Link className="ghost flow-back" to="/bots">
            Back to bots
          </Link>
        </header>

        <div className="flow-grid">
          <aside className="flow-steps">
            {steps.map((step, index) => (
              <div key={step.id} className={`flow-step ${index === activeStep ? 'active' : ''}`}>
                <div className="flow-step-number">{index + 1}</div>
                <div>
                  <div className="flow-step-title">{step.label}</div>
                  <div className="flow-step-desc">{step.description}</div>
                </div>
              </div>
            ))}
          </aside>
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
