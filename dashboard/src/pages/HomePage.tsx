import { Bot, ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../components/ui'

export default function HomePage() {
  return (
    <AnimatedPage className="page">
      <PageHeader title="Dashboard" />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow="Welcome"
          title="Get started"
          subtitle="Select a bot from the sidebar or create a new one to begin."
        />
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--ui-flow-brand-gradient)', color: '#fff' }}>
              <Bot size={22} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>Your AI bots</div>
              <p className="muted" style={{ margin: '0.15rem 0 0' }}>Manage, train, and deploy intelligent assistants.</p>
            </div>
            <Link to="/bots">
              <UiButton variant="secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                View bots
                <ArrowRight size={14} />
              </UiButton>
            </Link>
          </div>
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
