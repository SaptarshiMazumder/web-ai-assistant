import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { Bot, Building2, Sparkles, Zap } from 'lucide-react'
import { AnimatedPage, BentoGrid, BentoItem, GlassCard, MetricCard, SectionHeader, UiButton } from '../components/ui'

export default function DashboardPage() {
  const { bots, orgs, activeOrgId, isSuperAdmin } = useDashboardData()
  const botCount = bots.length
  const orgCount = orgs.length

  return (
    <AnimatedPage className="page">
      <PageHeader title="Dashboard" />
      <div className="page-body dashboard-redesign">
        <SectionHeader
          eyebrow="Control Center"
          title="Your AI operations at a glance"
          subtitle="Track growth, jump into core workflows, and keep every bot moving."
        />

        <BentoGrid>
          <BentoItem colSpan={2}>
            <GlassCard className="dashboard-hero-card">
              <div className="card-title">Active organization</div>
              <h3>{activeOrgId || 'No organization selected'}</h3>
              <p className="dashboard-hero-id">Primary workspace for analytics, bots, and settings.</p>
            </GlassCard>
          </BentoItem>
          <BentoItem>
            <MetricCard label="Bots" value={botCount} icon={<Bot size={15} />} trend="Live agents in workspace" />
          </BentoItem>
          <BentoItem>
            <MetricCard
              label="Organizations"
              value={isSuperAdmin ? orgCount : orgs.length || 1}
              icon={<Building2 size={15} />}
              trend="Visible teams"
            />
          </BentoItem>
          <BentoItem>
            <MetricCard label="Automation lift" value={82} suffix="%" icon={<Sparkles size={15} />} trend="+12% this month" />
          </BentoItem>
          <BentoItem colSpan={3}>
            <GlassCard>
              <SectionHeader eyebrow="Quick actions" title="Launch high-impact moves" />
              <div className="dashboard-quick-actions">
                <UiButton variant="primary">Create new bot</UiButton>
                <UiButton variant="secondary">Review conversations</UiButton>
                <UiButton variant="ghost">Open analytics report</UiButton>
                <UiButton variant="secondary">
                  <Zap size={14} />
                  <span>Run optimization scan</span>
                </UiButton>
              </div>
            </GlassCard>
          </BentoItem>
        </BentoGrid>
      </div>
    </AnimatedPage>
  )
}
