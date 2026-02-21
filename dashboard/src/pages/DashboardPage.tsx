import { useTranslation } from 'react-i18next'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { Bot, Building2, Sparkles, Zap } from 'lucide-react'
import { AnimatedPage, BentoGrid, BentoItem, GlassCard, MetricCard, SectionHeader, UiButton } from '../components/ui'

export default function DashboardPage() {
  const { t } = useTranslation()
  const { bots, orgs, activeOrgId, isSuperAdmin } = useDashboardData()
  const botCount = bots.length
  const orgCount = orgs.length

  return (
    <AnimatedPage className="page">
      <PageHeader title={t('dashboard.title', 'Dashboard')} />
      <div className="page-body dashboard-redesign">
        <SectionHeader
          eyebrow={t('dashboard.controlCenter', 'Control Center')}
          title={t('dashboard.operationsAtGlance', 'Your AI operations at a glance')}
          subtitle={t('dashboard.operationsSubtitle', 'Track growth, jump into core workflows, and keep every bot moving.')}
        />

        <BentoGrid>
          <BentoItem colSpan={2}>
            <GlassCard className="dashboard-hero-card">
              <div className="card-title">{t('dashboard.activeOrg', 'Active organization')}</div>
              <h3>{activeOrgId || t('dashboard.noOrgSelected', 'No organization selected')}</h3>
              <p className="dashboard-hero-id">{t('dashboard.primaryWorkspace', 'Primary workspace for analytics, bots, and settings.')}</p>
            </GlassCard>
          </BentoItem>
          <BentoItem>
            <MetricCard label={t('dashboard.botsLabel', 'Bots')} value={botCount} icon={<Bot size={15} />} trend={t('dashboard.botsTrend', 'Live agents in workspace')} />
          </BentoItem>
          <BentoItem>
            <MetricCard
              label={t('dashboard.orgsLabel', 'Organizations')}
              value={isSuperAdmin ? orgCount : orgs.length || 1}
              icon={<Building2 size={15} />}
              trend={t('dashboard.orgsTrend', 'Visible teams')}
            />
          </BentoItem>
          <BentoItem>
            <MetricCard label={t('dashboard.automationLift', 'Automation lift')} value={82} suffix="%" icon={<Sparkles size={15} />} trend={t('dashboard.automationTrend', '+12% this month')} />
          </BentoItem>
          <BentoItem colSpan={3}>
            <GlassCard>
              <SectionHeader eyebrow={t('dashboard.quickActionsEyebrow', 'Quick actions')} title={t('dashboard.quickActionsTitle', 'Launch high-impact moves')} />
              <div className="dashboard-quick-actions">
                <UiButton variant="primary">{t('dashboard.createNewBot', 'Create new bot')}</UiButton>
                <UiButton variant="secondary">{t('dashboard.reviewConversations', 'Review conversations')}</UiButton>
                <UiButton variant="ghost">{t('dashboard.openAnalytics', 'Open analytics report')}</UiButton>
                <UiButton variant="secondary">
                  <Zap size={14} />
                  <span>{t('dashboard.runOptimization', 'Run optimization scan')}</span>
                </UiButton>
              </div>
            </GlassCard>
          </BentoItem>
        </BentoGrid>
      </div>
    </AnimatedPage>
  )
}
