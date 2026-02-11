import { Plus } from 'lucide-react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, SectionHeader, StatusDot, UiButton } from '../components/ui'

export default function BotsPage() {
  const { bots, loading, isSuperAdmin, activeOrgId } = useDashboardData()
  const navigate = useNavigate()

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bots.</div>
  }

  if (!loading && bots.length === 0) {
    return <Navigate to="/create-bot" replace />
  }

  const canCreateBot = !isSuperAdmin || (activeOrgId && activeOrgId !== '__all__')

  return (
    <AnimatedPage className="page">
      <PageHeader title="Bots" />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow="Agents"
          title="Build, launch, and scale your bot fleet"
          subtitle="Every card is a live workspace with direct access to settings, analytics, and training."
        />

        {bots.length === 0 ? (
          <GlassCard>
            <EmptyState
              title="No bots yet"
              description="Start with one beautiful assistant and expand into a full AI team."
              action={
                <UiButton variant="primary" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
                  Create your first bot
                </UiButton>
              }
            />
          </GlassCard>
        ) : (
          <div className="bot-card-grid">
            <button className="cta-create-bot card" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
              <Plus className="create-bot-icon" aria-hidden="true" />
              <span>Create bot</span>
            </button>

            {bots.map((bot) => (
              <button key={bot.bot_id} className="card bot-card-modern" onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}>
                <div className="bot-card-title-row">
                  <div className="list-title">{bot.display_name}</div>
                  <StatusDot tone="success" />
                </div>
                <div className="bot-card-id">{bot.bot_id}</div>
              </button>
            ))}
          </div>
        )}
      </div>
    </AnimatedPage>
  )
}
