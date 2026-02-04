import { Plus } from 'lucide-react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'

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
    <div className="page">
      <PageHeader title="Bots" />
      <div className="page-body page-body-narrow">
        <div className="bot-row">
          <button className="create-bot-button" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
            <Plus className="create-bot-icon" aria-hidden="true" />
            <span>Create bot</span>
          </button>

          {bots.map((bot) => (
            <button key={bot.bot_id} className="bot-card" onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}>
              <div className="list-title">{bot.display_name}</div>
              <div className="muted">{bot.bot_id}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
