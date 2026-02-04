import { useEffect } from 'react'
import { Outlet, useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import PageHeader from '../../components/PageHeader'

export default function BotDetailLayout() {
  const { botId } = useParams()
  const { selectedBot, setSelectedBotId, isSuperAdmin, activeOrgId } = useDashboardData()

  useEffect(() => {
    if (botId) {
      setSelectedBotId(botId)
    }
  }, [botId, setSelectedBotId])

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bot details.</div>
  }

  return (
    <div className="page">
      <PageHeader title={selectedBot?.display_name || 'Bot'} />
      <div className="page-body page-body-centered">
        <Outlet />
      </div>
    </div>
  )
}
