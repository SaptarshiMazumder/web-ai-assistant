import { useAuth0 } from '@auth0/auth0-react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { DashboardDataProvider, useDashboardData } from './hooks/useDashboardData'
import { BackgroundTaskProvider } from './contexts/BackgroundTaskContext'
import { DialogProvider } from './contexts/DialogContext'
import { BackgroundTaskIndicator } from './components/BackgroundTaskIndicator'
import DashboardLayout from './layouts/DashboardLayout'
import AccountPage from './pages/AccountPage'
import BillingPage from './pages/BillingPage'
import BotsPage from './pages/BotsPage'
import HomePage from './pages/HomePage'
import DomainPage from './pages/DomainPage'
import OrgPage from './pages/OrgPage'
import SettingsPage from './pages/SettingsPage'
import UsersPage from './pages/UsersPage'
import CreateBotLayout from './pages/createBot/CreateBotLayout'
import BotDesignTab from './pages/bot/BotDesignTab'
import BotDetailLayout from './pages/bot/BotDetailLayout'
import AddSourcePage from './pages/bot/AddSourcePage'
import BotKnowledgeTab from './pages/bot/BotKnowledgeTab'
import BotOverviewTab from './pages/bot/BotOverviewTab'
import BotSettingsTab from './pages/bot/BotSettingsTab'
import BotTestingTab from './pages/bot/BotTestingTab'
import BotConversationsTab from './pages/bot/BotConversationsTab'
import BotEscalationsTab from './pages/bot/BotEscalationsTab'
import BotLineSettingsTab from './pages/bot/BotLineSettingsTab'

import BotSuggestedMessagesTab from './pages/bot/BotSuggestedMessagesTab'
import BotWelcomeMessagesTab from './pages/bot/BotWelcomeMessagesTab'
import BotLeadsTab from './pages/bot/BotLeadsTab'
import BotHumanSupportTab from './pages/bot/BotHumanSupportTab'
import BotWebsiteSettingsTab from './pages/bot/BotWebsiteSettingsTab'
import BotImageAssetsTab from './pages/bot/BotImageAssetsTab'
import BotMenuListTab from './pages/bot/BotMenuListTab'
import BotNotificationsTab from './pages/bot/BotNotificationsTab'

function mostRecentBotId(bots: { bot_id: string; created_at: string }[]): string | null {
  if (bots.length === 0) return null
  const sorted = [...bots].sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
  return sorted[0]?.bot_id ?? null
}

function DashboardRedirect() {
  const { bots, loading, botsLoadedOnce } = useDashboardData()
  const mostRecentId = mostRecentBotId(bots)
  if (!botsLoadedOnce || loading) return null
  if (bots.length === 0) return <Navigate to="/create-bot" replace />
  if (mostRecentId) return <Navigate to={`/bots/${mostRecentId}/overview`} replace />
  return <Navigate to="/create-bot" replace />
}

export default function App() {
  const { isAuthenticated, isLoading: authLoading, loginWithRedirect } = useAuth0()

  if (authLoading) {
    return (
      <div className="app-shell">
        <main className="content">
          <div className="empty-panel">Loading authentication...</div>
        </main>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="app-shell">
        <main className="content">
          <div className="empty-panel">
            <div className="page-title">Sign in to Web AI Admin</div>
            <button className="primary" onClick={() => loginWithRedirect()}>
              Sign in
            </button>
          </div>
        </main>
      </div>
    )
  }

  return (
    <BackgroundTaskProvider>
      <DialogProvider>
        <DashboardDataProvider>
          <BrowserRouter>
            <BackgroundTaskIndicator />
            <Routes>
            <Route path="/create-bot/urls" element={<Navigate to="/create-bot/sources" replace />} />
            <Route path="/create-bot/*" element={<CreateBotLayout />} />
            <Route path="/" element={<DashboardLayout />}>
              <Route index element={<HomePage />} />
              <Route path="dashboard" element={<DashboardRedirect />} />
              <Route path="org" element={<OrgPage />} />
              <Route path="org/members" element={<Navigate to="/org" replace />} />
              <Route path="bots" element={<BotsPage />} />
              <Route path="bots/:botId" element={<BotDetailLayout />}>
                <Route index element={<Navigate to="overview" replace />} />
                <Route path="overview" element={<BotOverviewTab />} />
                <Route path="notifications" element={<BotNotificationsTab />} />
                <Route path="knowledge" element={<BotKnowledgeTab />} />
                <Route path="info-links" element={<Navigate to="../knowledge" replace />} />
                <Route path="image-assets" element={<BotImageAssetsTab />} />
                <Route path="menu-list" element={<BotMenuListTab />} />
                <Route path="sources" element={<Navigate to="knowledge" replace />} />
                <Route path="sources/new" element={<AddSourcePage />} />
                <Route path="design" element={<BotDesignTab />} />
                <Route path="welcome-messages" element={<BotWelcomeMessagesTab />} />
                <Route path="suggested-messages" element={<BotSuggestedMessagesTab />} />
                <Route path="human-support" element={<BotHumanSupportTab />} />

                <Route path="testing" element={<BotTestingTab />} />
                <Route path="conversations" element={<BotConversationsTab />} />
                <Route path="escalations" element={<BotEscalationsTab />} />
                <Route path="leads" element={<BotLeadsTab />} />
                <Route path="website" element={<BotWebsiteSettingsTab />} />
                <Route path="line" element={<BotLineSettingsTab />} />
                <Route path="settings" element={<BotSettingsTab />} />
              </Route>
              <Route path="users" element={<UsersPage />} />
              <Route path="domain" element={<DomainPage />} />
              <Route path="account" element={<AccountPage />} />
              <Route path="billing" element={<BillingPage />} />
              <Route path="settings" element={<SettingsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
          </BrowserRouter>
        </DashboardDataProvider>
      </DialogProvider>
    </BackgroundTaskProvider>
  )
}
