import { useAuth0 } from '@auth0/auth0-react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { DashboardDataProvider, useDashboardData } from './hooks/useDashboardData'
import DashboardLayout from './layouts/DashboardLayout'
import AccountPage from './pages/AccountPage'
import BillingPage from './pages/BillingPage'
import BotsPage from './pages/BotsPage'
import HomePage from './pages/HomePage'
import DomainPage from './pages/DomainPage'
import OrgPage from './pages/OrgPage'
import SettingsPage from './pages/SettingsPage'
import UsersPage from './pages/UsersPage'
import CreateBotDetailsPage from './pages/createBot/CreateBotDetailsPage'
import CreateBotEmbedPage from './pages/createBot/CreateBotEmbedPage'
import CreateBotLayout from './pages/createBot/CreateBotLayout'
import CreateBotProgressPage from './pages/createBot/CreateBotProgressPage'
import CreateBotUrlsPage from './pages/createBot/CreateBotUrlsPage'
import CreateBotAdditionalSourcesPage from './pages/createBot/CreateBotAdditionalSourcesPage'
import CreateBotWidgetPage from './pages/createBot/CreateBotWidgetPage'
import BotDesignTab from './pages/bot/BotDesignTab'
import BotDetailLayout from './pages/bot/BotDetailLayout'
import AddSourcePage from './pages/bot/AddSourcePage'
import BotKnowledgeTab from './pages/bot/BotKnowledgeTab'
import BotOverviewTab from './pages/bot/BotOverviewTab'
import BotSettingsTab from './pages/bot/BotSettingsTab'
import BotTestingTab from './pages/bot/BotTestingTab'
import BotConversationsTab from './pages/bot/BotConversationsTab'
import BotEscalationsTab from './pages/bot/BotEscalationsTab'
import BotInstagramSettingsTab from './pages/bot/BotInstagramSettingsTab'
import BotLineSettingsTab from './pages/bot/BotLineSettingsTab'

import BotSuggestedMessagesTab from './pages/bot/BotSuggestedMessagesTab'
import BotPersonaTab from './pages/bot/BotPersonaTab'
import BotLeadsTab from './pages/bot/BotLeadsTab'
import BotWebsiteSettingsTab from './pages/bot/BotWebsiteSettingsTab'
import BotBusinessAssetsTab from './pages/bot/BotBusinessAssetsTab'
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
    <DashboardDataProvider>
      <BrowserRouter>
        <Routes>
          {/* Create-bot step order/paths: see flowConfig.ts. When adding a step, add entry there and a Route here. */}
          <Route path="/create-bot" element={<CreateBotLayout />}>
            <Route index element={<CreateBotDetailsPage />} />
            <Route path="sources" element={<CreateBotUrlsPage />} />
            <Route path="urls" element={<Navigate to="/create-bot/sources" replace />} />
            <Route path="additional-sources" element={<CreateBotAdditionalSourcesPage />} />
            <Route path="progress" element={<CreateBotProgressPage />} />
            <Route path="widget" element={<CreateBotWidgetPage />} />
            <Route path="embed" element={<CreateBotEmbedPage />} />
            <Route path="*" element={<Navigate to="/create-bot" replace />} />
          </Route>
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
              <Route path="business-assets" element={<BotBusinessAssetsTab />} />
              <Route path="sources" element={<Navigate to="knowledge" replace />} />
              <Route path="sources/new" element={<AddSourcePage />} />
              <Route path="design" element={<BotDesignTab />} />
              <Route path="suggested-messages" element={<BotSuggestedMessagesTab />} />
              <Route path="persona" element={<BotPersonaTab />} />

              <Route path="testing" element={<BotTestingTab />} />
              <Route path="conversations" element={<BotConversationsTab />} />
              <Route path="escalations" element={<BotEscalationsTab />} />
              <Route path="leads" element={<BotLeadsTab />} />
              <Route path="website" element={<BotWebsiteSettingsTab />} />
              <Route path="instagram" element={<BotInstagramSettingsTab />} />
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
  )
}
