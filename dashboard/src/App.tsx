import { useAuth0 } from '@auth0/auth0-react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { DashboardDataProvider } from './hooks/useDashboardData'
import DashboardLayout from './layouts/DashboardLayout'
import AccountPage from './pages/AccountPage'
import BotsPage from './pages/BotsPage'
import DashboardPage from './pages/DashboardPage'
import DomainPage from './pages/DomainPage'
import OrgMembersPage from './pages/OrgMembersPage'
import OrgPage from './pages/OrgPage'
import SettingsPage from './pages/SettingsPage'
import UsersPage from './pages/UsersPage'
import BotAnalyticsTab from './pages/bot/BotAnalyticsTab'
import BotDesignTab from './pages/bot/BotDesignTab'
import BotDetailLayout from './pages/bot/BotDetailLayout'
import BotKnowledgeTab from './pages/bot/BotKnowledgeTab'
import BotOverviewTab from './pages/bot/BotOverviewTab'
import BotSettingsTab from './pages/bot/BotSettingsTab'
import BotSourcesTab from './pages/bot/BotSourcesTab'

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
          <Route path="/" element={<DashboardLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="org" element={<OrgPage />} />
            <Route path="org/members" element={<OrgMembersPage />} />
            <Route path="bots" element={<BotsPage />} />
            <Route path="bots/:botId" element={<BotDetailLayout />}>
              <Route index element={<Navigate to="overview" replace />} />
              <Route path="overview" element={<BotOverviewTab />} />
              <Route path="knowledge" element={<BotKnowledgeTab />} />
              <Route path="sources" element={<BotSourcesTab />} />
              <Route path="design" element={<BotDesignTab />} />
              <Route path="settings" element={<BotSettingsTab />} />
              <Route path="analytics" element={<BotAnalyticsTab />} />
            </Route>
            <Route path="users" element={<UsersPage />} />
            <Route path="domain" element={<DomainPage />} />
            <Route path="account" element={<AccountPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </DashboardDataProvider>
  )
}
