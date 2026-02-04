import type { LucideIcon } from 'lucide-react'
import {
  LayoutDashboard,
  Bot,
  User,
  Users,
  CreditCard,
  Home,
  Plus,
  BookOpen,
  Palette,
  MessageCircle,
  FlaskConical,
  MessageSquare,
  Bell,
  Settings,
} from 'lucide-react'

export type PrimaryNavItem = {
  id: string
  label: string
  path: string
  icon: LucideIcon
}

export type SecondaryNavItem = {
  id: string
  label: string
  to: string
  icon: LucideIcon
}

export const primaryNavConfig: PrimaryNavItem[] = [
  { id: 'home', label: 'Home', path: '/', icon: Home },
  { id: 'bots', label: 'Bots', path: '/bots', icon: Bot },
  { id: 'account', label: 'Account', path: '/account', icon: User },
  { id: 'team', label: 'Team', path: '/org', icon: Users },
  { id: 'billing', label: 'Billing and subscriptions', path: '/billing', icon: CreditCard },
  { id: 'settings', label: 'Settings', path: '/settings', icon: Settings },
]

export const homeSecondaryItems: SecondaryNavItem[] = [
  { id: 'dashboard', label: 'Dashboard', to: '/', icon: LayoutDashboard },
]

export const accountSecondaryItems: SecondaryNavItem[] = [
  { id: 'profile', label: 'Profile', to: '/account', icon: User },
]

export const teamSecondaryItems: SecondaryNavItem[] = [
  { id: 'team', label: 'Team', to: '/org', icon: Users },
]

export const billingSecondaryItems: SecondaryNavItem[] = [
  { id: 'billing', label: 'Billing and subscriptions', to: '/billing', icon: CreditCard },
]

export const botsSecondaryItemsBase: SecondaryNavItem[] = [
  { id: 'all', label: 'All agents', to: '/bots', icon: Bot },
  { id: 'create', label: 'Create AI Agent', to: '/create-bot', icon: Plus },
]

export const botTabSecondaryItems = (botId: string) => [
  { id: 'overview', label: 'Overview', to: `/bots/${botId}/overview`, icon: LayoutDashboard },
  { id: 'knowledge', label: 'Knowledge', to: `/bots/${botId}/knowledge`, icon: BookOpen },
  { id: 'design', label: 'Design', to: `/bots/${botId}/design`, icon: Palette },
  { id: 'suggested', label: 'Suggestions', to: `/bots/${botId}/suggested-messages`, icon: MessageCircle },
  { id: 'testing', label: 'Testing', to: `/bots/${botId}/testing`, icon: FlaskConical },
  { id: 'conversations', label: 'Conversations', to: `/bots/${botId}/conversations`, icon: MessageSquare },
  { id: 'escalations', label: 'Escalations', to: `/bots/${botId}/escalations`, icon: Bell },
  { id: 'settings', label: 'Settings', to: `/bots/${botId}/settings`, icon: Settings },
]

export type SidebarItem = {
  id: string
  label: string
  to: string
  children?: SidebarItem[]
}

export const sidebarConfig: SidebarItem[] = []
