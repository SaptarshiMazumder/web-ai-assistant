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
  MessageSquare,
  FlaskConical,

  Bell,
  Settings,
  Instagram,
  Globe,
  UserCircle,
  UsersRound,
  History,
  Briefcase,
  AlertCircle,
  Link2,
} from 'lucide-react'
import { LineIcon } from '../assets/icons/LineIcon'

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
  header?: string // Optional category header
  separator?: boolean // Optional separator before item
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

export const botTabSecondaryItems = (botId: string): SecondaryNavItem[] => [
  // Top-level (no header)
  { id: 'overview', label: 'Overview', to: `/bots/${botId}/overview`, icon: LayoutDashboard },
  { id: 'notifications', label: 'Notifications', to: `/bots/${botId}/notifications`, icon: Bell },

  // Sources section (was Knowledge)
  { id: 'knowledge', label: 'Sources', to: `/bots/${botId}/knowledge`, icon: BookOpen, header: 'Agent Knowledge' },
  { id: 'info-links', label: 'Info Links', to: `/bots/${botId}/info-links`, icon: Link2, header: 'Agent Knowledge' },

  { id: 'image-assets', label: 'Image Assets', to: `/bots/${botId}/image-assets`, icon: Briefcase, header: 'Agent Knowledge' },

  // Appearance section (was Design)
  { id: 'design', label: 'Appearance', to: `/bots/${botId}/design`, icon: Palette, header: 'Agent Design' },
  { id: 'suggested-messages', label: 'Suggested Messages', to: `/bots/${botId}/suggested-messages`, icon: MessageSquare, header: 'Agent Design' },
  { id: 'persona', label: 'Personas', to: `/bots/${botId}/persona`, icon: UserCircle, header: 'Agent Design' },
  { id: 'testing', label: 'Testing', to: `/bots/${botId}/testing`, icon: FlaskConical, header: 'Agent Design' },

  // Install section
  { id: 'website', label: 'Website', to: `/bots/${botId}/website`, icon: Globe, header: 'Install' },
  { id: 'instagram', label: 'Instagram', to: `/bots/${botId}/instagram`, icon: Instagram, header: 'Install' },
  { id: 'line', label: 'LINE', to: `/bots/${botId}/line`, icon: LineIcon, header: 'Install' },

  // Contacts section
  { id: 'conversations', label: 'Conversations History', to: `/bots/${botId}/conversations`, icon: History, header: 'Contacts' },
  { id: 'escalations', label: 'Escalations', to: `/bots/${botId}/escalations`, icon: AlertCircle, header: 'Contacts' },
  { id: 'leads', label: 'Leads', to: `/bots/${botId}/leads`, icon: UsersRound, header: 'Contacts' },

  // Settings (standalone with separator)
  { id: 'settings', label: 'Settings', to: `/bots/${botId}/settings`, icon: Settings, separator: true },
]

export type SidebarItem = {
  id: string
  label: string
  to: string
  children?: SidebarItem[]
}

export const sidebarConfig: SidebarItem[] = []
