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
  { id: 'home', label: 'nav.home', path: '/', icon: Home },
  { id: 'bots', label: 'nav.bots', path: '/bots', icon: Bot },
  { id: 'account', label: 'nav.account', path: '/account', icon: User },
  { id: 'team', label: 'nav.team', path: '/org', icon: Users },
  { id: 'billing', label: 'nav.billing', path: '/billing', icon: CreditCard },
  { id: 'settings', label: 'nav.settings', path: '/settings', icon: Settings },
]

export const homeSecondaryItems: SecondaryNavItem[] = [
  { id: 'dashboard', label: 'nav.dashboard', to: '/', icon: LayoutDashboard },
]

export const accountSecondaryItems: SecondaryNavItem[] = [
  { id: 'profile', label: 'nav.profile', to: '/account', icon: User },
]

export const teamSecondaryItems: SecondaryNavItem[] = [
  { id: 'team', label: 'nav.team', to: '/org', icon: Users },
]

export const billingSecondaryItems: SecondaryNavItem[] = [
  { id: 'billing', label: 'nav.billing', to: '/billing', icon: CreditCard },
]

export const botsSecondaryItemsBase: SecondaryNavItem[] = [
  { id: 'all', label: 'nav.allAgents', to: '/bots', icon: Bot },
  { id: 'create', label: 'nav.createAgent', to: '/create-bot', icon: Plus },
]

export const botTabSecondaryItems = (botId: string): SecondaryNavItem[] => [
  // Top-level (no header)
  { id: 'overview', label: 'nav.overview', to: `/bots/${botId}/overview`, icon: LayoutDashboard },
  { id: 'notifications', label: 'nav.notifications', to: `/bots/${botId}/notifications`, icon: Bell },

  // Sources section (was Knowledge)
  { id: 'knowledge', label: 'nav.sources', to: `/bots/${botId}/knowledge`, icon: BookOpen, header: 'nav.agentKnowledge' },
  { id: 'info-links', label: 'nav.infoLinks', to: `/bots/${botId}/info-links`, icon: Link2, header: 'nav.agentKnowledge' },

  { id: 'image-assets', label: 'nav.imageAssets', to: `/bots/${botId}/image-assets`, icon: Briefcase, header: 'nav.agentKnowledge' },

  // Appearance section (was Design)
  { id: 'design', label: 'nav.appearance', to: `/bots/${botId}/design`, icon: Palette, header: 'nav.agentDesign' },
  { id: 'suggested-messages', label: 'nav.suggestedMessages', to: `/bots/${botId}/suggested-messages`, icon: MessageSquare, header: 'nav.agentDesign' },
  { id: 'persona', label: 'nav.personas', to: `/bots/${botId}/persona`, icon: UserCircle, header: 'nav.agentDesign' },
  { id: 'testing', label: 'nav.testing', to: `/bots/${botId}/testing`, icon: FlaskConical, header: 'nav.agentDesign' },

  // Install section
  { id: 'website', label: 'nav.website', to: `/bots/${botId}/website`, icon: Globe, header: 'nav.install' },
  { id: 'instagram', label: 'nav.instagram', to: `/bots/${botId}/instagram`, icon: Instagram, header: 'nav.install' },
  { id: 'line', label: 'nav.line', to: `/bots/${botId}/line`, icon: LineIcon, header: 'nav.install' },

  // Contacts section
  { id: 'conversations', label: 'nav.conversationsHistory', to: `/bots/${botId}/conversations`, icon: History, header: 'nav.contacts' },
  { id: 'escalations', label: 'nav.escalatedConversations', to: `/bots/${botId}/escalations`, icon: AlertCircle, header: 'nav.contacts' },
  { id: 'leads', label: 'nav.leads', to: `/bots/${botId}/leads`, icon: UsersRound, header: 'nav.contacts' },

  // Settings (standalone with separator)
  { id: 'settings', label: 'nav.settings', to: `/bots/${botId}/settings`, icon: Settings, separator: true },
]

export type SidebarItem = {
  id: string
  label: string
  to: string
  children?: SidebarItem[]
}

export const sidebarConfig: SidebarItem[] = []
