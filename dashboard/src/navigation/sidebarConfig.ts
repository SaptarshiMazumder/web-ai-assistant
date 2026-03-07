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
  Globe,
  UsersRound,
  History,
  Briefcase,
  AlertCircle,
  UtensilsCrossed,
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

/** Knowledge tab IDs: config-driven per platform. */
const KNOWLEDGE_TAB_IMAGE = 'image-assets'
const KNOWLEDGE_TAB_MENU = 'menu-list'

/** Map config value (image|menu) to tab id. */
const CONFIG_TO_TAB: Record<string, string> = {
  image: KNOWLEDGE_TAB_IMAGE,
  menu: KNOWLEDGE_TAB_MENU,
}

const ALL_KNOWLEDGE_TABS: Omit<SecondaryNavItem, 'to'>[] = [
  { id: KNOWLEDGE_TAB_IMAGE, label: 'nav.imageAssets', icon: Briefcase, header: 'nav.agentKnowledge' },
  { id: KNOWLEDGE_TAB_MENU, label: 'nav.menuList', icon: UtensilsCrossed, header: 'nav.agentKnowledge' },
]

export function botTabSecondaryItems(
  botId: string,
  knowledgeTabs?: string[] | null
): SecondaryNavItem[] {
  const tabs = knowledgeTabs && knowledgeTabs.length > 0
    ? knowledgeTabs
    : ['image']
  const allowedIds = new Set(tabs.map((t) => CONFIG_TO_TAB[String(t).toLowerCase()]).filter(Boolean))

  const knowledgeItems = ALL_KNOWLEDGE_TABS
    .filter((item) => allowedIds.has(item.id))
    .map((item) => ({ ...item, to: `/bots/${botId}/${item.id}` }))

  return [
    // Top-level (no header)
    { id: 'overview', label: 'nav.overview', to: `/bots/${botId}/overview`, icon: LayoutDashboard },
    { id: 'notifications', label: 'nav.notifications', to: `/bots/${botId}/notifications`, icon: Bell },

    // Sources section (was Knowledge)
    { id: 'knowledge', label: 'nav.sources', to: `/bots/${botId}/knowledge`, icon: BookOpen, header: 'nav.agentKnowledge' },

    ...knowledgeItems,

    // Appearance section (was Design)
    { id: 'design', label: 'nav.appearance', to: `/bots/${botId}/design`, icon: Palette, header: 'nav.agentDesign' },
    { id: 'suggested-messages', label: 'nav.suggestedMessages', to: `/bots/${botId}/suggested-messages`, icon: MessageSquare, header: 'nav.agentDesign' },
    { id: 'human-support', label: 'nav.humanSupport', to: `/bots/${botId}/human-support`, icon: AlertCircle, header: 'nav.agentDesign' },
    { id: 'testing', label: 'nav.testing', to: `/bots/${botId}/testing`, icon: FlaskConical, header: 'nav.agentDesign' },

    // Install section
    { id: 'website', label: 'nav.website', to: `/bots/${botId}/website`, icon: Globe, header: 'nav.install' },
    { id: 'line', label: 'nav.line', to: `/bots/${botId}/line`, icon: LineIcon, header: 'nav.install' },

    // Contacts section
    { id: 'conversations', label: 'nav.conversationsHistory', to: `/bots/${botId}/conversations`, icon: History, header: 'nav.contacts' },
    { id: 'escalations', label: 'nav.escalatedConversations', to: `/bots/${botId}/escalations`, icon: AlertCircle, header: 'nav.contacts' },
    { id: 'leads', label: 'nav.leads', to: `/bots/${botId}/leads`, icon: UsersRound, header: 'nav.contacts' },

    // Settings (standalone with separator)
    { id: 'settings', label: 'nav.settings', to: `/bots/${botId}/settings`, icon: Settings, separator: true },
  ]
}

export type SidebarItem = {
  id: string
  label: string
  to: string
  children?: SidebarItem[]
}

export const sidebarConfig: SidebarItem[] = []
