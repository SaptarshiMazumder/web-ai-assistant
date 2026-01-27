export type SidebarItem = {
  id: string
  label: string
  to: string
  children?: SidebarItem[]
}

export const sidebarConfig: SidebarItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    to: '/',
  },
  {
    id: 'org',
    label: 'Org',
    to: '/org',
  },
  {
    id: 'bots',
    label: 'Bots',
    to: '/bots',
  },
  {
    id: 'account',
    label: 'Account',
    to: '/account',
  },
  {
    id: 'settings',
    label: 'Settings',
    to: '/settings',
  },
]
