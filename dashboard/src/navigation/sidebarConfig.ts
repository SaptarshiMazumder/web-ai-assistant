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
    children: [
      { id: 'org-overview', label: 'Overview', to: '/org' },
      { id: 'org-users', label: 'Users', to: '/org/members' },
    ],
  },
  {
    id: 'bots',
    label: 'Bots',
    to: '/bots',
  },
  {
    id: 'users',
    label: 'Users',
    to: '/users',
  },
  {
    id: 'domain',
    label: 'Domain',
    to: '/domain',
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
