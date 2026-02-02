export type SidebarItem = {
  id: string
  label: string
  to: string
  children?: SidebarItem[]
}

export const sidebarConfig: SidebarItem[] = []
