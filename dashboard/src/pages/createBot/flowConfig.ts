/**
 * Single source of truth for the create-bot flow.
 * Add/remove/reorder steps here; routes and navigation are derived from this.
 */

export const CREATE_BOT_STEPS = [
  {
    id: 'details',
    path: '/create-bot',
    label: 'Name + Website',
    description: 'Give your bot a name and the site to learn from.',
  },
  {
    id: 'urls',
    path: '/create-bot/urls',
    label: 'Select URLs',
    description: 'Choose which pages should be included.',
  },
  {
    id: 'training',
    path: '/create-bot/progress',
    label: 'Training',
    description: 'We will start processing your sources.',
  },
  {
    id: 'widget',
    path: '/create-bot/widget',
    label: 'Design widget',
    description: 'Customize how the chat widget looks on your site.',
  },
  {
    id: 'embed',
    path: '/create-bot/embed',
    label: 'Add script',
    description: 'Add the script to your website to show the chatbot.',
  },
] as const

export type CreateBotStepId = (typeof CREATE_BOT_STEPS)[number]['id']

export function getCreateBotStepIndex(pathname: string): number {
  const index = CREATE_BOT_STEPS.findIndex((step) => step.path === pathname)
  return index >= 0 ? index : 0
}

export function getCreateBotNextPath(pathname: string): string | null {
  const index = getCreateBotStepIndex(pathname)
  const next = CREATE_BOT_STEPS[index + 1]
  return next ? next.path : null
}

export function getCreateBotPrevPath(pathname: string): string | null {
  const index = getCreateBotStepIndex(pathname)
  const prev = CREATE_BOT_STEPS[index - 1]
  return prev ? prev.path : null
}

export const CREATE_BOT_FIRST_PATH = CREATE_BOT_STEPS[0].path
