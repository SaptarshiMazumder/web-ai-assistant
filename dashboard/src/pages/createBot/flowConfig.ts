/**
 * Single source of truth for the create-bot flow.
 * Add/remove/reorder steps here; routes and navigation are derived from this.
 */

export type CreateBotStepId = 'details' | 'hosting' | 'sources' | 'urls' | 'training' | 'widget' | 'embed'

export type CreateBotStep = {
  id: CreateBotStepId
  path: string
  label: string
  description: string
}

const BASE_STEPS: readonly CreateBotStep[] = [
  {
    id: 'details',
    path: '/create-bot',
    label: 'Name',
    description: 'Pick a name customers will see.',
  },
  {
    id: 'hosting',
    path: '/create-bot/hosting',
    label: 'Website',
    description: 'Tell us where your pages live.',
  },
  {
    id: 'sources',
    path: '/create-bot/sources',
    label: 'Add info',
    description: 'Add pages and PDFs to teach your helper.',
  },
  {
    id: 'training',
    path: '/create-bot/progress',
    label: 'Getting ready',
    description: 'We’ll start learning from what you added.',
  },
  {
    id: 'widget',
    path: '/create-bot/widget',
    label: 'Design widget',
    description: 'Pick colors and greeting messages.',
  },
  {
    id: 'embed',
    path: '/create-bot/embed',
    label: 'Add to website',
    description: 'Share the install code with your web person.',
  },
] as const

const SHARED_URL_STEP: CreateBotStep = {
  id: 'urls',
  path: '/create-bot/urls',
  label: 'Add links',
  description: 'Add links for pricing, hours, booking, contact, etc.',
} as const

export const CREATE_BOT_STEPS: readonly CreateBotStep[] = BASE_STEPS

export function getCreateBotSteps(contentHosting?: 'own' | 'shared' | null): readonly CreateBotStep[] {
  if (contentHosting === 'shared') {
    return [...BASE_STEPS.slice(0, 3), SHARED_URL_STEP, ...BASE_STEPS.slice(3)]
  }
  return BASE_STEPS
}

export function getCreateBotStepIndex(pathname: string, steps: ReadonlyArray<CreateBotStep> = CREATE_BOT_STEPS): number {
  const index = steps.findIndex((step) => step.path === pathname)
  return index >= 0 ? index : 0
}

export function getCreateBotNextPath(pathname: string, steps: ReadonlyArray<CreateBotStep> = CREATE_BOT_STEPS): string | null {
  const index = getCreateBotStepIndex(pathname, steps)
  const next = steps[index + 1]
  return next ? next.path : null
}

export function getCreateBotPrevPath(pathname: string, steps: ReadonlyArray<CreateBotStep> = CREATE_BOT_STEPS): string | null {
  const index = getCreateBotStepIndex(pathname, steps)
  const prev = steps[index - 1]
  return prev ? prev.path : null
}

export const CREATE_BOT_FIRST_PATH = BASE_STEPS[0].path
