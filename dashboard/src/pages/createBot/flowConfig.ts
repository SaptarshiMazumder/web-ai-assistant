/**
 * Single source of truth for the create-bot flow.
 * Add/remove/reorder steps here; routes and navigation are derived from this.
 */

import autographIcon from '../../assets/icons8/autograph.png'
import internetBrowserIcon from '../../assets/icons8/internet-browser.png'
import googleDocsIcon from '../../assets/icons8/google-docs.png'
import chainIcon from '../../assets/icons8/chain.png'
import learningIcon from '../../assets/icons8/learning.png'
import paintPaletteIcon from '../../assets/icons8/paint-palette.png'
import googleCodeIcon from '../../assets/icons8/google-code.png'

export type CreateBotStepId = 'details' | 'hosting' | 'sources' | 'urls' | 'training' | 'topics' | 'widget' | 'embed'

export type CreateBotStep = {
  id: CreateBotStepId
  path: string
  label: string
  description: string
  icon: string
  iconUrl?: string
}

const BASE_STEPS: readonly CreateBotStep[] = [
  {
    id: 'details',
    path: '/create-bot',
    label: 'Name',
    description: 'Pick a name customers will see.',
    icon: 'badge',
    iconUrl: autographIcon,
  },
  {
    id: 'hosting',
    path: '/create-bot/hosting',
    label: 'Website',
    description: 'Tell us where your pages live.',
    icon: 'language',
    iconUrl: internetBrowserIcon,
  },
  {
    id: 'sources',
    path: '/create-bot/sources',
    label: 'Pages and files',
    description: 'Pick website pages and upload PDFs to teach your assistant.',
    icon: 'source',
    iconUrl: googleDocsIcon,
  },
  {
    id: 'training',
    path: '/create-bot/progress',
    label: 'Getting ready',
    description: 'We\u2019ll start learning from what you added.',
    icon: 'model_training',
    iconUrl: learningIcon,
  },
  {
    id: 'topics',
    path: '/create-bot/topics',
    label: 'Helpful links',
    description: 'Optional: add important links for common customer questions.',
    icon: 'link',
    iconUrl: chainIcon,
  },
  {
    id: 'widget',
    path: '/create-bot/widget',
    label: 'Design widget',
    description: 'Pick colors and greeting messages.',
    icon: 'palette',
    iconUrl: paintPaletteIcon,
  },
  {
    id: 'embed',
    path: '/create-bot/embed',
    label: 'Add to website',
    description: 'Share the install code with your web person.',
    icon: 'code',
    iconUrl: googleCodeIcon,
  },
] as const

export const CREATE_BOT_STEPS: readonly CreateBotStep[] = BASE_STEPS

export function getCreateBotSteps(): readonly CreateBotStep[] {
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
