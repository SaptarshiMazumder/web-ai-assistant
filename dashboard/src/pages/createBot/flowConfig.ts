/**
 * Single source of truth for the create-bot flow.
 * Add/remove/reorder steps here; routes and navigation are derived from this.
 */

import autographIcon from '../../assets/icons8/autograph.png'
import googleDocsIcon from '../../assets/icons8/google-docs.png'
import learningIcon from '../../assets/icons8/learning.png'
import paintPaletteIcon from '../../assets/icons8/paint-palette.png'
import googleCodeIcon from '../../assets/icons8/google-code.png'

export type CreateBotStepId = 'details' | 'sources' | 'urls' | 'additional-sources' | 'training' | 'topics' | 'widget' | 'embed'

export type CreateBotStep = {
  id: CreateBotStepId
  path: string
  labelKey: string
  label: string
  descriptionKey: string
  description: string
  icon: string
  iconUrl?: string
}

const BASE_STEPS: readonly CreateBotStep[] = [
  {
    id: 'details',
    path: '/create-bot',
    labelKey: 'createBot.stepDetailsLabel',
    label: 'Name',
    descriptionKey: 'createBot.stepDetailsDescription',
    description: 'Pick a name customers will see.',
    icon: 'badge',
    iconUrl: autographIcon,
  },
  {
    id: 'sources',
    path: '/create-bot/sources',
    labelKey: 'createBot.stepSourcesLabel',
    label: 'Website',
    descriptionKey: 'createBot.stepSourcesDescription',
    description: 'Add sources from your website.',
    icon: 'source',
    iconUrl: googleDocsIcon,
  },
  {
    id: 'additional-sources',
    path: '/create-bot/additional-sources',
    labelKey: 'createBot.stepAdditionalSourcesLabel',
    label: 'More sources',
    descriptionKey: 'createBot.stepAdditionalSourcesDescription',
    description: 'Add PDFs, text docs, or custom content.',
    icon: 'library_add',
    iconUrl: googleDocsIcon,
  },
  {
    id: 'training',
    path: '/create-bot/progress',
    labelKey: 'createBot.stepTrainingLabel',
    label: 'Agent Training',
    descriptionKey: 'createBot.stepTrainingDescription',
    description: "We'll start learning from what you added.",
    icon: 'model_training',
    iconUrl: learningIcon,
  },
  {
    id: 'widget',
    path: '/create-bot/widget',
    labelKey: 'createBot.stepWidgetLabel',
    label: 'Design widget',
    descriptionKey: 'createBot.stepWidgetDescription',
    description: 'Customize agent\'s appearance.',
    icon: 'palette',
    iconUrl: paintPaletteIcon,
  },
  {
    id: 'embed',
    path: '/create-bot/embed',
    labelKey: 'createBot.stepEmbedLabel',
    label: 'Install Agent',
    descriptionKey: 'createBot.stepEmbedDescription',
    description: 'Make your agent available to your customers.',
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
