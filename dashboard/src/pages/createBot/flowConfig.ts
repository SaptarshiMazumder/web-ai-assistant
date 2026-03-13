import autographIcon from '../../assets/icons8/autograph.png'
import chainIcon from '../../assets/icons8/chain.png'
import googleDocsIcon from '../../assets/icons8/google-docs.png'
import learningIcon from '../../assets/icons8/learning.png'
import paintPaletteIcon from '../../assets/icons8/paint-palette.png'
import googleCodeIcon from '../../assets/icons8/google-code.png'

export type CreateBotStepGroupId = string

export type CreateBotScreenId = string

export type CreateBotScreenComponent =
  | 'details'
  | 'source_urls'
  | 'additional_sources'
  | 'training_progress'
  | 'image_extraction_permission'
  | 'suggested_messages'
  | 'widget_design'
  | 'embed_install'
  | 'action_destination_url'

export type CreateBotScreenVisibility = {
  businessTypes?: string[]
  requiresSelectedReservationPlatform?: boolean
  reservationPlatformIds?: string[]
  requiresWorkflowSteps?: string[]
}

export type CreateBotStepGroup = {
  id: CreateBotStepGroupId
  label: string
  description: string
  icon: string
  iconUrl?: string
}

export type CreateBotScreen = {
  id: CreateBotScreenId
  path: string
  stepGroupId: CreateBotStepGroupId
  component: CreateBotScreenComponent
  actionKey?: string
  title?: string
  subtitle?: string
  fieldLabel?: string
  fieldPlaceholder?: string
  fieldHelper?: string
  fallbackNotice?: string
  visibility?: CreateBotScreenVisibility
}

export type CreateBotFlowConfig = {
  stepGroups: CreateBotStepGroup[]
  screenDefinitions: Record<string, CreateBotScreen>
  screenOrder: string[]
}

export type CreateBotFlowState = {
  businessType: string
  reservationPlatform: string
  workflowSteps: string[]
}

type RawCreateBotFlowConfig = {
  step_groups?: Array<{
    id?: string
    label?: string
    description?: string
  }>
  screen_definitions?: Record<string, {
    id?: string
    path?: string
    step_group?: string
    component?: string
    action_key?: string
    title?: string
    subtitle?: string
    field_label?: string
    field_placeholder?: string
    field_helper?: string
    fallback_notice?: string
    visibility?: {
      business_types?: string[]
      requires_selected_reservation_platform?: boolean
      reservation_platform_ids?: string[]
      requires_workflow_steps?: string[]
    }
  }>
  screen_order?: string[]
}

const STEP_GROUP_DECORATIONS: Record<string, { icon: string; iconUrl: string }> = {
  details: { icon: 'badge', iconUrl: autographIcon },
  sources: { icon: 'source', iconUrl: googleDocsIcon },
  additional_sources: { icon: 'library_add', iconUrl: googleDocsIcon },
  training: { icon: 'model_training', iconUrl: learningIcon },
  suggested_messages: { icon: 'quickreply', iconUrl: chainIcon },
  widget: { icon: 'palette', iconUrl: paintPaletteIcon },
  embed: { icon: 'code', iconUrl: googleCodeIcon },
}

const DEFAULT_STEP_GROUP_DECORATION = STEP_GROUP_DECORATIONS.details

const DEFAULT_STEP_GROUPS: ReadonlyArray<Omit<CreateBotStepGroup, 'icon' | 'iconUrl'>> = [
  {
    id: 'details',
    label: 'Name',
    description: 'Pick a name customers will see.',
  },
  {
    id: 'sources',
    label: 'Website',
    description: 'Add sources from your website.',
  },
  {
    id: 'additional_sources',
    label: 'More sources',
    description: 'Add PDFs, text docs, or custom content.',
  },
  {
    id: 'training',
    label: 'Agent Training',
    description: "We'll start learning from what you added.",
  },
  {
    id: 'suggested_messages',
    label: 'Suggested messages',
    description: 'Set quick actions users can tap first.',
  },
  {
    id: 'widget',
    label: 'Design appearance',
    description: "Customize your agent's Website and LINE appearance.",
  },
  {
    id: 'embed',
    label: 'Install Agent',
    description: 'Make your agent available to your customers.',
  },
] as const

const DEFAULT_SCREEN_DEFINITIONS: Record<string, CreateBotScreen> = {
  details: {
    id: 'details',
    path: '',
    stepGroupId: 'details',
    component: 'details',
  },
  sources: {
    id: 'sources',
    path: 'sources',
    stepGroupId: 'sources',
    component: 'source_urls',
  },
  additional_sources: {
    id: 'additional_sources',
    path: 'additional-sources',
    stepGroupId: 'additional_sources',
    component: 'additional_sources',
  },
  reservation_destination: {
    id: 'reservation_destination',
    path: 'reservation-destination',
    stepGroupId: 'additional_sources',
    component: 'action_destination_url',
    actionKey: 'reservation',
    title: 'Choose where reservation taps should go',
    subtitle: 'You can keep the selected platform URL, or set a customer-facing destination URL of your own.',
    fieldLabel: 'Customer-facing reservation URL',
    fieldPlaceholder: 'https://your-restaurant.com/reserve',
    fieldHelper: 'Optional. If set, this is the reservation link customers receive in chat and action buttons.',
    fallbackNotice: 'Skip this to keep sending customers to {platform_label}: {platform_url}',
    visibility: {
      businessTypes: ['restaurant'],
      requiresSelectedReservationPlatform: true,
    },
  },
  training: {
    id: 'training',
    path: 'progress',
    stepGroupId: 'training',
    component: 'training_progress',
  },
  image_extraction_permission: {
    id: 'image_extraction_permission',
    path: 'image-extraction-permission',
    stepGroupId: 'training',
    component: 'image_extraction_permission',
    visibility: {
      requiresWorkflowSteps: ['asset_extraction'],
    },
  },
  suggested_messages: {
    id: 'suggested_messages',
    path: 'suggested-messages',
    stepGroupId: 'suggested_messages',
    component: 'suggested_messages',
  },
  widget: {
    id: 'widget',
    path: 'widget',
    stepGroupId: 'widget',
    component: 'widget_design',
  },
  embed: {
    id: 'embed',
    path: 'embed',
    stepGroupId: 'embed',
    component: 'embed_install',
  },
}

const DEFAULT_SCREEN_ORDER: readonly string[] = [
  'details',
  'sources',
  'additional_sources',
  'reservation_destination',
  'image_extraction_permission',
  'training',
  'suggested_messages',
  'widget',
  'embed',
]

function buildDecoratedStepGroup(base: Omit<CreateBotStepGroup, 'icon' | 'iconUrl'>): CreateBotStepGroup {
  const decoration = STEP_GROUP_DECORATIONS[base.id] || DEFAULT_STEP_GROUP_DECORATION
  return {
    ...base,
    icon: decoration.icon,
    iconUrl: decoration.iconUrl,
  }
}

function normalizeStepGroupId(value: string): CreateBotStepGroupId | null {
  const normalized = value.trim()
  return normalized || null
}

function normalizeScreenId(value: string): CreateBotScreenId | null {
  const normalized = value.trim()
  return normalized || null
}

function normalizeComponent(value: string): CreateBotScreenComponent | null {
  const normalized = value.trim() as CreateBotScreenComponent
  if (
    normalized === 'details' ||
    normalized === 'source_urls' ||
    normalized === 'additional_sources' ||
    normalized === 'training_progress' ||
    normalized === 'image_extraction_permission' ||
    normalized === 'suggested_messages' ||
    normalized === 'widget_design' ||
    normalized === 'embed_install' ||
    normalized === 'action_destination_url'
  ) {
    return normalized
  }
  return null
}

export function buildCreateBotPath(path: string): string {
  const trimmed = (path || '').trim().replace(/^\/+|\/+$/g, '')
  return trimmed ? `/create-bot/${trimmed}` : '/create-bot'
}

export function getCreateBotRelativePath(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, '')
  if (normalized === '/create-bot' || normalized === '') return ''
  if (normalized.startsWith('/create-bot/')) return normalized.slice('/create-bot/'.length)
  return ''
}

export function getDefaultCreateBotFlowConfig(): CreateBotFlowConfig {
  const stepGroups = DEFAULT_STEP_GROUPS.map(buildDecoratedStepGroup)
  const screenDefinitions = Object.fromEntries(
    Object.entries(DEFAULT_SCREEN_DEFINITIONS).map(([screenId, screen]) => [screenId, { ...screen }])
  )
  return {
    stepGroups,
    screenDefinitions,
    screenOrder: [...DEFAULT_SCREEN_ORDER],
  }
}

export function normalizeCreateBotFlowConfig(raw: RawCreateBotFlowConfig | null | undefined): CreateBotFlowConfig {
  const fallback = getDefaultCreateBotFlowConfig()
  if (!raw || typeof raw !== 'object') return fallback

  const rawStepGroups = Array.isArray(raw.step_groups) ? raw.step_groups : []
  const stepGroups = rawStepGroups
    .map((group) => {
      const groupId = normalizeStepGroupId(String(group?.id || '').trim())
      if (!groupId) return null
      return buildDecoratedStepGroup({
        id: groupId,
        label: String(group?.label || fallback.stepGroups.find((item) => item.id === groupId)?.label || groupId),
        description: String(
          group?.description || fallback.stepGroups.find((item) => item.id === groupId)?.description || ''
        ),
      })
    })
    .filter(Boolean) as CreateBotStepGroup[]

  const resolvedStepGroups = stepGroups.length > 0 ? stepGroups : fallback.stepGroups
  const resolvedStepGroupIds = new Set(resolvedStepGroups.map((group) => group.id))

  const rawDefinitions = raw.screen_definitions && typeof raw.screen_definitions === 'object' ? raw.screen_definitions : {}
  const hasRawDefinitions = Object.keys(rawDefinitions).length > 0
  const screenDefinitions: Record<string, CreateBotScreen> = hasRawDefinitions ? {} : { ...fallback.screenDefinitions }
  for (const [rawScreenId, rawScreen] of Object.entries(rawDefinitions)) {
    const screenId = normalizeScreenId(String(rawScreenId || '').trim())
    if (!screenId || !rawScreen || typeof rawScreen !== 'object') continue
    const fallbackScreen = fallback.screenDefinitions[screenId]
    const stepGroupId =
      normalizeStepGroupId(String(rawScreen.step_group || '').trim()) || fallbackScreen?.stepGroupId || null
    if (!stepGroupId || !resolvedStepGroupIds.has(stepGroupId)) continue
    const component =
      normalizeComponent(String(rawScreen.component || '').trim()) || fallbackScreen?.component || null
    if (!component) continue
    const businessTypes = Array.isArray(rawScreen.visibility?.business_types)
      ? rawScreen.visibility.business_types.map((item) => String(item).trim().toLowerCase()).filter(Boolean)
      : fallbackScreen?.visibility?.businessTypes
    const reservationPlatformIds = Array.isArray(rawScreen.visibility?.reservation_platform_ids)
      ? rawScreen.visibility.reservation_platform_ids.map((item) => String(item).trim().toLowerCase()).filter(Boolean)
      : fallbackScreen?.visibility?.reservationPlatformIds
    const requiresWorkflowSteps = Array.isArray(rawScreen.visibility?.requires_workflow_steps)
      ? rawScreen.visibility.requires_workflow_steps.map((item) => String(item).trim().toLowerCase()).filter(Boolean)
      : fallbackScreen?.visibility?.requiresWorkflowSteps
    screenDefinitions[screenId] = {
      ...(fallbackScreen || {
        id: screenId,
        path: '',
        stepGroupId,
        component,
      }),
      id: screenId,
      path: String(rawScreen.path ?? fallbackScreen?.path ?? ''),
      stepGroupId,
      component,
      actionKey: String(rawScreen.action_key || fallbackScreen?.actionKey || '').trim() || undefined,
      title: String(rawScreen.title || fallbackScreen?.title || '').trim() || undefined,
      subtitle: String(rawScreen.subtitle || fallbackScreen?.subtitle || '').trim() || undefined,
      fieldLabel: String(rawScreen.field_label || fallbackScreen?.fieldLabel || '').trim() || undefined,
      fieldPlaceholder: String(rawScreen.field_placeholder || fallbackScreen?.fieldPlaceholder || '').trim() || undefined,
      fieldHelper: String(rawScreen.field_helper || fallbackScreen?.fieldHelper || '').trim() || undefined,
      fallbackNotice: String(rawScreen.fallback_notice || fallbackScreen?.fallbackNotice || '').trim() || undefined,
      visibility: {
        businessTypes,
        requiresSelectedReservationPlatform:
          typeof rawScreen.visibility?.requires_selected_reservation_platform === 'boolean'
            ? rawScreen.visibility.requires_selected_reservation_platform
            : fallbackScreen?.visibility?.requiresSelectedReservationPlatform,
        reservationPlatformIds,
        requiresWorkflowSteps,
      },
    }
  }

  const rawOrder = Array.isArray(raw.screen_order) ? raw.screen_order : []
  const normalizedRawOrder = rawOrder
    .map((item) => normalizeScreenId(String(item || '').trim()))
    .filter(Boolean)
    .filter((screenId, index, allIds) => allIds.indexOf(screenId) === index)
    .filter((screenId): screenId is string => Boolean(screenId && screenDefinitions[screenId]))

  const derivedOrder = hasRawDefinitions
    ? Object.keys(screenDefinitions)
    : [...fallback.screenOrder]

  return {
    stepGroups: resolvedStepGroups,
    screenDefinitions,
    screenOrder: normalizedRawOrder.length > 0 ? normalizedRawOrder : derivedOrder,
  }
}

export function isCreateBotScreenVisible(screen: CreateBotScreen, state: CreateBotFlowState): boolean {
  const visibility = screen.visibility
  if (!visibility) return true
  const businessTypes = visibility.businessTypes || []
  if (businessTypes.length > 0) {
    const currentBusinessType = String(state.businessType || '').trim().toLowerCase()
    if (!currentBusinessType || !businessTypes.includes(currentBusinessType)) return false
  }
  if (visibility.requiresSelectedReservationPlatform && !String(state.reservationPlatform || '').trim()) {
    return false
  }
  const reservationPlatformIds = visibility.reservationPlatformIds || []
  if (reservationPlatformIds.length > 0) {
    const currentPlatform = String(state.reservationPlatform || '').trim().toLowerCase()
    if (!currentPlatform || !reservationPlatformIds.includes(currentPlatform)) return false
  }
  const requiresWorkflowSteps = visibility.requiresWorkflowSteps || []
  if (requiresWorkflowSteps.length > 0) {
    const workflowSet = new Set((state.workflowSteps || []).map((step) => String(step || '').trim().toLowerCase()).filter(Boolean))
    if (!requiresWorkflowSteps.every((step) => workflowSet.has(String(step || '').trim().toLowerCase()))) return false
  }
  return true
}

export function getVisibleCreateBotScreens(
  config: CreateBotFlowConfig,
  state: CreateBotFlowState
): CreateBotScreen[] {
  return config.screenOrder
    .map((screenId) => config.screenDefinitions[screenId])
    .filter((screen): screen is CreateBotScreen => Boolean(screen))
    .filter((screen) => isCreateBotScreenVisible(screen, state))
}

export function getVisibleCreateBotStepGroups(
  config: CreateBotFlowConfig,
  screens: ReadonlyArray<CreateBotScreen>
): CreateBotStepGroup[] {
  const visibleStepGroupIds = new Set(screens.map((screen) => screen.stepGroupId))
  return config.stepGroups.filter((stepGroup) => visibleStepGroupIds.has(stepGroup.id))
}

export function getCreateBotCurrentScreen(
  pathname: string,
  screens: ReadonlyArray<CreateBotScreen>
): CreateBotScreen | null {
  const relativePath = getCreateBotRelativePath(pathname)
  return screens.find((screen) => screen.path === relativePath) || null
}

export function getCreateBotStepIndex(
  pathname: string,
  screens: ReadonlyArray<CreateBotScreen>,
  stepGroups: ReadonlyArray<CreateBotStepGroup>
): number {
  const currentScreen = getCreateBotCurrentScreen(pathname, screens)
  const activeStepGroupId = currentScreen?.stepGroupId || stepGroups[0]?.id
  const index = stepGroups.findIndex((stepGroup) => stepGroup.id === activeStepGroupId)
  return index >= 0 ? index : 0
}

export function getCreateBotNextPath(
  pathname: string,
  screens: ReadonlyArray<CreateBotScreen>
): string | null {
  const currentScreen = getCreateBotCurrentScreen(pathname, screens)
  if (!currentScreen) return screens[0] ? buildCreateBotPath(screens[0].path) : null
  const index = screens.findIndex((screen) => screen.id === currentScreen.id)
  const next = index >= 0 ? screens[index + 1] : null
  return next ? buildCreateBotPath(next.path) : null
}

export function getCreateBotPrevPath(
  pathname: string,
  screens: ReadonlyArray<CreateBotScreen>
): string | null {
  const currentScreen = getCreateBotCurrentScreen(pathname, screens)
  if (!currentScreen) return null
  const index = screens.findIndex((screen) => screen.id === currentScreen.id)
  const prev = index > 0 ? screens[index - 1] : null
  return prev ? buildCreateBotPath(prev.path) : null
}

export function getCreateBotFirstPath(screens: ReadonlyArray<CreateBotScreen>): string {
  return screens[0] ? buildCreateBotPath(screens[0].path) : '/create-bot'
}
