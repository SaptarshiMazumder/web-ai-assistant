import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { FlowSelect } from './FlowSelect'
import { LinePhonePreview } from './LinePhonePreview'

type SuggestedActionsState = {
  theme_mode: string
  layout: string
  card_background_color: string
  card_text_color: string
  button_background_color: string
  button_text_color: string
}

type AssetCarouselState = {
  bubble_size: string
  image_aspect_ratio: string
  body_background_color: string
  body_text_color: string
  overlay_background_color: string
}

type RichMenuStyleState = {
  background: string
  text: string
  muted_text: string
  button_background: string
  button_text: string
  border: string
  accent: string
}

type RichMenuActionState = {
  id: string
  enabled: boolean
  icon: string
  labels: { en: string; ja: string }
}

type RichMenuState = {
  styles: {
    normal: RichMenuStyleState
    support: RichMenuStyleState
  }
  actions: RichMenuActionState[]
  layouts: {
    normal: string[]
    support: string[]
  }
}

type LineDesignState = {
  suggested_actions: SuggestedActionsState
  asset_carousel: AssetCarouselState
  rich_menu: RichMenuState
}
type SuggestedPreviewMessage = {
  label?: string
  prompt?: string
  message?: string
  type?: string
  fastPathBinding?: string
  binding?: string
}

const HEX_RE = /^#(?:[0-9a-fA-F]{6})$/
const MAX_SUGGESTED_PREVIEW_ITEMS = 10
const SUGGESTED_BINDING_TO_RICH_ACTION_ID: Record<string, string> = {
  reservation: 'reserve',
  reserve: 'reserve',
  menu: 'menu',
  show_menu: 'menu',
  support: 'support',
  escalate: 'support',
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function asList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const item of value) {
    const str = String(item || '').trim()
    if (!str || seen.has(str)) continue
    seen.add(str)
    out.push(str)
  }
  return out
}

function asHex(value: unknown, fallback: string): string {
  const raw = String(value || '').trim()
  if (!raw) return fallback
  const normalized = raw.startsWith('#') ? raw : `#${raw}`
  return HEX_RE.test(normalized) ? normalized : fallback
}

function getRichMenuLabelsFromSuggestedPreviewMessages(messages: SuggestedPreviewMessage[] | undefined): Record<string, string> {
  const out: Record<string, string> = {}
  if (!Array.isArray(messages)) return out
  for (const item of messages) {
    const row = item && typeof item === 'object' ? item : {}
    const label = String(row.label || '').trim()
    if (!label) continue
    let binding = String(row.fastPathBinding || row.binding || '').trim().toLowerCase()
    if (!binding) {
      const suggestedType = String(row.type || '').trim().toLowerCase()
      if (suggestedType === 'show_menu') binding = 'show_menu'
      else if (suggestedType === 'escalate') binding = 'escalate'
    }
    const actionId = SUGGESTED_BINDING_TO_RICH_ACTION_ID[binding]
    if (!actionId || out[actionId]) continue
    out[actionId] = label
  }
  return out
}

function mergeState(
  profile: Record<string, unknown> | null,
  overrides: Record<string, unknown> | null,
  suggestedPreviewMessages?: SuggestedPreviewMessage[]
): LineDesignState {
  const profileRoot = asRecord(profile)
  const suggestedProfile = asRecord(profileRoot.suggested_actions)
  const suggestedDefaults = asRecord(suggestedProfile.defaults)
  const suggestedOptions = asRecord(suggestedProfile.options)
  const suggestedOverride = asRecord(asRecord(overrides).suggested_actions)
  const themeModes = asList(suggestedOptions.theme_modes)
  const layouts = asList(suggestedOptions.layouts)
  const themeModeDefault = String(suggestedDefaults.theme_mode || 'light').trim() || 'light'
  const layoutDefault = String(suggestedDefaults.layout || 'column').trim() || 'column'
  const themeMode = String(suggestedOverride.theme_mode || themeModeDefault).trim() || themeModeDefault
  const layout = String(suggestedOverride.layout || layoutDefault).trim() || layoutDefault

  const assetProfile = asRecord(profileRoot.asset_carousel)
  const assetDefaults = asRecord(assetProfile.defaults)
  const assetOptions = asRecord(assetProfile.options)
  const assetOverride = asRecord(asRecord(overrides).asset_carousel)
  const bubbleSizes = asList(assetOptions.bubble_sizes)
  const imageRatios = asList(assetOptions.image_aspect_ratios)
  const bubbleSizeDefault = String(assetDefaults.bubble_size || 'micro').trim() || 'micro'
  const ratioDefault = String(assetDefaults.image_aspect_ratio || '4:3').trim() || '4:3'
  const bubbleSize = String(assetOverride.bubble_size || bubbleSizeDefault).trim() || bubbleSizeDefault
  const imageAspectRatio = String(assetOverride.image_aspect_ratio || ratioDefault).trim() || ratioDefault

  const richProfile = asRecord(profileRoot.rich_menu)
  const richDefaults = asRecord(richProfile.defaults)
  const richOverride = asRecord(asRecord(overrides).rich_menu)
  const defaultStyles = asRecord(richDefaults.styles)
  const overrideStyles = asRecord(richOverride.styles)
  const normalStyleDefaults = asRecord(defaultStyles.normal)
  const supportStyleDefaults = asRecord(defaultStyles.support)
  const normalOverride = asRecord(overrideStyles.normal)
  const supportOverride = asRecord(overrideStyles.support)
  const defaultActions = Array.isArray(richDefaults.actions) ? (richDefaults.actions as unknown[]) : []
  const suggestedLabelByActionId = getRichMenuLabelsFromSuggestedPreviewMessages(suggestedPreviewMessages)
  const actionMap = new Map<string, RichMenuActionState>()
  for (const item of defaultActions) {
    const row = asRecord(item)
    const id = String(row.id || '').trim()
    if (!id) continue
    const labels = asRecord(row.labels)
    const suggestedLabel = suggestedLabelByActionId[id]
    actionMap.set(id, {
      id,
      enabled: row.enabled !== false,
      icon: String(row.icon || id).trim() || id,
      labels: {
        en: suggestedLabel || String(labels.en || '').trim() || id,
        ja: suggestedLabel || String(labels.ja || '').trim() || String(labels.en || '').trim() || id,
      },
    })
  }
  const actions = Array.from(actionMap.values())
  const actionIds = actions.map((item) => item.id)

  const defaultLayouts = asRecord(richDefaults.layouts)
  const normalLayoutRaw = asList(defaultLayouts.normal)
  const supportLayoutRaw = asList(defaultLayouts.support)
  const normalizeLayout = (raw: string[]) => {
    const ordered = raw.filter((id) => actionIds.includes(id))
    for (const id of actionIds) {
      if (!ordered.includes(id)) ordered.push(id)
    }
    return ordered
  }

  return {
    suggested_actions: {
      theme_mode: themeModes.includes(themeMode) ? themeMode : (themeModes[0] || 'light'),
      layout: layouts.includes(layout) ? layout : (layouts[0] || 'column'),
      card_background_color: asHex(
        suggestedOverride.card_background_color,
        asHex(suggestedDefaults.card_background_color, '#ffffff')
      ),
      card_text_color: asHex(suggestedOverride.card_text_color, asHex(suggestedDefaults.card_text_color, '#1f2937')),
      button_background_color: asHex(
        suggestedOverride.button_background_color,
        asHex(suggestedDefaults.button_background_color, '#f3f4f6')
      ),
      button_text_color: asHex(suggestedOverride.button_text_color, asHex(suggestedDefaults.button_text_color, '#374151')),
    },
    asset_carousel: {
      bubble_size: bubbleSizes.includes(bubbleSize) ? bubbleSize : (bubbleSizes[0] || 'micro'),
      image_aspect_ratio: imageRatios.includes(imageAspectRatio) ? imageAspectRatio : (imageRatios[0] || '4:3'),
      body_background_color: asHex(
        assetOverride.body_background_color,
        asHex(assetDefaults.body_background_color, '#111827')
      ),
      body_text_color: asHex(assetOverride.body_text_color, asHex(assetDefaults.body_text_color, '#ffffff')),
      overlay_background_color: asHex(
        assetOverride.overlay_background_color,
        asHex(assetDefaults.overlay_background_color, '#111827')
      ),
    },
    rich_menu: {
      styles: {
        normal: {
          background: asHex(normalOverride.background, asHex(normalStyleDefaults.background, '#f5f7fb')),
          text: asHex(normalOverride.text, asHex(normalStyleDefaults.text, '#0f172a')),
          muted_text: asHex(normalOverride.muted_text, asHex(normalStyleDefaults.muted_text, '#475569')),
          button_background: asHex(normalOverride.button_background, asHex(normalStyleDefaults.button_background, '#ffffff')),
          button_text: asHex(normalOverride.button_text, asHex(normalStyleDefaults.button_text, '#0f172a')),
          border: asHex(normalOverride.border, asHex(normalStyleDefaults.border, '#d7dde7')),
          accent: asHex(normalOverride.accent, asHex(normalStyleDefaults.accent, '#06c755')),
        },
        support: {
          background: asHex(supportOverride.background, asHex(supportStyleDefaults.background, '#0f172a')),
          text: asHex(supportOverride.text, asHex(supportStyleDefaults.text, '#f8fafc')),
          muted_text: asHex(supportOverride.muted_text, asHex(supportStyleDefaults.muted_text, '#cbd5e1')),
          button_background: asHex(
            supportOverride.button_background,
            asHex(supportStyleDefaults.button_background, '#fef3c7')
          ),
          button_text: asHex(supportOverride.button_text, asHex(supportStyleDefaults.button_text, '#92400e')),
          border: asHex(supportOverride.border, asHex(supportStyleDefaults.border, '#d7dde7')),
          accent: asHex(supportOverride.accent, asHex(supportStyleDefaults.accent, '#f59e0b')),
        },
      },
      actions,
      layouts: {
        normal: normalizeLayout(normalLayoutRaw),
        support: normalizeLayout(supportLayoutRaw),
      },
    },
  }
}

function toOverrides(state: LineDesignState): Record<string, unknown> {
  return {
    suggested_actions: state.suggested_actions,
    asset_carousel: state.asset_carousel,
    rich_menu: {
      styles: state.rich_menu.styles,
    },
  }
}

type LineDesignFormProps = {
  profile: Record<string, unknown> | null
  value: Record<string, unknown> | null
  onChange: (next: Record<string, unknown>) => void
  suggestedPreviewMessages?: SuggestedPreviewMessage[]
  actions?: ReactNode
  botName?: string
}

function ColorField({
  label,
  value,
  onChange,
}: {
  label: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <div className="design-form-field">
      <label className="design-form-label">{label}</label>
      <div className="design-form-color-row">
        <input
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="design-form-color-swatch"
        />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="design-form-input design-color-hex-input"
        />
      </div>
    </div>
  )
}

export function LineDesignForm({ profile, value, onChange, suggestedPreviewMessages, actions, botName }: LineDesignFormProps) {
  const { t } = useTranslation()
  const [state, setState] = useState<LineDesignState>(() => mergeState(profile, value, suggestedPreviewMessages))

  useEffect(() => {
    setState(mergeState(profile, value, suggestedPreviewMessages))
  }, [profile, value, suggestedPreviewMessages])

  const updateState = useCallback(
    (next: LineDesignState) => {
      setState(next)
      onChange(toOverrides(next))
    },
    [onChange]
  )

  const updateSuggested = useCallback(
    (key: keyof SuggestedActionsState, val: string | boolean) => {
      updateState({
        ...state,
        suggested_actions: { ...state.suggested_actions, [key]: val },
      })
    },
    [state, updateState]
  )

  /** Atomic update for linked fields (card+carousel bg/text/border) */
  const updateBoth = useCallback(
    (sugKey: keyof SuggestedActionsState, assetKey: keyof AssetCarouselState, val: string) => {
      updateState({
        ...state,
        suggested_actions: { ...state.suggested_actions, [sugKey]: val },
        asset_carousel: { ...state.asset_carousel, [assetKey]: val },
      })
    },
    [state, updateState]
  )

  const suggestedButtons = useMemo(() => {
    const previewLabels = Array.isArray(suggestedPreviewMessages)
      ? suggestedPreviewMessages
          .map((item) => String(item?.label || item?.prompt || item?.type || '').trim())
          .filter((item) => item.length > 0)
          .slice(0, MAX_SUGGESTED_PREVIEW_ITEMS)
      : []
    if (previewLabels.length) return previewLabels
    const ordered = state.rich_menu.layouts.normal
      .map((id) => state.rich_menu.actions.find((action) => action.id === id))
      .filter((item): item is RichMenuActionState => Boolean(item))
      .filter((item) => item.enabled)
      .map((item) => item.labels.en || item.id)
      .slice(0, MAX_SUGGESTED_PREVIEW_ITEMS)
    return ordered.length ? ordered : ['Action 1', 'Action 2']
  }, [state.rich_menu.actions, state.rich_menu.layouts.normal, suggestedPreviewMessages])
  const suggestedRows = useMemo(() => {
    if (state.suggested_actions.layout === 'row') {
      const rows: string[][] = []
      for (let idx = 0; idx < suggestedButtons.length; idx += 2) {
        const row = suggestedButtons.slice(idx, idx + 2)
        if (row.length) rows.push(row)
      }
      return rows
    }
    return suggestedButtons.map((item) => [item])
  }, [state.suggested_actions.layout, suggestedButtons])

  return (
    <div className="flow-panel-body flow-panel-body--wide">
      <div className="widget-design-grid">
        <div className="design-form">
          {/* ── Single Color Palette Section ───────────────────────────── */}
          <section className="ui-glass-card">
            <div className="card-title">{t('lineDesign.colorPalette', 'Colors')}</div>
            <div className="design-form-section">
              <div className="design-form-row">
                <div className="design-form-field">
                  <label className="design-form-label">{t('lineDesign.darkMode', 'Dark mode')}</label>
                  <FlowSelect
                    value={state.suggested_actions.theme_mode}
                    onChange={(next) => updateSuggested('theme_mode', String(next))}
                    options={[
                      { value: 'light', label: t('widgetDesign.light', 'Light') },
                      { value: 'dark', label: t('widgetDesign.dark', 'Dark') },
                    ]}
                  />
                </div>
              </div>

              <div className="design-form-row">
                <ColorField
                  label={t('lineDesign.cardBg', 'Card & Carousel bg')}
                  value={state.suggested_actions.card_background_color}
                  onChange={(val) => updateBoth('card_background_color', 'body_background_color', val)}
                />
                <ColorField
                  label={t('lineDesign.cardText', 'Card & Carousel text')}
                  value={state.suggested_actions.card_text_color}
                  onChange={(val) => updateBoth('card_text_color', 'body_text_color', val)}
                />
                <ColorField
                  label={t('lineDesign.buttonBg', 'Button bg')}
                  value={state.suggested_actions.button_background_color}
                  onChange={(val) => updateSuggested('button_background_color', val)}
                />
              </div>

              <div className="design-form-row">
                <ColorField
                  label={t('lineDesign.buttonText', 'Button text')}
                  value={state.suggested_actions.button_text_color}
                  onChange={(val) => updateSuggested('button_text_color', val)}
                />
              </div>
            </div>
          </section>

          {/* Reset button */}
          <div style={{ marginTop: '1rem' }}>
            <button
              type="button"
              onClick={() => {
                if (confirm(t('lineDesign.resetConfirm', 'Reset all colors to defaults?'))) {
                  setState(mergeState(profile, null, suggestedPreviewMessages))
                }
              }}
              style={{
                padding: '8px 16px',
                background: '#f5f5f5',
                border: '1px solid #ddd',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500',
                color: '#666',
              }}
            >
              {t('lineDesign.resetColors', 'Reset colors')}
            </button>
          </div>

          {actions ? <div className="design-form-actions">{actions}</div> : null}
        </div>

        {/* LINE phone preview – right column */}
        <LinePhonePreview
          suggestedActions={state.suggested_actions}
          assetCarousel={state.asset_carousel}
          richMenu={state.rich_menu}
          suggestedRows={suggestedRows}
          botName={botName}
        />
      </div>
    </div>
  )
}
