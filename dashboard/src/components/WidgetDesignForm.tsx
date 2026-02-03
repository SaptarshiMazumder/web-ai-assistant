import { useCallback, useRef, useState } from 'react'
import { WidgetPreview } from '../pages/createBot/WidgetPreview'
import { SuggestedMessagesEditor } from './SuggestedMessagesEditor'

const FOOTER_MAX_LENGTH = 200

export type WidgetDesignState = {
  widgetPosition: 'bottom-right' | 'bottom-left'
  widgetPrimaryColor: string
  widgetTitle: string
  widgetSize: 'small' | 'medium' | 'large'
  welcomeMessage: string
  placeholder: string
  footerMessage: string
  theme: 'light' | 'dark'
  textColor: string
  launcherIconUrl: string
  launcherText: string
  headerIconUrl: string
  shareIconUrl: string
  maxHeight: number
  fontSize: 'small' | 'medium' | 'large'
  headerSize: 'small' | 'medium' | 'large'
  autoPopupWelcome: 'off' | '1s' | '2s' | '3s'
  autoScrollNewMessages: boolean
  displaySourcesInMessages: boolean
  sourcesLabel: string
  suggestedMessages: SuggestedMessageConfig[]
}

export type SuggestedMessageConfig = {
  id: string
  label: string
  type: 'user_message' | 'ai_response' | 'escalate'
  message?: string
  prompt?: string
}

export const DEFAULT_WIDGET_DESIGN_STATE: WidgetDesignState = {
  widgetPosition: 'bottom-right',
  widgetPrimaryColor: '#6366f1',
  widgetTitle: 'Chat',
  widgetSize: 'medium',
  welcomeMessage: 'Welcome! How can I help you today?',
  placeholder: 'Ask a question...',
  footerMessage: 'Powered by WebAI',
  theme: 'light',
  textColor: '#ffffff',
  launcherIconUrl: '',
  launcherText: 'Help',
  headerIconUrl: '',
  shareIconUrl: '',
  maxHeight: 720,
  fontSize: 'medium',
  headerSize: 'small',
  autoPopupWelcome: 'off',
  autoScrollNewMessages: true,
  displaySourcesInMessages: false,
  sourcesLabel: 'Sources',
  suggestedMessages: [
    { id: 'suggest_1', label: 'What can you do?', type: 'user_message', message: 'What can you do?' },
    { id: 'suggest_2', label: 'Ask a question', type: 'user_message', message: 'Ask a question' },
    { id: 'suggest_3', label: 'Get help', type: 'user_message', message: 'Get help' },
    { id: 'suggest_4', label: 'Escalate to support', type: 'escalate' },
  ],
}

export function widgetConfigToState(config: Record<string, unknown> | null): WidgetDesignState {
  const d = { ...DEFAULT_WIDGET_DESIGN_STATE }
  if (!config || typeof config !== 'object') return d
  const pos = config.position
  if (pos === 'bottom-right' || pos === 'bottom-left') d.widgetPosition = pos
  if (typeof config.color === 'string') d.widgetPrimaryColor = config.color
  if (typeof config.title === 'string') d.widgetTitle = config.title
  const size = config.size
  if (size === 'small' || size === 'medium' || size === 'large') d.widgetSize = size
  if (typeof config.welcomeMessage === 'string') d.welcomeMessage = config.welcomeMessage
  if (typeof config.placeholder === 'string') d.placeholder = config.placeholder
  if (typeof config.footer === 'string') d.footerMessage = config.footer
  const theme = config.theme
  if (theme === 'light' || theme === 'dark') d.theme = theme
  if (typeof config.textColor === 'string') d.textColor = config.textColor
  if (typeof config.launcherIcon === 'string') d.launcherIconUrl = config.launcherIcon
  if (typeof config.launcherText === 'string') d.launcherText = config.launcherText
  if (typeof config.headerIcon === 'string') d.headerIconUrl = config.headerIcon
  if (typeof config.shareIcon === 'string') d.shareIconUrl = config.shareIcon
  if (typeof config.maxHeight === 'number' && config.maxHeight >= 400 && config.maxHeight <= 800) d.maxHeight = config.maxHeight
  const fs = config.fontSize
  if (fs === 'small' || fs === 'medium' || fs === 'large') d.fontSize = fs
  const hs = config.headerSize
  if (hs === 'small' || hs === 'medium' || hs === 'large') d.headerSize = hs
  const ap = config.autoPopup
  if (ap === 'off' || ap === '1s' || ap === '2s' || ap === '3s') d.autoPopupWelcome = ap
  if (typeof config.autoScroll === 'boolean') d.autoScrollNewMessages = config.autoScroll
  if (typeof config.displaySources === 'boolean') d.displaySourcesInMessages = config.displaySources
  if (typeof config.sourcesLabel === 'string') d.sourcesLabel = config.sourcesLabel
  if (Array.isArray((config as Record<string, unknown>).suggestedMessages)) {
    const raw = (config as Record<string, unknown>).suggestedMessages as SuggestedMessageConfig[]
    d.suggestedMessages = raw
      .map((item, idx): SuggestedMessageConfig | null => {
        const label = typeof item?.label === 'string' ? item.label : ''
        if (!label) return null
        const type =
          item?.type === 'ai_response' || item?.type === 'user_message' || item?.type === 'escalate'
            ? item.type
            : 'user_message'
        const message = typeof item?.message === 'string' ? item.message : undefined
        const prompt = typeof item?.prompt === 'string' ? item.prompt : undefined
        return { id: String(item?.id || `suggest_${idx}`), label, type, message, prompt }
      })
      .filter((item): item is SuggestedMessageConfig => Boolean(item))
  }
  return d
}

export function stateToWidgetConfig(s: WidgetDesignState): Record<string, unknown> {
  return {
    position: s.widgetPosition,
    color: s.widgetPrimaryColor,
    title: s.widgetTitle || 'Chat',
    size: s.widgetSize,
    welcomeMessage: s.welcomeMessage || undefined,
    placeholder: s.placeholder,
    footer: s.footerMessage || undefined,
    theme: s.theme,
    textColor: s.textColor,
    launcherIcon: s.launcherIconUrl || undefined,
    launcherText: s.launcherText,
    headerIcon: s.headerIconUrl || undefined,
    shareIcon: s.shareIconUrl || undefined,
    maxHeight: s.maxHeight,
    fontSize: s.fontSize,
    headerSize: s.headerSize,
    autoPopup: s.autoPopupWelcome,
    autoScroll: s.autoScrollNewMessages,
    displaySources: s.displaySourcesInMessages,
    sourcesLabel: s.sourcesLabel,
    suggestedMessages: s.suggestedMessages,
  }
}

export type WidgetDesignFormProps = {
  value: WidgetDesignState
  onChange: <K extends keyof WidgetDesignState>(key: K, value: WidgetDesignState[K]) => void
  banner?: React.ReactNode
  actions?: React.ReactNode
  showSuggestedMessages?: boolean
}

export function WidgetDesignForm({
  value,
  onChange,
  banner,
  actions,
  showSuggestedMessages = true,
}: WidgetDesignFormProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const headerInputRef = useRef<HTMLInputElement>(null)
  const launcherInputRef = useRef<HTMLInputElement>(null)
  const shareInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (setter: (url: string) => void, inputRef: React.RefObject<HTMLInputElement | null>) => {
      return (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return
        const url = URL.createObjectURL(file)
        setter(url)
        if (inputRef.current) inputRef.current.value = ''
      }
    },
    []
  )

  const update = useCallback(
    <K extends keyof WidgetDesignState>(key: K, val: WidgetDesignState[K]) => {
      onChange(key, val)
    },
    [onChange]
  )

  const {
    widgetPosition,
    widgetPrimaryColor,
    widgetTitle,
    widgetSize,
    welcomeMessage,
    placeholder,
    footerMessage,
    theme,
    textColor,
    launcherIconUrl,
    launcherText,
    headerIconUrl,
    shareIconUrl,
    maxHeight,
    fontSize,
    headerSize,
    autoPopupWelcome,
    autoScrollNewMessages,
    displaySourcesInMessages,
    sourcesLabel,
    suggestedMessages,
  } = value

  return (
    <div className="flow-panel-body">
      <div>
        <h2 className="card-title" style={{ marginBottom: '0.25rem' }}>Design the chat widget</h2>
        <p className="card-subtitle" style={{ margin: 0 }}>
          Customize how the widget appears. Changes update the preview on the right.
        </p>
      </div>

      {banner}

      <div className="widget-design-grid">
        <div className="design-form">
          <section className="card">
            <div className="card-title">Basics</div>
            <div className="design-form-section">
              <div className="design-form-row">
                <div className="design-form-field">
                  <label className="design-form-label">Theme</label>
                  <select
                    className="design-form-input"
                    value={theme}
                    onChange={(e) => update('theme', e.target.value as 'light' | 'dark')}
                    style={{ minWidth: '200px' }}
                  >
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                  </select>
                </div>
                <div className="design-form-field">
                  <label className="design-form-label">Accent color</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type="color"
                      value={widgetPrimaryColor}
                      onChange={(e) => update('widgetPrimaryColor', e.target.value)}
                      className="design-form-color-swatch"
                    />
                    <input
                      type="text"
                      value={widgetPrimaryColor}
                      onChange={(e) => update('widgetPrimaryColor', e.target.value)}
                      placeholder="#6366f1"
                      className="design-form-input"
                      style={{ width: '180px' }}
                    />
                  </div>
                </div>
                <div className="design-form-field">
                  <label className="design-form-label">Text color</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type="color"
                      value={textColor}
                      onChange={(e) => update('textColor', e.target.value)}
                      className="design-form-color-swatch"
                    />
                    <input
                      type="text"
                      value={textColor}
                      onChange={(e) => update('textColor', e.target.value)}
                      placeholder="#ffffff"
                      className="design-form-input"
                      style={{ width: '140px' }}
                    />
                  </div>
                </div>
              </div>
              <div className="design-form-field design-form-field-full">
                <label className="design-form-label">Initial welcome message</label>
                <span className="design-form-hint">First message shown by the bot when the chat opens.</span>
                <textarea
                  className="design-form-input"
                  value={welcomeMessage}
                  onChange={(e) => update('welcomeMessage', e.target.value)}
                  placeholder="Welcome! How can I help you today?"
                  rows={2}
                  style={{ resize: 'vertical', width: '100%' }}
                />
              </div>
            </div>
          </section>

          <section className="design-form-section">
            <button
              type="button"
              className="design-form-advanced-trigger"
              onClick={() => setAdvancedOpen((o) => !o)}
              aria-expanded={advancedOpen}
            >
              <span>Advanced</span>
              <span className="design-form-advanced-trigger-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </span>
            </button>
            {advancedOpen && (
              <div className="design-form-advanced-content">
                <section className="card">
                  <div className="card-title">Messages</div>
                  <div className="design-form-section">
                    <div className="design-form-field design-form-field-full">
                      <label className="design-form-label">Placeholder message</label>
                      <input
                        type="text"
                        className="design-form-input"
                        value={placeholder}
                        onChange={(e) => update('placeholder', e.target.value)}
                        placeholder="Ask a question..."
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div className="design-form-field design-form-field-full">
                      <label className="design-form-label">Footer message</label>
                      <span className="design-form-hint">Optional message under the chat input. Supports markdown.</span>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                        <textarea
                          className="design-form-input"
                          value={footerMessage}
                          onChange={(e) => update('footerMessage', e.target.value.slice(0, FOOTER_MAX_LENGTH))}
                          placeholder=""
                          rows={3}
                          style={{ flex: 1, minWidth: 0, resize: 'vertical' }}
                        />
                        <span className="muted" style={{ flexShrink: 0, paddingTop: '0.625rem', fontSize: '0.9375rem' }}>
                          {footerMessage.length}/{FOOTER_MAX_LENGTH}
                        </span>
                      </div>
                    </div>
                    {showSuggestedMessages && (
                      <SuggestedMessagesEditor
                        suggestedMessages={suggestedMessages}
                        onChange={(next) => update('suggestedMessages', next)}
                      />
                    )}
                  </div>
                </section>

                <section className="card design-form-card-icons">
                  <div className="card-title">Icons</div>
                  <div className="design-form-section">
                    <div className="design-form-row">
                      <div className="design-form-field">
                        <label className="design-form-label">Launcher icon</label>
                        <span className="design-form-hint">Click to choose image. 100×100px recommended.</span>
                        <input ref={launcherInputRef} type="file" accept="image/*" onChange={handleFile((url) => update('launcherIconUrl', url), launcherInputRef)} style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }} aria-hidden />
                        <div className="design-form-icon-wrap">
                          <button
                            type="button"
                            onClick={() => launcherInputRef.current?.click()}
                            className="design-form-input"
                            style={{ width: 100, height: 100, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', cursor: 'pointer' }}
                          >
                            {launcherIconUrl ? (
                              <img src={launcherIconUrl} alt="Launcher" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                            ) : (
                              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>
                            )}
                          </button>
                          {launcherIconUrl && (
                            <button type="button" className="design-form-icon-clear" onClick={(e) => { e.stopPropagation(); update('launcherIconUrl', ''); }} aria-label="Remove launcher icon">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="design-form-field">
                        <label className="design-form-label">Launcher text</label>
                        <input type="text" className="design-form-input" value={launcherText} onChange={(e) => update('launcherText', e.target.value)} placeholder="Help" style={{ width: '100%', minWidth: '220px' }} />
                      </div>
                    </div>
                    <div className="design-form-row">
                      <div className="design-form-field">
                        <label className="design-form-label">Header icon (bot avatar)</label>
                        <span className="design-form-hint">Click to choose image. 100×100px recommended.</span>
                        <input ref={headerInputRef} type="file" accept="image/*" onChange={handleFile((url) => update('headerIconUrl', url), headerInputRef)} style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }} aria-hidden />
                        <div className="design-form-icon-wrap">
                          <button type="button" onClick={() => headerInputRef.current?.click()} className="design-form-input" style={{ width: 100, height: 100, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', cursor: 'pointer' }}>
                            {headerIconUrl ? <img src={headerIconUrl} alt="Header" style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>}
                          </button>
                          {headerIconUrl && (
                            <button type="button" className="design-form-icon-clear" onClick={(e) => { e.stopPropagation(); update('headerIconUrl', ''); }} aria-label="Remove header icon">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="design-form-field">
                        <label className="design-form-label">Share icon</label>
                        <span className="design-form-hint">Click to choose image.</span>
                        <input ref={shareInputRef} type="file" accept="image/*" onChange={handleFile((url) => update('shareIconUrl', url), shareInputRef)} style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }} aria-hidden />
                        <div className="design-form-icon-wrap">
                          <button type="button" onClick={() => shareInputRef.current?.click()} className="design-form-input" style={{ width: 100, height: 100, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', cursor: 'pointer' }}>
                            {shareIconUrl ? <img src={shareIconUrl} alt="Share" style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <span className="muted" style={{ fontSize: '0.9375rem' }}>—</span>}
                          </button>
                          {shareIconUrl && (
                            <button type="button" className="design-form-icon-clear" onClick={(e) => { e.stopPropagation(); update('shareIconUrl', ''); }} aria-label="Remove share icon">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </section>

                <section className="card">
                  <div className="card-title">Layout & size</div>
                  <div className="design-form-section">
                    <div className="design-form-field">
                      <label className="design-form-label">Max height</label>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <input type="range" min={400} max={800} step={20} value={maxHeight} onChange={(e) => update('maxHeight', Number(e.target.value))} className="design-form-input design-form-range-full" style={{ flex: 1, minWidth: 0 }} />
                        <span style={{ fontSize: '1rem', minWidth: '3rem' }}>{maxHeight}px</span>
                      </div>
                    </div>
                    <div className="design-form-field">
                      <label className="design-form-label">Position</label>
                      <div className="design-form-radio-group">
                        <label className="design-form-radio-card">
                          <input type="radio" name="widgetPosition" value="bottom-right" checked={widgetPosition === 'bottom-right'} onChange={() => update('widgetPosition', 'bottom-right')} />
                          <span>Bottom right</span>
                        </label>
                        <label className="design-form-radio-card">
                          <input type="radio" name="widgetPosition" value="bottom-left" checked={widgetPosition === 'bottom-left'} onChange={() => update('widgetPosition', 'bottom-left')} />
                          <span>Bottom left</span>
                        </label>
                      </div>
                    </div>
                    <div className="design-form-field design-form-field-full">
                      <label className="design-form-label">Widget title</label>
                      <input type="text" className="design-form-input" value={widgetTitle} onChange={(e) => update('widgetTitle', e.target.value)} placeholder="Chat" style={{ width: '100%' }} />
                    </div>
                    <div className="design-form-field">
                      <label className="design-form-label">Size</label>
                      <div className="design-form-radio-group">
                        {(['small', 'medium', 'large'] as const).map((size) => (
                          <label key={size} className="design-form-radio-card">
                            <input type="radio" name="widgetSize" value={size} checked={widgetSize === size} onChange={() => update('widgetSize', size)} />
                            <span style={{ textTransform: 'capitalize' }}>{size}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                </section>

                <section className="card">
                  <div className="card-title">Behaviour</div>
                  <div className="design-form-section">
                    <div className="design-form-row">
                      <div className="design-form-field">
                        <label className="design-form-label">Font size</label>
                        <select className="design-form-input" value={fontSize} onChange={(e) => update('fontSize', e.target.value as 'small' | 'medium' | 'large')} style={{ minWidth: '120px' }}>
                          <option value="small">Small</option><option value="medium">Medium</option><option value="large">Large</option>
                        </select>
                      </div>
                      <div className="design-form-field">
                        <label className="design-form-label">Header size</label>
                        <select className="design-form-input" value={headerSize} onChange={(e) => update('headerSize', e.target.value as 'small' | 'medium' | 'large')} style={{ minWidth: '200px' }}>
                          <option value="small">Small</option><option value="medium">Medium</option><option value="large">Large</option>
                        </select>
                      </div>
                    </div>
                    <div className="design-form-row">
                      <div className="design-form-field">
                        <label className="design-form-label">Auto popup welcome</label>
                        <select className="design-form-input" value={autoPopupWelcome} onChange={(e) => update('autoPopupWelcome', e.target.value as 'off' | '1s' | '2s' | '3s')} style={{ minWidth: '160px' }}>
                          <option value="off">Off</option><option value="1s">1s</option><option value="2s">2s</option><option value="3s">3s</option>
                        </select>
                      </div>
                      <div className="design-form-field">
                        <label className="design-form-label">Auto scroll</label>
                        <select className="design-form-input" value={autoScrollNewMessages ? 'yes' : 'no'} onChange={(e) => update('autoScrollNewMessages', e.target.value === 'yes')} style={{ minWidth: '160px' }}>
                          <option value="yes">Yes</option><option value="no">No</option>
                        </select>
                      </div>
                      <div className="design-form-field">
                        <label className="design-form-label">Display sources</label>
                        <select className="design-form-input" value={displaySourcesInMessages ? 'yes' : 'no'} onChange={(e) => update('displaySourcesInMessages', e.target.value === 'yes')} style={{ minWidth: '160px' }}>
                          <option value="no">No</option><option value="yes">Yes</option>
                        </select>
                      </div>
                    </div>
                    {displaySourcesInMessages && (
                      <div className="design-form-field design-form-field-full">
                        <label className="design-form-label">Sources label</label>
                        <input type="text" className="design-form-input" value={sourcesLabel} onChange={(e) => update('sourcesLabel', e.target.value)} placeholder="Sources" style={{ width: '100%' }} />
                      </div>
                    )}
                  </div>
                </section>
              </div>
            )}
          </section>
        </div>

        <WidgetPreview
          position={widgetPosition}
          primaryColor={widgetPrimaryColor}
          title={widgetTitle || 'Chat'}
          size={widgetSize}
          welcomeMessage={welcomeMessage}
          placeholder={placeholder}
          footerMessage={footerMessage}
          theme={theme}
          textColor={textColor}
          headerIconUrl={headerIconUrl}
          launcherIconUrl={launcherIconUrl}
          launcherText={launcherText}
          maxHeight={maxHeight}
          fontSize={fontSize}
          headerSize={headerSize}
          suggestedMessages={suggestedMessages.map((msg) => ({ id: msg.id, label: msg.label }))}
        />
      </div>


      {actions != null && <div className="flow-actions">{actions}</div>}
    </div>
  )
}
