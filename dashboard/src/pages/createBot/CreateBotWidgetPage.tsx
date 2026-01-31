import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Check } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'
import { WidgetPreview } from './WidgetPreview'
import type { CreateBotStep4Slice } from './CreateBotContext'

const FOOTER_MAX_LENGTH = 200

function step4ToWidgetConfig(step4: CreateBotStep4Slice): Record<string, unknown> {
  return {
    position: step4.widgetPosition,
    color: step4.widgetPrimaryColor,
    title: step4.widgetTitle || 'Chat',
    size: step4.widgetSize,
    welcomeMessage: step4.welcomeMessage || undefined,
    placeholder: step4.placeholder,
    footer: step4.footerMessage || undefined,
    theme: step4.theme,
    textColor: step4.textColor,
    launcherIcon: step4.launcherIconUrl || undefined,
    launcherText: step4.launcherText,
    headerIcon: step4.headerIconUrl || undefined,
    shareIcon: step4.shareIconUrl || undefined,
    maxHeight: step4.maxHeight,
    fontSize: step4.fontSize,
    headerSize: step4.headerSize,
    autoPopup: step4.autoPopupWelcome,
    autoScroll: step4.autoScrollNewMessages,
    displaySources: step4.displaySourcesInMessages,
    sourcesLabel: step4.sourcesLabel,
  }
}

export default function CreateBotWidgetPage() {
  const navigate = useNavigate()
  const { saveWidgetConfig } = useDashboardData()
  const { step1, step3, step4, flow } = useCreateBotFlow()
  const [saving, setSaving] = useState(false)
  const { botName } = step1
  const { botId, trainingStage, localError: trainingError } = step3
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const {
    widgetPosition,
    setWidgetPosition,
    widgetPrimaryColor,
    setWidgetPrimaryColor,
    widgetTitle,
    setWidgetTitle,
    widgetSize,
    setWidgetSize,
    welcomeMessage,
    setWelcomeMessage,
    placeholder,
    setPlaceholder,
    footerMessage,
    setFooterMessage,
    theme,
    setTheme,
    textColor,
    setTextColor,
    launcherIconUrl,
    setLauncherIconUrl,
    launcherText,
    setLauncherText,
    headerIconUrl,
    setHeaderIconUrl,
    shareIconUrl,
    setShareIconUrl,
    maxHeight,
    setMaxHeight,
    fontSize,
    setFontSize,
    headerSize,
    setHeaderSize,
    autoPopupWelcome,
    setAutoPopupWelcome,
    autoScrollNewMessages,
    setAutoScrollNewMessages,
    displaySourcesInMessages,
    setDisplaySourcesInMessages,
    sourcesLabel,
    setSourcesLabel,
  } = step4

  const headerInputRef = useRef<HTMLInputElement>(null)
  const launcherInputRef = useRef<HTMLInputElement>(null)
  const shareInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  useEffect(() => {
    if (botName && widgetTitle === 'Chat') {
      setWidgetTitle(botName.trim())
    }
  }, [botName, widgetTitle, setWidgetTitle])

  const handleContinue = async () => {
    if (!botId || !flow.nextPath) return
    setSaving(true)
    try {
      await saveWidgetConfig(botId, step4ToWidgetConfig(step4))
      navigate(flow.nextPath)
    } catch {
      setSaving(false)
    }
  }

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

  const resetTextColor = () => setTextColor('#ffffff')

  return (
    <div className="flow-panel-body">
      <div>
        <h2 className="card-title" style={{ marginBottom: '0.25rem' }}>Design the chat widget</h2>
        <p className="card-subtitle" style={{ margin: 0 }}>
          Customize how the widget appears. Changes update the preview on the right.
        </p>
      </div>

      {trainingStage === 'training' && (
        <div className="design-form-training-in-progress" style={{ marginBottom: 0, color: widgetPrimaryColor }}>
          <span className="discovery-loading-dots" aria-hidden>
            <span />
            <span />
            <span />
          </span>
          <span>Your bot is training. Design your widget in the meantime.</span>
        </div>
      )}
      {trainingStage === 'complete' && !trainingError && (
        <div
          className="design-form-training-done"
          style={{ color: widgetPrimaryColor, marginBottom: 0 }}
        >
          <Check size={20} strokeWidth={2.5} aria-hidden />
          <span>Your bot has finished learning from your website.</span>
        </div>
      )}
      {trainingError && (
        <div className="alert error" style={{ marginBottom: 0 }}>
          {trainingError}
        </div>
      )}

      <div className="widget-design-grid">
        <div className="design-form" style={{ accentColor: widgetPrimaryColor }}>
          <section className="card">
            <div className="card-title">Basics</div>
            <div className="design-form-section">
            <div className="design-form-row">
              <div className="design-form-field">
                <label className="design-form-label">Theme</label>
                <select
                  className="design-form-input"
                  value={theme}
                  onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
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
                    onChange={(e) => setWidgetPrimaryColor(e.target.value)}
                    className="design-form-color-swatch"
                  />
                  <input
                    type="text"
                    value={widgetPrimaryColor}
                    onChange={(e) => setWidgetPrimaryColor(e.target.value)}
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
                    onChange={(e) => setTextColor(e.target.value)}
                    className="design-form-color-swatch"
                  />
                  <input
                    type="text"
                    value={textColor}
                    onChange={(e) => setTextColor(e.target.value)}
                    placeholder="#ffffff"
                    className="design-form-input"
                    style={{ width: '140px' }}
                  />
                </div>
              </div>
            </div>
            <div className="design-form-row">
              <div className="design-form-field">
                <button type="button" className="design-form-text-btn" onClick={resetTextColor}>
                  Reset text color
                </button>
              </div>
            </div>
            <div className="design-form-field design-form-field-full">
              <label className="design-form-label">Initial welcome message</label>
              <span className="design-form-hint">First message shown by the bot when the chat opens.</span>
              <textarea
                className="design-form-input"
                value={welcomeMessage}
                onChange={(e) => setWelcomeMessage(e.target.value)}
                placeholder="Welcome! How can I help you today?"
                rows={2}
                style={{ resize: 'vertical', width: '100%' }}
              />
            </div>
            </div>
          </section>

          {/* Advanced: collapsible */}
          <section className="design-form-section">
            <button
              type="button"
              className="design-form-advanced-trigger"
              onClick={() => setAdvancedOpen((o) => !o)}
              aria-expanded={advancedOpen}
            >
              <span>Advanced</span>
              <span className="design-form-advanced-trigger-icon" style={{ background: widgetPrimaryColor }}>
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
              onChange={(e) => setPlaceholder(e.target.value)}
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
                onChange={(e) => setFooterMessage(e.target.value.slice(0, FOOTER_MAX_LENGTH))}
                placeholder=""
                rows={3}
                style={{ flex: 1, minWidth: 0, resize: 'vertical' }}
              />
              <span className="muted" style={{ flexShrink: 0, paddingTop: '0.625rem', fontSize: '0.9375rem' }}>
                {footerMessage.length}/{FOOTER_MAX_LENGTH}
              </span>
            </div>
          </div>
            </div>
          </section>

          <section className="card design-form-card-icons">
            <div className="card-title">Icons</div>
            <div className="design-form-section">
          <div className="design-form-row">
          <div className="design-form-field">
            <label className="design-form-label">Launcher icon</label>
            <span className="design-form-hint">Click to choose image. 100×100px recommended.</span>
            <input ref={launcherInputRef} type="file" accept="image/*" onChange={handleFile(setLauncherIconUrl, launcherInputRef)} style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }} aria-hidden />
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
                <button
                  type="button"
                  className="design-form-icon-clear"
                  onClick={(e) => { e.stopPropagation(); setLauncherIconUrl(''); }}
                  aria-label="Remove launcher icon"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
                </button>
              )}
            </div>
          </div>
          <div className="design-form-field">
            <label className="design-form-label">Launcher text</label>
            <input type="text" className="design-form-input" value={launcherText} onChange={(e) => setLauncherText(e.target.value)} placeholder="Help" style={{ width: '100%', minWidth: '220px' }} />
          </div>
          </div>
          <div className="design-form-row">
          <div className="design-form-field">
            <label className="design-form-label">Header icon (bot avatar)</label>
            <span className="design-form-hint">Click to choose image. 100×100px recommended.</span>
            <input ref={headerInputRef} type="file" accept="image/*" onChange={handleFile(setHeaderIconUrl, headerInputRef)} style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }} aria-hidden />
            <div className="design-form-icon-wrap">
              <button
                type="button"
                onClick={() => headerInputRef.current?.click()}
                className="design-form-input"
                style={{ width: 100, height: 100, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', cursor: 'pointer' }}
              >
                {headerIconUrl ? <img src={headerIconUrl} alt="Header" style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>}
              </button>
              {headerIconUrl && (
                <button type="button" className="design-form-icon-clear" onClick={(e) => { e.stopPropagation(); setHeaderIconUrl(''); }} aria-label="Remove header icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
                </button>
              )}
            </div>
          </div>
          <div className="design-form-field">
            <label className="design-form-label">Share icon</label>
            <span className="design-form-hint">Click to choose image.</span>
            <input ref={shareInputRef} type="file" accept="image/*" onChange={handleFile(setShareIconUrl, shareInputRef)} style={{ position: 'absolute', width: 0, height: 0, opacity: 0, pointerEvents: 'none' }} aria-hidden />
            <div className="design-form-icon-wrap">
              <button
                type="button"
                onClick={() => shareInputRef.current?.click()}
                className="design-form-input"
                style={{ width: 100, height: 100, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', cursor: 'pointer' }}
              >
                {shareIconUrl ? <img src={shareIconUrl} alt="Share" style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <span className="muted" style={{ fontSize: '0.9375rem' }}>—</span>}
              </button>
              {shareIconUrl && (
                <button type="button" className="design-form-icon-clear" onClick={(e) => { e.stopPropagation(); setShareIconUrl(''); }} aria-label="Remove share icon">
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
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', accentColor: widgetPrimaryColor } as React.CSSProperties}>
              <input type="range" min={400} max={800} step={20} value={maxHeight} onChange={(e) => setMaxHeight(Number(e.target.value))} className="design-form-input design-form-range-full" style={{ flex: 1, minWidth: 0 }} />
              <span style={{ fontSize: '1rem', minWidth: '3rem' }}>{maxHeight}px</span>
            </div>
          </div>
          <div className="design-form-field">
            <label className="design-form-label">Position</label>
            <div className="design-form-radio-group">
              <label className="design-form-radio-card">
                <input type="radio" name="widgetPosition" value="bottom-right" checked={widgetPosition === 'bottom-right'} onChange={() => setWidgetPosition('bottom-right')} />
                <span>Bottom right</span>
              </label>
              <label className="design-form-radio-card">
                <input type="radio" name="widgetPosition" value="bottom-left" checked={widgetPosition === 'bottom-left'} onChange={() => setWidgetPosition('bottom-left')} />
                <span>Bottom left</span>
              </label>
            </div>
          </div>
          <div className="design-form-field design-form-field-full">
            <label className="design-form-label">Widget title</label>
            <input type="text" className="design-form-input" value={widgetTitle} onChange={(e) => setWidgetTitle(e.target.value)} placeholder="Chat" style={{ width: '100%' }} />
          </div>
          <div className="design-form-field">
            <label className="design-form-label">Size</label>
            <div className="design-form-radio-group">
              {(['small', 'medium', 'large'] as const).map((size) => (
                <label key={size} className="design-form-radio-card">
                  <input type="radio" name="widgetSize" value={size} checked={widgetSize === size} onChange={() => setWidgetSize(size)} />
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
              <select className="design-form-input" value={fontSize} onChange={(e) => setFontSize(e.target.value as 'small' | 'medium' | 'large')} style={{ minWidth: '120px' }}>
                <option value="small">Small</option><option value="medium">Medium</option><option value="large">Large</option>
              </select>
            </div>
            <div className="design-form-field">
              <label className="design-form-label">Header size</label>
              <select className="design-form-input" value={headerSize} onChange={(e) => setHeaderSize(e.target.value as 'small' | 'medium' | 'large')} style={{ minWidth: '200px' }}>
                <option value="small">Small</option><option value="medium">Medium</option><option value="large">Large</option>
              </select>
            </div>
          </div>
          <div className="design-form-row">
            <div className="design-form-field">
              <label className="design-form-label">Auto popup welcome</label>
              <select className="design-form-input" value={autoPopupWelcome} onChange={(e) => setAutoPopupWelcome(e.target.value as 'off' | '1s' | '2s' | '3s')} style={{ minWidth: '160px' }}>
                <option value="off">Off</option><option value="1s">1s</option><option value="2s">2s</option><option value="3s">3s</option>
              </select>
            </div>
            <div className="design-form-field">
              <label className="design-form-label">Auto scroll</label>
              <select className="design-form-input" value={autoScrollNewMessages ? 'yes' : 'no'} onChange={(e) => setAutoScrollNewMessages(e.target.value === 'yes')} style={{ minWidth: '160px' }}>
                <option value="yes">Yes</option><option value="no">No</option>
              </select>
            </div>
            <div className="design-form-field">
              <label className="design-form-label">Display sources</label>
              <select className="design-form-input" value={displaySourcesInMessages ? 'yes' : 'no'} onChange={(e) => setDisplaySourcesInMessages(e.target.value === 'yes')} style={{ minWidth: '160px' }}>
                <option value="no">No</option><option value="yes">Yes</option>
              </select>
            </div>
          </div>
          {displaySourcesInMessages && (
            <div className="design-form-field design-form-field-full">
              <label className="design-form-label">Sources label</label>
              <input type="text" className="design-form-input" value={sourcesLabel} onChange={(e) => setSourcesLabel(e.target.value)} placeholder="Sources" style={{ width: '100%' }} />
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
        />
      </div>

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        <button type="button" className="primary" onClick={() => void handleContinue()} disabled={saving}>
          {saving ? 'Saving…' : 'Continue'}
        </button>
      </div>
    </div>
  )
}
