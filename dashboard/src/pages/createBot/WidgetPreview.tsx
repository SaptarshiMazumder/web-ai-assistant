/**
 * Live preview of the chat widget for the Design step.
 * Reflects all step4 settings: position, colors, title, size, placeholder, footer, theme,
 * header/launcher icons, launcher text, max height, font/header size.
 * Dimensions match backend api/widget/widget.js so Design, Testing, and website are identical.
 */

import { WIDGET_SIZE_DIMENSIONS } from '../../constants/widgetSizes'

type WidgetPreviewProps = {
  position: 'bottom-right' | 'bottom-left'
  primaryColor: string
  title: string
  size: 'small' | 'medium' | 'large'
  welcomeMessage?: string
  placeholder?: string
  footerMessage?: string
  theme?: 'light' | 'dark'
  textColor?: string
  headerIconUrl?: string
  launcherIconUrl?: string
  launcherText?: string
  maxHeight?: number
  fontSize?: 'small' | 'medium' | 'large'
  headerSize?: 'small' | 'medium' | 'large'
  suggestedMessages?: { id: string; label: string }[]
}

const FONT_SIZE_MAP = { small: 12, medium: 14, large: 16 } as const
const HEADER_PADDING_MAP = { small: 10, medium: 14, large: 18 } as const
const HEADER_FONT_MAP = { small: 13, medium: 15, large: 17 } as const

export function WidgetPreview({
  position: _position,
  primaryColor,
  title,
  size,
  welcomeMessage = 'Welcome! How can I help you today?',
  placeholder = 'Ask a question...',
  footerMessage = '',
  theme = 'light',
  textColor = '#ffffff',
  headerIconUrl = '',
  launcherIconUrl = '',
  launcherText = 'Help',
  maxHeight = 720,
  fontSize = 'medium',
  headerSize = 'small',
  suggestedMessages = [],
}: WidgetPreviewProps) {
  const { width, height: baseHeight } = WIDGET_SIZE_DIMENSIONS[size]
  const height = Math.min(baseHeight, maxHeight)
  const isDark = theme === 'dark'
  const contentBg = isDark ? '#1e293b' : '#ffffff'
  const contentColor = isDark ? '#e2e8f0' : '#334155'
  const bubbleBg = isDark ? '#334155' : '#f1f5f9'
  const inputBg = isDark ? '#0f172a' : '#f8fafc'
  const borderColor = isDark ? '#475569' : '#e2e8f0'
  const headerPadding = HEADER_PADDING_MAP[headerSize]
  const headerFontSize = HEADER_FONT_MAP[headerSize]
  const msgFontSize = FONT_SIZE_MAP[fontSize]

  return (
    <div className="widget-preview-wrap">
      <div className="widget-preview-stage" style={{ width: `${width}px` }}>
        <div
          className="widget-preview-window"
          style={{
            width: `${width}px`,
            height: `${height}px`,
            maxHeight: `${maxHeight}px`,
            borderRadius: '16px',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
            border: `1px solid ${borderColor}`,
          }}
        >
          {/* Header */}
          <div
            style={{
              background: primaryColor,
              color: textColor,
              padding: `${headerPadding}px 16px`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexShrink: 0,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              {headerIconUrl ? (
                <img
                  src={headerIconUrl}
                  alt=""
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: '50%',
                    objectFit: 'cover',
                  }}
                />
              ) : (
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: '50%',
                    background: 'rgba(255,255,255,0.25)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                  </svg>
                </div>
              )}
              <span style={{ fontWeight: 600, fontSize: `${headerFontSize}px` }}>{title || 'Chat'}</span>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: '#22c55e',
                  flexShrink: 0,
                }}
                title="Live"
              />
            </div>
          </div>

          {/* Chat content */}
          <div
            style={{
              flex: 1,
              background: contentBg,
              display: 'flex',
              flexDirection: 'column',
              padding: '16px',
              minHeight: 0,
            }}
          >
            <div style={{ flex: 1, overflow: 'auto' }}>
              <div
                style={{
                  display: 'flex',
                  gap: '10px',
                  alignItems: 'flex-start',
                  marginBottom: '12px',
                }}
              >
                {headerIconUrl ? (
                  <img
                    src={headerIconUrl}
                    alt=""
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      objectFit: 'cover',
                      flexShrink: 0,
                    }}
                  />
                ) : (
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      background: primaryColor,
                      flexShrink: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                    </svg>
                  </div>
                )}
                <div
                  style={{
                    background: bubbleBg,
                    color: contentColor,
                    padding: '10px 14px',
                    borderRadius: '12px 12px 12px 4px',
                    fontSize: `${msgFontSize}px`,
                    lineHeight: 1.5,
                    maxWidth: 'calc(100% - 38px)',
                    minWidth: 0,
                    overflowWrap: 'anywhere',
                    wordBreak: 'break-word',
                  }}
                >
                  {welcomeMessage}
                </div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '4px' }}>
                {(suggestedMessages.length ? suggestedMessages : [
                  { id: 'default_1', label: 'What can you do?' },
                  { id: 'default_2', label: 'Ask a question' },
                  { id: 'default_3', label: 'Get help' },
                ]).map((q) => (
                  <button
                    key={q.id}
                    type="button"
                    style={{
                      padding: '8px 12px',
                      borderRadius: '20px',
                      border: `1px solid ${borderColor}`,
                      background: contentBg,
                      fontSize: `${Math.max(11, msgFontSize - 2)}px`,
                      color: contentColor,
                      cursor: 'pointer',
                    }}
                  >
                    {q.label}
                  </button>
                ))}
              </div>
              {/* Typing bubble – matches live widget (avatar + grey bubble with three dots) */}
              <div
                style={{
                  display: 'flex',
                  gap: '10px',
                  alignItems: 'flex-start',
                  marginTop: '12px',
                }}
              >
                {headerIconUrl ? (
                  <img
                    src={headerIconUrl}
                    alt=""
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      objectFit: 'cover',
                      flexShrink: 0,
                    }}
                  />
                ) : (
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      background: primaryColor,
                      flexShrink: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                    </svg>
                  </div>
                )}
                <div
                  className="widget-preview-typing-bubble"
                  style={{
                    background: bubbleBg,
                    borderRadius: '12px 12px 12px 4px',
                    padding: '12px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 4,
                  }}
                >
                  <span className="widget-preview-typing-dot" />
                  <span className="widget-preview-typing-dot" />
                  <span className="widget-preview-typing-dot" />
                </div>
              </div>
            </div>

            {/* Input area – matches live widget: no lines above/below, just rounded input box on white */}
            <div style={{ flexShrink: 0, paddingTop: '12px', marginTop: '8px' }}>
              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                  alignItems: 'center',
                  background: inputBg,
                  border: `1px solid ${borderColor}`,
                  borderRadius: '12px',
                  padding: '8px 12px',
                }}
              >
                <input
                  type="text"
                  readOnly
                  placeholder={placeholder}
                  style={{
                    flex: 1,
                    border: 'none',
                    background: 'transparent',
                    fontSize: `${msgFontSize}px`,
                    outline: 'none',
                    color: contentColor,
                  }}
                />
                <button
                  type="button"
                  title="Send"
                  style={{
                    width: 40,
                    height: 40,
                    padding: 0,
                    border: 'none',
                    borderRadius: '50%',
                    background: primaryColor,
                    color: textColor,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 6l6 6-6 6" />
                  </svg>
                </button>
              </div>
              <div style={{ fontSize: '11px', color: isDark ? '#94a3b8' : '#64748b', marginTop: '6px', textAlign: 'center' }}>
                {footerMessage || 'Powered by WebAI'}
              </div>
            </div>
          </div>
        </div>

        {/* Launcher button below the widget */}
        <div
          title={launcherText}
          style={{
            width: 56,
            height: 56,
            borderRadius: '50%',
            background: primaryColor,
            color: textColor,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
            cursor: 'pointer',
            overflow: 'hidden',
          }}
        >
          {launcherIconUrl ? (
            <img src={launcherIconUrl} alt={launcherText} style={{ width: 28, height: 28, objectFit: 'contain' }} />
          ) : (
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
          )}
        </div>
      </div>
    </div>
  )
}
