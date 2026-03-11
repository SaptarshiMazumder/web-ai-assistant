/**
 * Realistic LINE chat phone preview – iPhone 15 Pro style frame.
 * - chat_dark_mode controls the LINE chat UI (header, bg, bubbles, input)
 * - theme_mode controls the card/flex-message theme (light/dark)
 * - chat_background_color is a separate setting with LINE default grey
 * - single show_border toggle for both cards and carousel
 */
import { useState } from 'react'

type SuggestedActionsPreview = {
  theme_mode: string
  layout: string
  card_background_color: string
  card_text_color: string
  button_background_color: string
  button_text_color: string
}

type AssetCarouselPreview = {
  bubble_size: string
  image_aspect_ratio: string
  body_background_color: string
  body_text_color: string
  overlay_background_color: string
}

type RichMenuStylePreview = {
  background: string
  text: string
  muted_text: string
  button_background: string
  button_text: string
  border: string
  accent: string
}

type RichMenuActionPreview = {
  id: string
  enabled: boolean
  icon: string
  labels: { en: string; ja: string }
}

type RichMenuPreview = {
  styles: {
    normal: RichMenuStylePreview
    support: RichMenuStylePreview
  }
  actions: RichMenuActionPreview[]
  layouts: {
    normal: string[]
    support: string[]
  }
}

export type LinePhonePreviewProps = {
  suggestedActions: SuggestedActionsPreview
  assetCarousel: AssetCarouselPreview
  richMenu: RichMenuPreview
  suggestedRows: string[][]
  botName?: string
}

/* ── Rich-menu icon SVGs ──────────────────────────────────────────────── */
function RichMenuIcon({ actionId, color, size = 18 }: { actionId: string; color: string; size?: number }) {
  const s = { width: size, height: size, fill: 'none', stroke: color, strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  switch (actionId) {
    case 'reserve':
      return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>)
    case 'menu':
      return (<svg viewBox="0 0 24 24" {...s}><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/></svg>)
    case 'support':
      return (<svg viewBox="0 0 24 24" {...s}><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>)
    case 'back_to_ai':
      return (<svg viewBox="0 0 24 24" {...s}><polyline points="15 18 9 12 15 6"/></svg>)
    case 'chat':
      return (<svg viewBox="0 0 24 24" {...s}><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 014 11.5 8.5 8.5 0 018.7 3.9a8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>)
    case 'help':
      return (<svg viewBox="0 0 24 24" {...s}><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>)
    case 'link':
      return (<svg viewBox="0 0 24 24" {...s}><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>)
    default:
      return (<svg viewBox="0 0 24 24" {...s}><circle cx="12" cy="12" r="3"/></svg>)
  }
}

/* Placeholder food images as inline SVGs with warm/cool tones */
function FoodImagePlaceholder({ variant }: { variant: 0 | 1 }) {
  if (variant === 0) {
    return (
      <svg width="100%" height="100%" viewBox="0 0 200 150" preserveAspectRatio="xMidYMid slice">
        <defs>
          <radialGradient id="food0bg" cx="40%" cy="45%" r="70%">
            <stop offset="0%" stopColor="#f97316"/>
            <stop offset="50%" stopColor="#ea580c"/>
            <stop offset="100%" stopColor="#9a3412"/>
          </radialGradient>
        </defs>
        <rect width="200" height="150" fill="url(#food0bg)"/>
        {/* plate */}
        <ellipse cx="100" cy="82" rx="52" ry="38" fill="rgba(255,255,255,0.15)"/>
        <ellipse cx="100" cy="80" rx="44" ry="32" fill="rgba(255,255,255,0.10)"/>
        {/* sushi pieces */}
        <rect x="72" y="65" width="22" height="12" rx="3" fill="#fbbf24" opacity="0.9"/>
        <rect x="72" y="62" width="22" height="6" rx="2" fill="#ef4444" opacity="0.85"/>
        <rect x="100" y="68" width="22" height="12" rx="3" fill="#fde68a" opacity="0.85"/>
        <rect x="100" y="65" width="22" height="6" rx="2" fill="#f97316" opacity="0.9"/>
        {/* garnish */}
        <circle cx="88" cy="90" r="4" fill="#22c55e" opacity="0.6"/>
        <circle cx="115" cy="88" r="3" fill="#22c55e" opacity="0.5"/>
      </svg>
    )
  }
  return (
    <svg width="100%" height="100%" viewBox="0 0 200 150" preserveAspectRatio="xMidYMid slice">
      <defs>
        <radialGradient id="food1bg" cx="55%" cy="40%" r="65%">
          <stop offset="0%" stopColor="#818cf8"/>
          <stop offset="45%" stopColor="#6366f1"/>
          <stop offset="100%" stopColor="#312e81"/>
        </radialGradient>
      </defs>
      <rect width="200" height="150" fill="url(#food1bg)"/>
      {/* plate */}
      <ellipse cx="100" cy="82" rx="50" ry="36" fill="rgba(255,255,255,0.12)"/>
      {/* steak */}
      <ellipse cx="100" cy="76" rx="32" ry="18" fill="#92400e" opacity="0.8"/>
      <ellipse cx="100" cy="74" rx="28" ry="14" fill="#b45309" opacity="0.7"/>
      {/* grill marks */}
      <line x1="80" y1="70" x2="120" y2="70" stroke="#78350f" strokeWidth="1.5" opacity="0.4"/>
      <line x1="82" y1="76" x2="118" y2="76" stroke="#78350f" strokeWidth="1.5" opacity="0.4"/>
      {/* sides */}
      <circle cx="75" cy="88" r="6" fill="#22c55e" opacity="0.55"/>
      <circle cx="128" cy="86" r="5" fill="#fbbf24" opacity="0.5"/>
      <circle cx="122" cy="92" r="4" fill="#ef4444" opacity="0.45"/>
    </svg>
  )
}

export function LinePhonePreview({
  suggestedActions,
  assetCarousel,
  richMenu,
  suggestedRows,
  botName = 'Bot',
}: LinePhonePreviewProps) {
  const [richMenuOpen, setRichMenuOpen] = useState(false)

  /* Chat UI is dark by default */
  const isDarkMode = suggestedActions.theme_mode === 'dark'
  const LINE_GREEN = '#06c755'

  const headerBg = LINE_GREEN
  const chatBg = isDarkMode ? '#121212' : '#7494a5'
  const botBubbleBg = isDarkMode ? '#1f1f1f' : '#ffffff'
  const botBubbleText = isDarkMode ? '#e5e5e5' : '#1a1a1a'
  const inputBarBg = isDarkMode ? '#171717' : '#ffffff'
  const inputFieldBg = isDarkMode ? '#252525' : '#f0f0f0'
  const inputIconColor = isDarkMode ? '#5a5a5a' : '#8696a0'
  const inputPlaceholderColor = isDarkMode ? '#555' : '#999'
  const dateBadgeBg = isDarkMode ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.22)'
  const senderColor = isDarkMode ? 'rgba(255,255,255,0.50)' : 'rgba(255,255,255,0.72)'

  /* Card colors always use picker values */
  const cardBg = suggestedActions.card_background_color
  const cardText = suggestedActions.card_text_color
  const buttonBg = suggestedActions.button_background_color
  const buttonText = suggestedActions.button_text_color
  const carouselBg = assetCarousel.body_background_color
  const carouselText = assetCarousel.body_text_color

  const ratio = assetCarousel.image_aspect_ratio.replace(':', ' / ')

  /* Rich-menu – respond to theme mode */
  const rmStyleBase = richMenu.styles.normal
  const rmStyle = isDarkMode
    ? {
        background: '#171717',
        text: '#e5e5e5',
        muted_text: '#5a5a5a',
        button_background: '#252525',
        button_text: '#b0b0b0',
        border: '#3a3a3a',
        accent: '#4a9eff',
      }
    : rmStyleBase
  const rmActions = richMenu.layouts.normal
    .map((id) => richMenu.actions.find((a) => a.id === id))
    .filter((a): a is RichMenuActionPreview => a != null && a.enabled)
    .slice(0, 3)

  return (
    <div className="line-phone-preview-wrap">
      <div className="line-iphone-frame">
        <div className="line-iphone-screen">
          {/* Dynamic Island */}
          <div className="line-iphone-island" />

          {/* Status bar */}
          <div className="line-phone-statusbar" style={{ background: headerBg }}>
            <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.2 }}>9:41</span>
            <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
              <svg width="15" height="11" viewBox="0 0 15 11" fill="white">
                <rect x="0" y="7" width="2.8" height="4" rx="0.6"/><rect x="4" y="5" width="2.8" height="6" rx="0.6"/>
                <rect x="8" y="2.5" width="2.8" height="8.5" rx="0.6"/><rect x="12" y="0" width="2.8" height="11" rx="0.6"/>
              </svg>
              <svg width="14" height="11" viewBox="0 0 16 12" fill="white">
                <path d="M8 10.5a1.25 1.25 0 110 2.5 1.25 1.25 0 010-2.5zM4.5 8.2a5 5 0 017 0" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
                <path d="M2 5.5a9 9 0 0112 0" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <svg width="24" height="11" viewBox="0 0 25 12" fill="none">
                <rect x="0.5" y="0.5" width="20" height="11" rx="2" stroke="white" strokeOpacity="0.4"/>
                <rect x="2" y="2" width="15" height="8" rx="1" fill="white"/>
                <rect x="22" y="3.5" width="2.5" height="5" rx="0.8" fill="white" fillOpacity="0.4"/>
              </svg>
            </div>
          </div>

          {/* LINE header */}
          <div className="line-phone-header" style={{ background: headerBg }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
              <span style={{ fontWeight: 600, fontSize: 16, color: '#fff' }}>{botName}</span>
            </div>
            <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/></svg>
            </div>
          </div>

          {/* ── Chat area ──────────────────────────────────────────────── */}
          <div className="line-phone-chat" style={{ background: chatBg }}>
            {/* Date chip */}
            <div style={{ textAlign: 'center', margin: '10px 0 14px' }}>
              <span style={{ background: dateBadgeBg, color: '#fff', fontSize: 10, padding: '3px 10px', borderRadius: 10 }}>Today</span>
            </div>

            {/* Bot welcome */}
            <div className="line-msg-row line-msg-bot">
              <div className="line-msg-avatar" style={{ background: LINE_GREEN }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
              </div>
              <div>
                <div className="line-msg-sender" style={{ color: senderColor }}>{botName}</div>
                <div className="line-msg-bubble line-msg-bubble--bot" style={{ background: botBubbleBg, color: botBubbleText }}>
                  Welcome! How can I help you today?
                </div>
              </div>
            </div>

            {/* User message */}
            <div className="line-msg-row line-msg-user">
              <div className="line-msg-bubble line-msg-bubble--user" style={{ background: LINE_GREEN, color: '#fff' }}>
                Show me the menu
              </div>
            </div>

            {/* Suggested-actions flex card */}
            <div className="line-msg-row line-msg-bot">
              <div className="line-msg-avatar" style={{ background: LINE_GREEN }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
              </div>
              <div style={{ maxWidth: '82%' }}>
                <div className="line-msg-sender" style={{ color: senderColor }}>{botName}</div>
                <div
                  className="line-flex-card"
                  style={{
                    background: cardBg,
                    borderRadius: 14,
                    padding: '10px 10px',
                    maxWidth: 210,
                  }}
                >
                  <div style={{ color: cardText, fontWeight: 700, fontSize: 11.5, marginBottom: 7 }}>
                    What would you like to do?
                  </div>
                  <div style={{ display: 'grid', gap: 4 }}>
                    {suggestedRows.map((row, rIdx) => (
                      <div key={`sr-${rIdx}`} style={{ display: 'grid', gap: 4, gridTemplateColumns: `repeat(${row.length}, minmax(0, 1fr))` }}>
                        {row.map((label, lIdx) => (
                          <div
                            key={`sb-${rIdx}-${lIdx}`}
                            style={{
                              background: buttonBg,
                              color: buttonText,
                              borderRadius: 8,
                              padding: '5px 6px',
                              textAlign: 'center',
                              fontSize: 10,
                              fontWeight: 600,
                              lineHeight: 1.3,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {label}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Asset carousel */}
            <div className="line-msg-row line-msg-bot">
              <div className="line-msg-avatar" style={{ background: LINE_GREEN }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
              </div>
              <div style={{ maxWidth: '88%' }}>
                <div className="line-carousel-scroll">
                {[{ title: 'Sushi Platter', price: '¥1,200', img: 0 as const }, { title: 'Chef Special', price: '¥1,800', img: 1 as const }].map((item) => (
                  <div
                    key={item.title}
                    className="line-carousel-card"
                    style={{
                      borderRadius: 12,
                      overflow: 'hidden',
                      flexShrink: 0,
                      width: 130,
                    }}
                  >
                    <div style={{ aspectRatio: ratio, minHeight: 50, overflow: 'hidden', position: 'relative' }}>
                      <FoodImagePlaceholder variant={item.img} />
                    </div>
                    <div style={{ padding: '6px 8px', background: carouselBg, color: carouselText }}>
                      <div style={{ fontSize: 10, fontWeight: 700, lineHeight: 1.3 }}>{item.title}</div>
                      <div style={{ fontSize: 8.5, opacity: 0.7, marginTop: 1 }}>{item.price}</div>
                    </div>
                  </div>
                ))}
              </div>
              </div>
            </div>

            {/* Typing indicator */}
            <div className="line-msg-row line-msg-bot" style={{ marginTop: 4 }}>
              <div className="line-msg-avatar" style={{ background: LINE_GREEN }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
              </div>
              <div className="line-msg-bubble line-msg-bubble--bot line-typing-bubble" style={{ background: botBubbleBg }}>
                <span className="line-typing-dot" style={{ background: '#999' }} />
                <span className="line-typing-dot" style={{ background: '#999' }} />
                <span className="line-typing-dot" style={{ background: '#999' }} />
              </div>
            </div>
          </div>

          {/* ── Input bar (hidden when rich menu open) ─────────────────── */}
          {!richMenuOpen && (
            <div className="line-phone-inputbar" style={{ background: inputBarBg, borderTop: '1px solid #e5e5e5' }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={inputIconColor} strokeWidth="1.8" strokeLinecap="round">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
              </svg>
              <button
                type="button"
                onClick={() => setRichMenuOpen(true)}
                className="line-richmenu-toggle-btn"
                title="Show menu"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={inputIconColor} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
                  <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
                </svg>
              </button>
              <div className="line-input-field" style={{ background: inputFieldBg, color: inputPlaceholderColor }}>Aa</div>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={inputIconColor} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
              </svg>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={inputIconColor} strokeWidth="1.8" strokeLinecap="round">
                <path d="M12 2a3 3 0 00-3 3v7a3 3 0 006 0V5a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>
              </svg>
            </div>
          )}

          {/* ── Rich menu (replaces input bar) ─────────────────────────── */}
          {richMenuOpen && rmActions.length > 0 && (
            <div
              className="line-phone-richmenu"
              style={{
                background: rmStyle.background,
                borderTop: `1px solid ${rmStyle.border}`,
              }}
            >
              {/* Keyboard toggle row */}
              <div style={{ display: 'flex', justifyContent: 'flex-start', padding: '4px 8px 2px' }}>
                <button
                  type="button"
                  onClick={() => setRichMenuOpen(false)}
                  className="line-richmenu-toggle-btn"
                  title="Show keyboard"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={rmStyle.muted_text} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="4" width="20" height="14" rx="2"/>
                    <line x1="6" y1="8" x2="6.01" y2="8"/><line x1="10" y1="8" x2="10.01" y2="8"/>
                    <line x1="14" y1="8" x2="14.01" y2="8"/><line x1="18" y1="8" x2="18.01" y2="8"/>
                    <line x1="6" y1="12" x2="6.01" y2="12"/><line x1="10" y1="12" x2="10.01" y2="12"/>
                    <line x1="14" y1="12" x2="14.01" y2="12"/><line x1="18" y1="12" x2="18.01" y2="12"/>
                    <line x1="8" y1="16" x2="16" y2="16"/>
                  </svg>
                </button>
              </div>
              <div className="line-richmenu-grid">
                {rmActions.map((action) => (
                  <div
                    key={action.id}
                    className="line-richmenu-btn"
                    style={{
                      background: rmStyle.button_background,
                      border: `1px solid ${rmStyle.border}`,
                      borderRadius: 10,
                    }}
                  >
                    <RichMenuIcon actionId={action.icon} color={rmStyle.accent} size={18} />
                    <span style={{ color: rmStyle.button_text, fontSize: 9, fontWeight: 600, marginTop: 3 }}>
                      {action.labels.en}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Home indicator */}
          <div className="line-phone-homebar" style={{ background: richMenuOpen && rmActions.length > 0 ? rmStyle.background : inputBarBg }}>
            <div className="line-phone-homeindicator" style={{ background: '#1a1a1a' }} />
          </div>
        </div>
      </div>
    </div>
  )
}
