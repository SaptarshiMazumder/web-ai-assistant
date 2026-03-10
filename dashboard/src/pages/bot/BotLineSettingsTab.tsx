import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { type LineChannelTestResult } from '../../hooks/useDashboardData'
import { useTranslation } from 'react-i18next'
import {
  Check, CheckCircle, Copy, ExternalLink, AlertCircle, Loader2,
  Trash2, Zap, MessageCircle, ChevronLeft, ChevronRight,
} from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton, GlassCard, GlassField } from '../../components/ui'
import { useDialog } from '../../contexts/DialogContext'

type LineChannelConfig = {
  channel_id: string
  bot_id: string
  org_id: string
  line_channel_id: string
  is_active: boolean
  created_at: string
  updated_at: string
  managed_rich_menu_enabled?: boolean
  rich_menu_sync_status?: string | null
  rich_menu_last_synced_at?: string | null
  rich_menu_last_error?: string | null
  rich_menu_variants?: Record<string, string>
}

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

const LINE_GREEN = '#06c755'
const LINE_GRADIENT = 'linear-gradient(135deg, #06c755 0%, #00b140 100%)'

/* â”€â”€â”€ Progress Bar (for the 5 card steps only, excludes Get Started) â”€â”€â”€ */
function StepProgress({
  current,
  total,
  labels,
  getProgressText,
}: {
  current: number
  total: number
  labels: string[]
  getProgressText: (step: number, totalSteps: number, currentLabel: string) => string
}) {
  return (
    <div style={{ marginBottom: '2rem' }}>
      {/* Step dots */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
        {Array.from({ length: total }).map((_, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: i === current ? '32px' : '10px',
              height: '10px',
              borderRadius: i === current ? '5px' : '50%',
              background: i < current ? LINE_GREEN : i === current ? LINE_GREEN : 'var(--ui-flow-border)',
              transition: 'all 0.3s ease',
              opacity: i <= current ? 1 : 0.4,
            }} />
          </div>
        ))}
      </div>
      {/* Step label */}
      <div style={{
        textAlign: 'center',
        marginTop: '0.75rem',
        fontSize: '0.85rem',
        fontWeight: 600,
        color: LINE_GREEN,
        letterSpacing: '0.03em',
      }}>
        {getProgressText(current + 1, total, labels[current])}
      </div>
    </div>
  )
}

type BotLineSettingsTabProps = {
  botIdOverride?: string | null
  onConnected?: () => void
}

export default function BotLineSettingsTab({ botIdOverride, onConnected }: BotLineSettingsTabProps = {}) {
  const { botId: routeBotId } = useParams()
  const botId = String(botIdOverride || routeBotId || '').trim()
  const dialog = useDialog()
  const { getAccessTokenSilently } = useAuth0()
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)
  const cardStepLabels = [
    tr('Enable API', 'APIã‚’æœ‰åŠ¹åŒ–'),
    tr('Auto-reply', 'è‡ªå‹•è¿”ä¿¡'),
    tr('Credentials', 'èªè¨¼æƒ…å ±'),
    tr('Webhook', 'Webhook'),
    tr('Connect', 'æŽ¥ç¶š'),
  ]
  const getProgressText = (step: number, totalSteps: number, currentLabel: string) =>
    isJa ? `ã‚¹ãƒ†ãƒƒãƒ— ${step}/${totalSteps} - ${currentLabel}` : `Step ${step} of ${totalSteps} - ${currentLabel}`

  const [lineChannelId, setLineChannelId] = useState('')
  const [lineChannelSecret, setLineChannelSecret] = useState('')
  const [lineAccessToken, setLineAccessToken] = useState('')
  const [isActive, setIsActive] = useState(true)
  const [existing, setExisting] = useState<LineChannelConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<LineChannelTestResult | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Wizard state
  const [currentStep, setCurrentStep] = useState(0)
  const [apiEnabled, setApiEnabled] = useState(false)
  const [webhookSet, setWebhookSet] = useState(false)
  const [autoReplyOff, setAutoReplyOff] = useState(false)

  const webhookUrl = botId ? `${API_BASE}/webhooks/line/${botId}` : ''

  const authedFetch = useCallback(async (path: string, init?: RequestInit): Promise<Response> => {
    const token = await getAccessTokenSilently()
    return fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers || {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    })
  }, [getAccessTokenSilently])

  const fetchLineAccountInfo = useCallback(async (showSpinner = false): Promise<LineChannelTestResult | null> => {
    if (!botId) return null
    if (showSpinner) setTesting(true)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel/test`, { method: 'POST' })
      const data = await resp.json() as LineChannelTestResult
      setTestResult(data)
      return data
    } catch (err) {
      const failed = { ok: false, message: (err as Error).message } satisfies LineChannelTestResult
      setTestResult(failed)
      return failed
    } finally {
      if (showSpinner) setTesting(false)
    }
  }, [botId, authedFetch])

  const loadConfig = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel`)
      if (resp.status === 404) {
        setExisting(null)
        setTestResult(null)
        return
      }
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as LineChannelConfig
      setExisting(data)
      setLineChannelId(data.line_channel_id)
      setIsActive(data.is_active)
      await fetchLineAccountInfo(false)
    } catch (err) {
      if ((err as Error).message?.includes('404') || (err as Error).message?.includes('Not Found')) {
        setExisting(null)
        setTestResult(null)
      } else {
        setError((err as Error).message)
      }
    } finally {
      setLoading(false)
    }
  }, [botId, authedFetch, fetchLineAccountInfo])

  useEffect(() => {
    if (!botId) return
    setError(null)
    setSuccess(null)
    void loadConfig()
  }, [botId, loadConfig])

  async function handleSave() {
    if (!botId) return
    setSaving(true)
    setError(null)
    setSuccess(null)
    setTestResult(null)
    try {
      const body: Record<string, unknown> = {
        line_channel_id: lineChannelId.trim(),
        line_channel_secret: lineChannelSecret.trim(),
        line_channel_access_token: lineAccessToken.trim(),
        is_active: isActive,
      }
      if (!body.line_channel_id) throw new Error(tr('Channel ID is required', 'ãƒãƒ£ãƒãƒ«IDã¯å¿…é ˆã§ã™'))
      if (!existing && (!body.line_channel_secret || !body.line_channel_access_token)) {
        throw new Error(tr('Channel Secret and Access Token are required for initial setup', 'åˆæœŸè¨­å®šã«ã¯Channel Secretã¨Access TokenãŒå¿…è¦ã§ã™'))
      }
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel`, {
        method: 'PUT',
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as LineChannelConfig
      setExisting(data)
      setLineChannelSecret('')
      setLineAccessToken('')
      await fetchLineAccountInfo(false)
      setCurrentStep(0)
      setSuccess(tr('Connected! Your bot is live on LINE.', 'æŽ¥ç¶šå®Œäº†ã€‚ãƒœãƒƒãƒˆã¯LINEã§ç¨¼åƒä¸­ã§ã™ã€‚'))
      if (currentStep === 5 || !existing) {
        onConnected?.()
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function handleTestConnection() {
    if (!botId) return
    setTestResult(null)
    setError(null)
    await fetchLineAccountInfo(true)
  }

  async function handleDelete() {
    if (!botId) return
    const confirmed = await dialog.confirm({
      title: tr('Disconnect LINE integration? Your bot will stop responding on LINE.', 'LINEé€£æºã‚’è§£é™¤ã—ã¾ã™ã‹ï¼Ÿãƒœãƒƒãƒˆã¯LINEã§è¿”ä¿¡ã—ãªããªã‚Šã¾ã™ã€‚'),
      confirmLabel: tr('Disconnect', 'é€£æºè§£é™¤'),
      cancelLabel: tr('Cancel', 'ã‚­ãƒ£ãƒ³ã‚»ãƒ«'),
      tone: 'danger',
    })
    if (!confirmed) return
    setDeleting(true)
    setError(null)
    setSuccess(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel`, { method: 'DELETE' })
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      setExisting(null)
      setLineChannelId('')
      setLineChannelSecret('')
      setLineAccessToken('')
      setIsActive(true)
      setTestResult(null)
      setCurrentStep(0)
      setApiEnabled(false)
      setWebhookSet(false)
      setAutoReplyOff(false)
      setSuccess(tr('LINE integration disconnected.', 'LINEé€£æºã‚’è§£é™¤ã—ã¾ã—ãŸã€‚'))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  function copyWebhookUrl() {
    navigator.clipboard.writeText(webhookUrl)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  async function handleNext() {
    if (currentStep === 3 && canAdvance()) {
      setSaving(true)
      setError(null)
      setSuccess(null)
      try {
        const body: Record<string, unknown> = {
          line_channel_id: lineChannelId.trim(),
          line_channel_secret: lineChannelSecret.trim(),
          line_channel_access_token: lineAccessToken.trim(),
          is_active: true,
        }
        const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel`, {
          method: 'PUT',
          body: JSON.stringify(body),
        })
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}))
          throw new Error((data as { detail?: string }).detail || resp.statusText)
        }
        const data = (await resp.json()) as LineChannelConfig
        setExisting(data)
        await fetchLineAccountInfo(false)
        setCurrentStep(4)
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setSaving(false)
      }
      return
    }
    setCurrentStep(currentStep + 1)
  }

  // Can the user advance to the next step?
  function canAdvance(): boolean {
    switch (currentStep) {
      case 0: return true // just informational
      case 1: return apiEnabled
      case 2: return autoReplyOff
      case 3: return !!lineChannelId.trim() && !!lineChannelSecret.trim() && !!lineAccessToken.trim()
      case 4: return webhookSet
      default: return false
    }
  }

  if (!botId) {
    return <div className="empty-panel">{tr('Select a bot to configure LINE integration.', 'LINEé€£æºã‚’è¨­å®šã™ã‚‹ãƒœãƒƒãƒˆã‚’é¸æŠžã—ã¦ãã ã•ã„ã€‚')}</div>
  }

  if (loading) {
    return (
      <AnimatedPage className="page-body">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', gap: '0.75rem', color: 'var(--text-secondary)' }}>
          <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
          {tr('Loading LINE settings...', 'LINEè¨­å®šã‚’èª­ã¿è¾¼ã¿ä¸­...')}
        </div>
      </AnimatedPage>
    )
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     Connected View
     â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  const lineAccountName = testResult?.display_name || testResult?.basic_id || existing?.line_channel_id || lineChannelId.trim()
  const lineAccountPictureUrl = testResult?.picture_url || null
  const connectedDateLabel = existing?.created_at
    ? new Date(existing.created_at).toLocaleDateString(isJa ? 'ja-JP' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    : tr('Not available', 'Not available')

  function renderLineAccountCard(
    title: string,
    subtitle: string,
    options?: { compact?: boolean; background?: string; border?: string; textColor?: string; mutedColor?: string }
  ) {
    if (!testResult?.ok || !lineAccountName) return null
    const compact = options?.compact === true
    const background = options?.background || 'rgba(255,255,255,0.14)'
    const border = options?.border || '1px solid rgba(255,255,255,0.22)'
    const textColor = options?.textColor || '#fff'
    const mutedColor = options?.mutedColor || 'rgba(255,255,255,0.78)'
    return (
      <div
        style={{
          marginTop: compact ? '1rem' : 0,
          padding: compact ? '0.9rem 1rem' : '1rem',
          borderRadius: compact ? '14px' : '16px',
          background,
          border,
          display: 'grid',
          gap: '0.8rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          {lineAccountPictureUrl ? (
            <img
              src={lineAccountPictureUrl}
              alt={lineAccountName}
              style={{
                width: compact ? '52px' : '64px',
                height: compact ? '52px' : '64px',
                borderRadius: '50%',
                objectFit: 'cover',
                border: compact ? '2px solid rgba(255,255,255,0.35)' : '3px solid rgba(255,255,255,0.35)',
                background: '#fff',
                flexShrink: 0,
              }}
            />
          ) : (
            <div
              style={{
                width: compact ? '52px' : '64px',
                height: compact ? '52px' : '64px',
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.24)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              <MessageCircle size={compact ? 24 : 28} color={textColor} />
            </div>
          )}
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: compact ? '0.78rem' : '0.82rem', fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: mutedColor }}>
              {title}
            </div>
            <div style={{ fontSize: compact ? '1rem' : '1.15rem', fontWeight: 700, color: textColor, lineHeight: 1.25, wordBreak: 'break-word' }}>
              {lineAccountName}
            </div>
            <div style={{ marginTop: '0.2rem', fontSize: compact ? '0.84rem' : '0.9rem', color: mutedColor, wordBreak: 'break-word' }}>
              {subtitle}
            </div>
          </div>
        </div>
        <div style={{ display: 'grid', gap: '0.45rem', fontSize: compact ? '0.83rem' : '0.88rem', color: textColor }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
            <span style={{ color: mutedColor }}>{tr('Basic ID', 'Basic ID')}</span>
            <span style={{ fontWeight: 600, wordBreak: 'break-all', textAlign: 'right' }}>{testResult?.basic_id || tr('Not available', 'æœªå–å¾—')}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
            <span style={{ color: mutedColor }}>{tr('Channel ID', 'ãƒãƒ£ãƒãƒ«ID')}</span>
            <span style={{ fontWeight: 600, wordBreak: 'break-all', textAlign: 'right' }}>{existing?.line_channel_id || lineChannelId.trim() || tr('Not available', 'æœªå–å¾—')}</span>
          </div>
        </div>
      </div>
    )
  }

  if (existing && currentStep === 0) {
    return (
      <AnimatedPage className="page-body">
        <SectionHeader
          eyebrow={tr('Integrations', 'é€£æº')}
          title={tr('LINE channel', 'LINEãƒãƒ£ãƒ³ãƒãƒ«')}
          subtitle={tr('Your bot is live and responding to messages on LINE.', 'ãƒœãƒƒãƒˆã¯LINEãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ã«è‡ªå‹•è¿”ä¿¡ä¸­ã§ã™ã€‚')}
        />

        {/* Status Hero */}
        <div style={{
          background: LINE_GRADIENT,
          borderRadius: '18px',
          padding: '2rem',
          marginBottom: '2rem',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 20px 60px rgba(6, 199, 85, 0.25)',
        }}>
          <div style={{
            position: 'absolute', top: '-50%', right: '-20%',
            width: '500px', height: '500px',
            background: 'radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 60%)',
            borderRadius: '50%', pointerEvents: 'none',
          }} />
          <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
              {lineAccountPictureUrl ? (
                <img
                  src={lineAccountPictureUrl}
                  alt={lineAccountName}
                  style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '50%',
                    objectFit: 'cover',
                    border: '3px solid rgba(255,255,255,0.35)',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
                    background: '#fff',
                    flexShrink: 0,
                  }}
                />
              ) : (
                <div style={{
                  width: '60px', height: '60px', borderRadius: '16px',
                  background: 'rgba(255,255,255,0.25)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  backdropFilter: 'blur(10px)',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
                }}>
                  <MessageCircle size={30} color="#fff" />
                </div>
              )}
              <div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', marginBottom: '0.25rem' }}>
                  {existing.is_active ? tr('Connected & Active', 'Connected & Active') : tr('Connected & Paused', 'Connected & Paused')}
                </div>
                <div style={{ color: 'rgba(255,255,255,0.92)', fontSize: '1rem', fontWeight: 700, lineHeight: 1.25, wordBreak: 'break-word' }}>
                  {lineAccountName}
                </div>
                <div style={{ color: 'rgba(255,255,255,0.82)', fontSize: '0.84rem', fontWeight: 600, marginTop: '0.35rem', wordBreak: 'break-all' }}>
                  {testResult?.basic_id ? `${testResult.basic_id} • ` : ''}{tr('Channel ID', 'Channel ID')}: {existing.line_channel_id}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button
                onClick={handleTestConnection}
                disabled={testing}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  backdropFilter: 'blur(10px)',
                  border: '2px solid rgba(255,255,255,0.3)',
                  borderRadius: '12px',
                  padding: '0.75rem 1.5rem',
                  color: '#fff', fontWeight: 600, fontSize: '0.95rem',
                  cursor: testing ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', gap: '0.6rem',
                  transition: 'all 0.2s',
                  opacity: testing ? 0.7 : 1,
                }}
                onMouseEnter={(e) => { if (!testing) { e.currentTarget.style.background = 'rgba(255,255,255,0.3)'; e.currentTarget.style.transform = 'translateY(-2px)' } }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.2)'; e.currentTarget.style.transform = 'translateY(0)' }}
              >
                {testing ? (<><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Testing...', 'Testing...')}</>) : (<><Zap size={18} /> {tr('Test Connection', 'Test Connection')}</>)}
              </button>
            </div>
          </div>
        </div>

        {testResult && (
          <div style={{ marginBottom: '1.5rem', color: testResult.ok ? LINE_GREEN : '#e74c3c', fontWeight: 600, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {testResult.ok ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
            {testResult.message}
          </div>
        )}
        {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
        {success && <div style={{ marginBottom: '1.5rem', color: LINE_GREEN, fontWeight: 600 }}>{success}</div>}

        <div style={{ display: 'grid', gap: '1.5rem' }}>
          {/* Connection Details */}
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.9rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
              <div>
                <div className="card-title" style={{ marginBottom: '0.35rem' }}>{tr('Connection details', 'Connection details')}</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.45 }}>
                  {tr('This is the LINE account connected to your bot.', 'This is the LINE account connected to your bot.')}
                </div>
              </div>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
                borderRadius: '999px', padding: '0.35rem 0.72rem',
                background: existing.is_active ? 'rgba(39,174,96,0.14)' : 'rgba(231,76,60,0.12)',
                border: existing.is_active ? '1px solid rgba(39,174,96,0.28)' : '1px solid rgba(231,76,60,0.28)',
                color: existing.is_active ? '#1f7a45' : '#b42318',
                fontSize: '0.79rem', fontWeight: 700,
              }}>
                {existing.is_active ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                {existing.is_active ? tr('Active', 'Active') : tr('Paused', 'Paused')}
              </span>
            </div>

            <div style={{
              marginBottom: '1rem',
              display: 'flex', alignItems: 'center', gap: '0.9rem', flexWrap: 'wrap',
              padding: '0.95rem 1rem',
              borderRadius: '14px',
              border: '1px solid rgba(6, 199, 85, 0.24)',
              background: 'linear-gradient(135deg, rgba(6, 199, 85, 0.12) 0%, rgba(6, 199, 85, 0.03) 100%)',
            }}>
              {lineAccountPictureUrl ? (
                <img
                  src={lineAccountPictureUrl}
                  alt={lineAccountName}
                  style={{
                    width: '56px',
                    height: '56px',
                    borderRadius: '50%',
                    objectFit: 'cover',
                    border: '2px solid rgba(6,199,85,0.24)',
                    background: '#fff',
                    flexShrink: 0,
                  }}
                />
              ) : (
                <div style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(6,199,85,0.12)',
                  border: '1px solid rgba(6,199,85,0.22)',
                  flexShrink: 0,
                }}>
                  <MessageCircle size={24} color={LINE_GREEN} />
                </div>
              )}
              <div style={{ minWidth: 0, flex: '1 1 240px' }}>
                <div style={{ fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, color: 'var(--text-secondary)' }}>
                  {tr('Connected LINE account', 'Connected LINE account')}
                </div>
                <div style={{ marginTop: '0.2rem', fontSize: '1.03rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.25, wordBreak: 'break-word' }}>
                  {lineAccountName || tr('Not available', 'Not available')}
                </div>
                {testResult?.basic_id ? (
                  <div style={{ marginTop: '0.25rem', fontSize: '0.84rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                    {tr('Basic ID', 'Basic ID')}: {testResult.basic_id}
                  </div>
                ) : null}
              </div>
              <code style={{
                padding: '0.45rem 0.65rem',
                borderRadius: '8px',
                background: 'rgba(255,255,255,0.7)',
                border: '1px solid rgba(6,199,85,0.25)',
                fontSize: '0.78rem',
                fontFamily: 'monospace',
                wordBreak: 'break-all',
              }}>
                {existing.line_channel_id}
              </code>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '0.75rem' }}>
              <div style={{ padding: '0.8rem 0.9rem', borderRadius: '12px', border: '1px solid var(--ui-flow-border)', background: 'var(--ui-flow-surface)' }}>
                <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-secondary)', fontWeight: 700, marginBottom: '0.3rem' }}>{tr('Connected', 'Connected')}</div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.4 }}>{connectedDateLabel}</div>
              </div>
            </div>
          </GlassCard>

          {/* Webhook URL */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1rem' }}>{tr('Webhook URL', 'Webhook URL')}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <code style={{
                flex: 1, padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)', borderRadius: '10px',
                fontSize: '0.9rem', wordBreak: 'break-all',
                border: '1.5px solid var(--ui-flow-border)',
              }}>
                {webhookUrl}
              </code>
              <UiButton variant={copied ? 'primary' : 'secondary'} onClick={copyWebhookUrl} style={{ padding: '0.85rem 1.1rem' }}>
                {copied ? <Check size={18} /> : <Copy size={18} />}
              </UiButton>
            </div>
          </GlassCard>

          {/* Manage Connection */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1rem' }}>{tr('Manage Connection', 'æŽ¥ç¶šç®¡ç†')}</div>

            {/* Update credentials (collapsed by default) */}
            <details style={{ marginBottom: '1rem' }}>
              <summary style={{ cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '1rem' }}>
                {tr('Update credentials', 'èªè¨¼æƒ…å ±ã‚’æ›´æ–°')}
              </summary>
              <div style={{ display: 'grid', gap: '1rem', paddingTop: '0.5rem' }}>
                <GlassField label={tr('Channel ID', 'ãƒãƒ£ãƒãƒ«ID')}>
                  <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} />
                </GlassField>
                <GlassField label={tr('Channel Secret', 'ãƒãƒ£ãƒãƒ«ã‚·ãƒ¼ã‚¯ãƒ¬ãƒƒãƒˆ')}>
                  <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)} placeholder={tr('Leave blank to keep current', 'ç©ºæ¬„ã§ç¾åœ¨ã®å€¤ã‚’ä¿æŒ')} />
                </GlassField>
                <GlassField label={tr('Channel Access Token', 'ãƒãƒ£ãƒãƒ«ã‚¢ã‚¯ã‚»ã‚¹ãƒˆãƒ¼ã‚¯ãƒ³')}>
                  <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)} placeholder={tr('Leave blank to keep current', 'ç©ºæ¬„ã§ç¾åœ¨ã®å€¤ã‚’ä¿æŒ')} />
                </GlassField>
                <UiButton variant="primary" onClick={handleSave} disabled={saving}>
                  {saving ? tr('Saving...', 'ä¿å­˜ä¸­...') : tr('Save Changes', 'å¤‰æ›´ã‚’ä¿å­˜')}
                </UiButton>
              </div>
            </details>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <UiButton
                variant="secondary"
                onClick={handleDelete}
                disabled={deleting}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#e74c3c', borderColor: '#e74c3c' }}
              >
                <Trash2 size={16} />
                {deleting ? tr('Removing...', 'è§£é™¤ä¸­...') : tr('Disconnect', 'é€£æºè§£é™¤')}
              </UiButton>
            </div>
          </GlassCard>
        </div>
      </AnimatedPage>
    )
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     Setup Wizard (not connected)
     â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  return (
    <AnimatedPage className="page-body">
      <SectionHeader
        eyebrow={tr('Integrations', 'é€£æº')}
        title={tr('Connect LINE', 'LINEã«æŽ¥ç¶š')}
        subtitle={tr('Follow the guided steps below to connect your LINE account.', 'ä»¥ä¸‹ã®ã‚¬ã‚¤ãƒ‰æ‰‹é †ã§LINEã‚¢ã‚«ã‚¦ãƒ³ãƒˆã‚’æŽ¥ç¶šã—ã¦ãã ã•ã„ã€‚')}
      />

      {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
      {success && <div style={{ marginBottom: '1.5rem', color: LINE_GREEN, fontWeight: 600 }}>{success}</div>}

      {currentStep > 0 && <StepProgress current={currentStep - 1} total={5} labels={cardStepLabels} getProgressText={getProgressText} />}

      <GlassCard>
        {/* â”€â”€ Step 0: Get Started â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {currentStep === 0 && (
          <div style={{ padding: '2rem 1rem' }}>
            {/* Hero */}
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <div style={{
                width: '80px', height: '80px', borderRadius: '20px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                margin: '0 auto 1.25rem',
                boxShadow: '0 12px 40px rgba(6, 199, 85, 0.3)',
              }}>
                <MessageCircle size={40} color="#fff" />
              </div>
              <h3 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 0.5rem 0' }}>
                {tr('Connect your LINE account', 'LINEã‚¢ã‚«ã‚¦ãƒ³ãƒˆã‚’æŽ¥ç¶š')}
              </h3>
              <p style={{
                margin: 0, color: 'var(--text-secondary)',
                fontSize: '1rem', maxWidth: '480px', marginLeft: 'auto', marginRight: 'auto', lineHeight: 1.6,
              }}>
                {tr(
                  'We\'ll walk you through very simple steps to connect your LINE business account, so your AI Agent can reply to messages automatically.',
                  'LINEãƒ“ã‚¸ãƒã‚¹ã‚¢ã‚«ã‚¦ãƒ³ãƒˆã‚’æŽ¥ç¶šã™ã‚‹æ‰‹é †ã‚’ã‚ã‹ã‚Šã‚„ã™ãæ¡ˆå†…ã—ã¾ã™ã€‚æŽ¥ç¶šå¾Œã¯AIã‚¨ãƒ¼ã‚¸ã‚§ãƒ³ãƒˆãŒè‡ªå‹•ã§è¿”ä¿¡ã—ã¾ã™ã€‚',
                )}
              </p>
            </div>

            {/* What you need */}
            <div style={{
              padding: '1rem 1.25rem',
              background: 'rgba(6,199,85,0.07)',
              borderRadius: '12px',
              border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.25rem',
              fontSize: '0.92rem',
              lineHeight: 1.7,
            }}>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <AlertCircle size={16} color={LINE_GREEN} /> {tr('What you need before starting', 'é–‹å§‹å‰ã«å¿…è¦ãªã‚‚ã®')}
              </div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {isJa ? (
                  <>LINE Official Accountï¼ˆå€‹äººç”¨LINEã‚¢ãƒ—ãƒªã¨ã¯åˆ¥ã®ãƒ“ã‚¸ãƒã‚¹ã‚¢ã‚«ã‚¦ãƒ³ãƒˆï¼‰</>
                ) : (
                  <>A <strong>LINE Official Account</strong> - this is a business account (different from your personal LINE app).</>
                )}
              </div>
              
            </div>

            {/* What the 5 steps cover */}
            <div style={{
              padding: '1rem 1.25rem',
              background: 'var(--ui-flow-surface)',
              borderRadius: '12px',
              border: '1px solid var(--ui-flow-border)',
              marginBottom: '2rem',
              fontSize: '0.9rem',
            }}>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.75rem' }}>{tr('Here\'s what we\'ll do in 5 simple steps:', '5ã¤ã®ç°¡å˜ãªæ‰‹é †ã§é€²ã‚ã¾ã™:')}</div>
              <div style={{ display: 'grid', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                {[
                  ['1', tr('Enable Messaging API (manager.line.biz)', 'Messaging APIã‚’æœ‰åŠ¹åŒ–ï¼ˆmanager.line.bizï¼‰')],
                  ['2', tr('Turn off Auto-reply (manager.line.biz)', 'è‡ªå‹•è¿”ä¿¡ã‚’OFFã«ã™ã‚‹ï¼ˆmanager.line.bizï¼‰')],
                  ['3', tr('Copy 3 codes (Developers Console)', '3ã¤ã®ã‚³ãƒ¼ãƒ‰ã‚’ã‚³ãƒ”ãƒ¼ï¼ˆDevelopers Consoleï¼‰')],
                  ['4', tr('Set your bot\'s address - webhook (LINE will verify)', 'ãƒœãƒƒãƒˆã®Webhook URLã‚’è¨­å®šï¼ˆLINEå´ã§æ¤œè¨¼ï¼‰')],
                  ['5', tr('Click "Activate" and you\'re done!', 'ã€Œæœ‰åŠ¹åŒ–ã€ã‚’æŠ¼ã—ã¦å®Œäº†')],
                ].map(([num, desc]) => (
                  <div key={num} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem' }}>
                    <div style={{
                      width: '22px', height: '22px', borderRadius: '50%',
                      background: LINE_GRADIENT, color: '#fff',
                      fontSize: '0.75rem', fontWeight: 700,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      flexShrink: 0, marginTop: '1px',
                    }}>{num}</div>
                    <span style={{ lineHeight: 1.5 }}>{desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* CTA */}
            <div style={{ textAlign: 'center' }}>
              <button
                onClick={() => setCurrentStep(1)}
                style={{
                  background: LINE_GRADIENT,
                  border: 'none', borderRadius: '14px',
                  padding: '1rem 2.5rem', color: '#fff',
                  fontWeight: 700, fontSize: '1.1rem', cursor: 'pointer',
                  display: 'inline-flex', alignItems: 'center', gap: '0.75rem',
                  boxShadow: '0 8px 32px rgba(6, 199, 85, 0.35)',
                  transition: 'all 0.25s',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 12px 40px rgba(6, 199, 85, 0.5)' }}
                onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 8px 32px rgba(6, 199, 85, 0.35)' }}
              >
                {tr('I have a LINE Official Account - Let\'s start', 'LINEå…¬å¼ã‚¢ã‚«ã‚¦ãƒ³ãƒˆãŒã‚ã‚Šã¾ã™ã€‚é–‹å§‹ã™ã‚‹')}
                <ChevronRight size={20} />
              </button>
              <div style={{ marginTop: '0.75rem', fontSize: '0.83rem', color: 'var(--text-secondary)' }}>
                {tr('Don\'t have one yet?', 'ã¾ã æŒã£ã¦ã„ã¾ã›ã‚“ã‹ï¼Ÿ')}{' '}
                <a href="https://www.linebiz.com/jp/entry/" target="_blank" rel="noopener noreferrer"
                  style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none' }}>
                  {tr('Create it for free first', 'ç„¡æ–™ã§ä½œæˆ')} <ExternalLink size={11} style={{ display: 'inline', verticalAlign: 'middle' }} />
                </a>
              </div>
            </div>
          </div>
        )}

        {/* â”€â”€ Step 1: Enable Messaging API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {currentStep === 1 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>1</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Enable Messaging API', 'Messaging APIã‚’æœ‰åŠ¹åŒ–')}</h3>
            </div>

            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'rgba(6,199,85,0.07)',
              borderRadius: '10px', border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.5rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6,
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px', color: LINE_GREEN }} />
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>{tr('Important:', 'é‡è¦:')}</strong> {tr('Use a computer browser - the Messaging API option is not available in the LINE mobile app.', 'PCãƒ–ãƒ©ã‚¦ã‚¶ã‚’ä½¿ç”¨ã—ã¦ãã ã•ã„ã€‚LINEãƒ¢ãƒã‚¤ãƒ«ã‚¢ãƒ—ãƒªã§ã¯Messaging APIè¨­å®šãŒåˆ©ç”¨ã§ãã¾ã›ã‚“ã€‚')}
              </span>
            </div>

            <div style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 2, marginBottom: '1.5rem' }}>
              <ol style={{ margin: 0, paddingLeft: '1.4rem' }}>
                <li>
                  Open{' '}
                  <a href="https://manager.line.biz/" target="_blank" rel="noopener noreferrer"
                    style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                    manager.line.biz <ExternalLink size={13} />
                  </a>
                  {' '}{tr('(the Official Account manager) and sign in', 'ï¼ˆå…¬å¼ã‚¢ã‚«ã‚¦ãƒ³ãƒˆãƒžãƒãƒ¼ã‚¸ãƒ£ãƒ¼ï¼‰ã«ãƒ­ã‚°ã‚¤ãƒ³')}
                </li>
                <li>{tr('Click your business account name', 'ãƒ“ã‚¸ãƒã‚¹ã‚¢ã‚«ã‚¦ãƒ³ãƒˆåã‚’ã‚¯ãƒªãƒƒã‚¯')}</li>
                <li>{tr('Click Settings in the top-right corner', 'å³ä¸Šã®è¨­å®šã‚’ã‚¯ãƒªãƒƒã‚¯')}</li>
                <li>{tr('In the left menu, click "Messaging API"', 'å·¦ãƒ¡ãƒ‹ãƒ¥ãƒ¼ã§ã€ŒMessaging APIã€ã‚’ã‚¯ãƒªãƒƒã‚¯')}</li>
                <li>{tr('Click the green "Enable Messaging API" button', 'ç·‘è‰²ã®ã€ŒEnable Messaging APIã€ã‚’ã‚¯ãƒªãƒƒã‚¯')}</li>
                <li>{tr('Enter a Provider name (company/brand) and click OK', 'Provideråï¼ˆä¼šç¤¾/ãƒ–ãƒ©ãƒ³ãƒ‰åï¼‰ã‚’å…¥åŠ›ã—ã¦OK')}</li>
              </ol>

              <div style={{
                margin: '1.25rem 0 0 0',
                padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)',
                borderRadius: '10px',
                border: '1px solid var(--ui-flow-border)',
                fontSize: '0.88rem',
              }}>
                <strong>{tr('Done when:', 'å®Œäº†æ¡ä»¶:')}</strong> {tr('You see a page with Channel ID and Channel Secret.', 'Channel IDã¨Channel SecretãŒè¡¨ç¤ºã•ã‚ŒãŸã‚‰å®Œäº†ã§ã™ã€‚')}
              </div>
            </div>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
            }}>
              <input type="checkbox" checked={apiEnabled} onChange={(e) => setApiEnabled(e.target.checked)} />
              {tr('Messaging API enabled - I can see Channel ID and Channel Secret', 'Messaging APIã‚’æœ‰åŠ¹åŒ–ã—ã€Channel ID/Channel Secretã‚’ç¢ºèªã—ã¾ã—ãŸ')}
            </label>
          </div>
        )}

        {/* â”€â”€ Step 2: Turn off Auto-reply â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {currentStep === 2 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>2</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Turn off Auto-reply messages', 'è‡ªå‹•è¿”ä¿¡ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ã‚’OFFã«ã™ã‚‹')}</h3>
            </div>

            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'rgba(6,199,85,0.07)',
              borderRadius: '10px', border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.5rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6,
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px', color: LINE_GREEN }} />
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>{tr('Why?', 'ç†ç”±:')}</strong>{' '}
                {tr('LINE sends a default "Thanks for your message!" when someone messages you. Turn it off so only your bot replies - otherwise customers get two replies.', 'LINEã®åˆæœŸè¨­å®šã§ã¯ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸å—ä¿¡æ™‚ã«è‡ªå‹•è¿”ä¿¡ã•ã‚Œã¾ã™ã€‚ã“ã‚Œã‚’OFFã«ã—ã¦ã€ãƒœãƒƒãƒˆã®ã¿ãŒè¿”ä¿¡ã™ã‚‹ã‚ˆã†ã«ã—ã¦ãã ã•ã„ã€‚')}
              </span>
            </div>

            <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              {tr('Still in', 'å¼•ãç¶šã')}{' '}
              <a href="https://manager.line.biz/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                manager.line.biz <ExternalLink size={13} />
              </a>
              :
            </p>
            <ol style={{ margin: 0, paddingLeft: '1.4rem', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 2 }}>
              <li>{tr('Click Settings -> "Response settings" in the left menu', 'å·¦ãƒ¡ãƒ‹ãƒ¥ãƒ¼ã®ã€ŒResponse settingsã€ã‚’é–‹ã')}</li>
              <li>{tr('Find "Auto-response messages" and turn it OFF', 'ã€ŒAuto-response messagesã€ã‚’OFFã«ã™ã‚‹')}</li>
            </ol>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
              marginTop: '1rem',
            }}>
              <input type="checkbox" checked={autoReplyOff} onChange={(e) => setAutoReplyOff(e.target.checked)} />
              {tr('Auto-response messages is OFF', 'Auto-response messagesã‚’OFFã«ã—ã¾ã—ãŸ')}
            </label>
          </div>
        )}

        {/* â”€â”€ Step 3: Copy Credentials (must save before webhook verify) â”€ */}
        {currentStep === 3 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>3</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Copy the 3 codes', '3ã¤ã®ã‚³ãƒ¼ãƒ‰ã‚’ã‚³ãƒ”ãƒ¼')}</h3>
            </div>

            <p style={{ margin: '0 0 1rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              {tr('Go to', 'æ¬¡ã¸ã‚¢ã‚¯ã‚»ã‚¹')}{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                developers.line.biz/console <ExternalLink size={13} />
              </a>
              {' '}{tr('and do the following:', 'ã—ã¦ã€ä»¥ä¸‹ã‚’å®Ÿæ–½ã—ã¦ãã ã•ã„:')}
            </p>

            <div style={{
              marginBottom: '1.5rem',
              padding: '1rem 1.25rem',
              background: 'var(--ui-flow-surface)',
              borderRadius: '12px',
              border: '1px solid var(--ui-flow-border)',
              fontSize: '0.92rem',
              lineHeight: 1.7,
            }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>{tr('A. Select or create a Provider', 'A. Providerã‚’é¸æŠžã¾ãŸã¯ä½œæˆ')}</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {tr('In the left panel, you will see a list of Providers. Select an existing one, or click "Create" to make a new one.', 'å·¦å´ã®ä¸€è¦§ã‹ã‚‰Providerã‚’é¸æŠžã™ã‚‹ã‹ã€ã€ŒCreateã€ã§æ–°è¦ä½œæˆã—ã¾ã™ã€‚')}
              </div>
            </div>

            <div style={{
              marginBottom: '1.5rem',
              padding: '1rem 1.25rem',
              background: 'var(--ui-flow-surface)',
              borderRadius: '12px',
              border: '1px solid var(--ui-flow-border)',
              fontSize: '0.92rem',
              lineHeight: 1.7,
            }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>{tr('B. Select your Messaging API channel', 'B. Messaging APIãƒãƒ£ãƒãƒ«ã‚’é¸æŠž')}</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {tr('Under your Provider, open the Messaging API channel you created in Step 1.', 'Step1ã§ä½œæˆã—ãŸMessaging APIãƒãƒ£ãƒãƒ«ã‚’é–‹ã„ã¦ãã ã•ã„ã€‚')}
              </div>
            </div>

            <p style={{ margin: '0 0 1rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              {tr('C. Copy these 3 values from the channel page and paste them below:', 'C. ãƒãƒ£ãƒãƒ«ç”»é¢ã®3é …ç›®ã‚’ã‚³ãƒ”ãƒ¼ã—ã¦ä»¥ä¸‹ã«è²¼ã‚Šä»˜ã‘ã¾ã™:')}
            </p>

            <div style={{ display: 'grid', gap: '1.25rem', marginBottom: '0.5rem' }}>
              <GlassField
                label={tr('1. Channel ID', '1. Channel ID')}
                helper={tr('Open "Basic settings", find "Channel ID", and copy it.', 'ã€ŒBasic settingsã€ã§ã€ŒChannel IDã€ã‚’è¦‹ã¤ã‘ã¦ã‚³ãƒ”ãƒ¼ã—ã¾ã™ã€‚')}
              >
                <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} placeholder={tr('Paste Channel ID', 'Channel IDã‚’è²¼ã‚Šä»˜ã‘')} />
              </GlassField>
              <GlassField
                label={tr('2. Channel Secret', '2. Channel Secret')}
                helper={tr('In "Basic settings", scroll to "Channel secret" and copy it.', 'ã€ŒBasic settingsã€ã®ã€ŒChannel secretã€ã‚’ã‚³ãƒ”ãƒ¼ã—ã¾ã™ã€‚')}
              >
                <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)} placeholder={tr('Paste Channel Secret', 'Channel Secretã‚’è²¼ã‚Šä»˜ã‘')} />
              </GlassField>
              <GlassField
                label={tr('3. Access Token', '3. Access Token')}
                helper={tr('In "Messaging API", find "Channel access token (long-lived)". Issue it if needed, then copy.', 'ã€ŒMessaging APIã€ã®ã€ŒChannel access token (long-lived)ã€ã‚’ã‚³ãƒ”ãƒ¼ã—ã¾ã™ã€‚ç©ºãªã‚‰å…ˆã«Issueã—ã¦ãã ã•ã„ã€‚')}
              >
                <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)} placeholder={tr('Paste Access Token', 'Access Tokenã‚’è²¼ã‚Šä»˜ã‘')} />
              </GlassField>
            </div>

            <p style={{ margin: '1rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {tr('These values are saved when you click Next so webhook verification works in the next step.', 'ã€Œæ¬¡ã¸ã€ã‚’æŠ¼ã—ãŸæ™‚ç‚¹ã§ä¿å­˜ã•ã‚Œã€æ¬¡ã®Webhookæ¤œè¨¼ã«ä½¿ç”¨ã•ã‚Œã¾ã™ã€‚')}
            </p>
          </div>
        )}

        {/* â”€â”€ Step 4: Set Webhook URL (channel must exist for LINE verify) â”€ */}
        {currentStep === 4 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>4</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Set your bot\'s address (Webhook URL)', 'ãƒœãƒƒãƒˆã®Webhook URLã‚’è¨­å®š')}</h3>
            </div>

            <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              {tr('In', 'æ¬¡ã®å ´æ‰€ã§')}{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                developers.line.biz/console <ExternalLink size={13} />
              </a>
              {' '}{tr('-> your channel -> "Messaging API" tab:', '-> å¯¾è±¡ãƒãƒ£ãƒãƒ« -> ã€ŒMessaging APIã€ã‚¿ãƒ–:')}
            </p>

            {renderLineAccountCard(
              tr('Connected LINE account', 'æŽ¥ç¶šä¸­ã®LINEã‚¢ã‚«ã‚¦ãƒ³ãƒˆ'),
              tr('Confirm this is the Official Account you want to finish setup for.', 'è¨­å®šã‚’å®Œäº†ã™ã‚‹å¯¾è±¡ã®LINEå…¬å¼ã‚¢ã‚«ã‚¦ãƒ³ãƒˆã‹ç¢ºèªã—ã¦ãã ã•ã„ã€‚'),
              {
                compact: true,
                background: 'rgba(6,199,85,0.08)',
                border: '1px solid rgba(6,199,85,0.22)',
                textColor: 'var(--text-primary)',
                mutedColor: 'var(--text-secondary)',
              }
            )}

            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              {tr('1. Copy this address:', '1. ã“ã®URLã‚’ã‚³ãƒ”ãƒ¼:')}
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <code style={{
                flex: 1, padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)', borderRadius: '10px',
                fontSize: '0.88rem', wordBreak: 'break-all',
                border: '1.5px solid var(--ui-flow-border)',
              }}>
                {webhookUrl}
              </code>
              <UiButton variant={copied ? 'primary' : 'secondary'} onClick={copyWebhookUrl} style={{ padding: '0.85rem 1.1rem', flexShrink: 0 }}>
                {copied ? <><Check size={16} /> {tr('Copied!', 'ã‚³ãƒ”ãƒ¼æ¸ˆã¿')}</> : <><Copy size={16} /> {tr('Copy', 'ã‚³ãƒ”ãƒ¼')}</>}
              </UiButton>
            </div>

            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              {tr('2. Paste into Webhook URL, click Update, turn Use webhook ON, then Verify.', '2. Webhook URLã«è²¼ã‚Šä»˜ã‘ã¦Updateã—ã€Use webhookã‚’ONã«ã—ã¦Verifyã—ã¾ã™ã€‚')}
            </p>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
              marginTop: '1rem',
            }}>
              <input type="checkbox" checked={webhookSet} onChange={(e) => setWebhookSet(e.target.checked)} />
              {tr('Webhook set and Verify passed', 'Webhookè¨­å®šã¨Verifyå®Œäº†')}
            </label>
          </div>
        )}

        {/* â”€â”€ Step 5: Connect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {currentStep === 5 && (
          <div style={{ textAlign: 'center', padding: '2rem 1rem' }}>
            <div style={{
              width: '64px', height: '64px', borderRadius: '50%',
              background: LINE_GRADIENT,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 1.5rem',
              boxShadow: '0 8px 32px rgba(6, 199, 85, 0.3)',
            }}>
              <Check size={32} color="#fff" />
            </div>
            <h3 style={{ fontSize: '1.3rem', fontWeight: 700, margin: '0 0 0.5rem 0' }}>
              {tr('Almost done! One last click...', 'ã‚‚ã†å°‘ã—ã§å®Œäº†ã€‚æœ€å¾Œã«1ã‚¯ãƒªãƒƒã‚¯ã§ã™ã€‚')}
            </h3>
            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              {tr('Your LINE account:', 'LINEã‚¢ã‚«ã‚¦ãƒ³ãƒˆ:')} <code style={{ fontFamily: 'monospace', fontWeight: 600 }}>{lineChannelId}</code>
            </p>
            {renderLineAccountCard(
              tr('Connected LINE account', 'æŽ¥ç¶šä¸­ã®LINEã‚¢ã‚«ã‚¦ãƒ³ãƒˆ'),
              tr('This is the Official Account that will start receiving messages after activation.', 'æœ‰åŠ¹åŒ–å¾Œã€ã“ã®LINEå…¬å¼ã‚¢ã‚«ã‚¦ãƒ³ãƒˆã§ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸å—ä¿¡ãŒå§‹ã¾ã‚Šã¾ã™ã€‚'),
              {
                compact: true,
                background: 'rgba(6,199,85,0.08)',
                border: '1px solid rgba(6,199,85,0.22)',
                textColor: 'var(--text-primary)',
                mutedColor: 'var(--text-secondary)',
              }
            )}
            <p style={{ margin: '0 0 2rem 0', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              {tr('Click the button below to activate your AI bot. After this, your bot will start replying to LINE messages automatically!', 'ä¸‹ã®ãƒœã‚¿ãƒ³ã‚’æŠ¼ã™ã¨AIãƒœãƒƒãƒˆãŒæœ‰åŠ¹åŒ–ã•ã‚Œã€LINEãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ã¸è‡ªå‹•è¿”ä¿¡ã‚’é–‹å§‹ã—ã¾ã™ã€‚')}
            </p>

            <button
              onClick={handleSave}
              disabled={saving}
              style={{
                background: LINE_GRADIENT,
                border: 'none', borderRadius: '14px',
                padding: '1rem 2.5rem', color: '#fff',
                fontWeight: 700, fontSize: '1.1rem',
                cursor: saving ? 'not-allowed' : 'pointer',
                display: 'inline-flex', alignItems: 'center', gap: '0.75rem',
                boxShadow: '0 8px 32px rgba(6, 199, 85, 0.35)',
                transition: 'all 0.25s',
                opacity: saving ? 0.75 : 1,
              }}
              onMouseEnter={(e) => { if (!saving) { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 12px 40px rgba(6, 199, 85, 0.5)' } }}
              onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 8px 32px rgba(6, 199, 85, 0.35)' }}
            >
              {saving ? (
                <><Loader2 size={22} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Connecting...', 'æŽ¥ç¶šä¸­...')}</>
              ) : (
                <><MessageCircle size={22} /> {tr('Activate Agent', 'ã‚¨ãƒ¼ã‚¸ã‚§ãƒ³ãƒˆã‚’æœ‰åŠ¹åŒ–')}</>
              )}
            </button>
          </div>
        )}

        {/* â”€â”€ Navigation Buttons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {currentStep > 0 && currentStep < 5 && (
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '1rem 1rem 0.5rem', borderTop: '1px solid var(--ui-flow-border)',
            marginTop: '1.5rem',
          }}>
            <button
              onClick={() => setCurrentStep(currentStep - 1)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                color: 'var(--text-secondary)', fontSize: '0.95rem', fontWeight: 500,
                padding: '0.5rem 0.75rem', borderRadius: '8px',
                transition: 'color 0.2s',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-secondary)' }}
            >
              <ChevronLeft size={18} /> {tr('Back', 'æˆ»ã‚‹')}
            </button>

            <button
              onClick={handleNext}
              disabled={!canAdvance() || saving}
              style={{
                background: canAdvance() && !saving ? LINE_GRADIENT : 'var(--ui-flow-border)',
                border: 'none', borderRadius: '10px',
                padding: '0.65rem 1.5rem', color: '#fff',
                fontWeight: 600, fontSize: '0.95rem',
                cursor: canAdvance() && !saving ? 'pointer' : 'not-allowed',
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                transition: 'all 0.2s',
                opacity: canAdvance() && !saving ? 1 : 0.5,
              }}
            >
              {currentStep === 3 && saving ? (
                <><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Saving...', 'ä¿å­˜ä¸­...')}</>
              ) : (
                <>{tr('Next', 'æ¬¡ã¸')} <ChevronRight size={18} /></>
              )}
            </button>
          </div>
        )}

        {/* Back button on step 5 */}
        {currentStep === 5 && (
          <div style={{
            display: 'flex', justifyContent: 'center',
            padding: '1rem 1rem 0.5rem', borderTop: '1px solid var(--ui-flow-border)',
            marginTop: '1.5rem',
          }}>
            <button
              onClick={() => setCurrentStep(4)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                color: 'var(--text-secondary)', fontSize: '0.95rem', fontWeight: 500,
                padding: '0.5rem 0.75rem', borderRadius: '8px',
              }}
            >
              <ChevronLeft size={18} /> {tr('Back', 'æˆ»ã‚‹')}
            </button>
          </div>
        )}
      </GlassCard>
    </AnimatedPage>
  )
}

