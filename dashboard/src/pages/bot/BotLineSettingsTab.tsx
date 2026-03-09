import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData, type LineChannelTestResult } from '../../hooks/useDashboardData'
import { useTranslation } from 'react-i18next'
import {
  Check, CheckCircle, Copy, ExternalLink, AlertCircle, Loader2,
  Trash2, Zap, MessageCircle, ChevronLeft, ChevronRight, RefreshCw,
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

/* ─── Progress Bar (for the 5 card steps only, excludes Get Started) ─── */
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

export default function BotLineSettingsTab() {
  const { botId } = useParams()
  const { selectedBot } = useDashboardData()
  const dialog = useDialog()
  const { getAccessTokenSilently } = useAuth0()
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)
  const cardStepLabels = [
    tr('Enable API', 'APIを有効化'),
    tr('Auto-reply', '自動返信'),
    tr('Credentials', '認証情報'),
    tr('Webhook', 'Webhook'),
    tr('Connect', '接続'),
  ]
  const getProgressText = (step: number, totalSteps: number, currentLabel: string) =>
    isJa ? `ステップ ${step}/${totalSteps} - ${currentLabel}` : `Step ${step} of ${totalSteps} - ${currentLabel}`

  const [lineChannelId, setLineChannelId] = useState('')
  const [lineChannelSecret, setLineChannelSecret] = useState('')
  const [lineAccessToken, setLineAccessToken] = useState('')
  const [isActive, setIsActive] = useState(true)
  const [existing, setExisting] = useState<LineChannelConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [resyncing, setResyncing] = useState(false)
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
      if (!body.line_channel_id) throw new Error(tr('Channel ID is required', 'チャネルIDは必須です'))
      if (!existing && (!body.line_channel_secret || !body.line_channel_access_token)) {
        throw new Error(tr('Channel Secret and Access Token are required for initial setup', '初期設定にはChannel SecretとAccess Tokenが必要です'))
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
      setSuccess(tr('Connected! Your bot is live on LINE.', '接続完了。ボットはLINEで稼働中です。'))
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
      title: tr('Disconnect LINE integration? Your bot will stop responding on LINE.', 'LINE連携を解除しますか？ボットはLINEで返信しなくなります。'),
      confirmLabel: tr('Disconnect', '連携解除'),
      cancelLabel: tr('Cancel', 'キャンセル'),
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
      setSuccess(tr('LINE integration disconnected.', 'LINE連携を解除しました。'))
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

  if (!selectedBot || !botId) {
    return <div className="empty-panel">{tr('Select a bot to configure LINE integration.', 'LINE連携を設定するボットを選択してください。')}</div>
  }

  async function handleResyncMenu() {
    if (!botId) return
    setResyncing(true)
    setError(null)
    setSuccess(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel/rich-menu/resync`, { method: 'POST' })
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as LineChannelConfig
      setExisting(data)
      if (data.rich_menu_sync_status === 'error') {
        setError(data.rich_menu_last_error || tr('LINE menu sync failed.', 'LINEメニューの同期に失敗しました。'))
      } else {
        setSuccess(tr('Managed LINE menu resynced.', 'LINEリッチメニューを再同期しました。'))
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setResyncing(false)
    }
  }

  if (loading) {
    return (
      <AnimatedPage className="page-body">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', gap: '0.75rem', color: 'var(--text-secondary)' }}>
          <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
          {tr('Loading LINE settings...', 'LINE設定を読み込み中...')}
        </div>
      </AnimatedPage>
    )
  }

  /* ═══════════════════════════════════════════════════════════════
     Connected View
     ═══════════════════════════════════════════════════════════════ */
  const richMenuStatus = existing?.rich_menu_sync_status || 'pending'
  const richMenuStatusLabel = (() => {
    switch (richMenuStatus) {
      case 'synced':
        return tr('Synced', '同期済み')
      case 'syncing':
        return tr('Syncing', '同期中')
      case 'inactive':
        return tr('Inactive', '停止中')
      case 'no_actions':
        return tr('No actions', '項目なし')
      case 'error':
        return tr('Error', 'エラー')
      default:
        return tr('Pending', '保留中')
    }
  })()
  const richMenuStatusColor = richMenuStatus === 'synced'
    ? '#27ae60'
    : richMenuStatus === 'error'
      ? '#e74c3c'
      : 'var(--text-secondary)'
  const richMenuVariantCount = Object.keys(existing?.rich_menu_variants || {}).length
  const lineAccountName = testResult?.display_name || testResult?.basic_id || existing?.line_channel_id || lineChannelId.trim()
  const lineAccountPictureUrl = testResult?.picture_url || null

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
            <span style={{ fontWeight: 600, wordBreak: 'break-all', textAlign: 'right' }}>{testResult?.basic_id || tr('Not available', '未取得')}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
            <span style={{ color: mutedColor }}>{tr('Channel ID', 'チャネルID')}</span>
            <span style={{ fontWeight: 600, wordBreak: 'break-all', textAlign: 'right' }}>{existing?.line_channel_id || lineChannelId.trim() || tr('Not available', '未取得')}</span>
          </div>
        </div>
      </div>
    )
  }

  if (existing) {
    return (
      <AnimatedPage className="page-body">
        <SectionHeader
          eyebrow={tr('Integrations', '連携')}
          title={tr('LINE channel', 'LINEチャンネル')}
          subtitle={tr('Your bot is live and responding to messages on LINE.', 'ボットはLINEメッセージに自動返信中です。')}
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
                  {tr('Connected & Active', '接続済み・有効')}
                </div>
                <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: '0.95rem', fontWeight: 500 }}>
                  {lineAccountName}
                  {testResult?.basic_id ? (
                    <>
                      {' '}·{' '}
                      <span>{testResult.basic_id}</span>
                    </>
                  ) : null}
                  {' '}&bull;{' '}{existing.is_active ? tr('Active', '有効') : tr('Paused', '一時停止')}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button
                onClick={handleResyncMenu}
                disabled={resyncing}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  backdropFilter: 'blur(10px)',
                  border: '2px solid rgba(255,255,255,0.3)',
                  borderRadius: '12px',
                  padding: '0.75rem 1.5rem',
                  color: '#fff', fontWeight: 600, fontSize: '0.95rem',
                  cursor: resyncing ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', gap: '0.6rem',
                  transition: 'all 0.2s',
                  opacity: resyncing ? 0.7 : 1,
                }}
                onMouseEnter={(e) => { if (!resyncing) { e.currentTarget.style.background = 'rgba(255,255,255,0.3)'; e.currentTarget.style.transform = 'translateY(-2px)' } }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.2)'; e.currentTarget.style.transform = 'translateY(0)' }}
              >
                {resyncing ? (<><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Resyncing...', '再同期中...')}</>) : (<><RefreshCw size={18} /> {tr('Resync LINE menu', 'LINEメニューを再同期')}</>)}
              </button>
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
                {testing ? (<><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Testing...', 'テスト中...')}</>) : (<><Zap size={18} /> {tr('Test Connection', '接続テスト')}</>)}
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
            <div className="card-title" style={{ marginBottom: '1rem' }}>{tr('Connection Details', '接続情報')}</div>
            {renderLineAccountCard(
              tr('Connected LINE account', '接続中のLINEアカウント'),
              tr('This is the Official Account currently connected to your bot.', '現在このボットに接続されているLINE公式アカウントです。'),
              {
                compact: true,
                background: 'var(--ui-flow-surface)',
                border: '1px solid var(--ui-flow-border)',
                textColor: 'var(--text-primary)',
                mutedColor: 'var(--text-secondary)',
              }
            )}
            <div style={{ display: 'grid', gap: '0.75rem', fontSize: '0.95rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Channel ID', 'チャネルID')}</span>
                <code style={{ fontSize: '0.85rem', fontFamily: 'monospace' }}>{existing.line_channel_id}</code>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Account name', 'アカウント名')}</span>
                <span style={{ fontWeight: 600 }}>{lineAccountName || tr('Not available', '未取得')}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Basic ID', 'Basic ID')}</span>
                <span style={{ fontWeight: 500 }}>{testResult?.basic_id || tr('Not available', '未取得')}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Status', 'ステータス')}</span>
                <span style={{ fontWeight: 600, color: existing.is_active ? '#27ae60' : '#e74c3c' }}>
                  {existing.is_active ? tr('Active', '有効') : tr('Paused', '一時停止')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Connected', '接続日')}</span>
                <span style={{ fontWeight: 500 }}>
                  {new Date(existing.created_at).toLocaleDateString(isJa ? 'ja-JP' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Managed menu', '管理メニュー')}</span>
                <span style={{ fontWeight: 600, color: richMenuStatusColor }}>{richMenuStatusLabel}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Menu variants', 'メニュー数')}</span>
                <span style={{ fontWeight: 500 }}>{richMenuVariantCount}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Last synced', '最終同期')}</span>
                <span style={{ fontWeight: 500 }}>
                  {existing.rich_menu_last_synced_at
                    ? new Date(existing.rich_menu_last_synced_at).toLocaleString(isJa ? 'ja-JP' : 'en-US')
                    : tr('Not yet', '未実行')}
                </span>
              </div>
            </div>
            {existing.rich_menu_last_error ? (
              <div style={{
                marginTop: '1rem',
                padding: '0.9rem 1rem',
                borderRadius: '14px',
                background: 'rgba(231, 76, 60, 0.08)',
                border: '1px solid rgba(231, 76, 60, 0.18)',
                color: '#b42318',
                fontSize: '0.9rem',
                lineHeight: 1.5,
              }}>
                <strong>{tr('Rich menu error', 'リッチメニューエラー')}</strong>
                <div>{existing.rich_menu_last_error}</div>
              </div>
            ) : null}
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
            <div className="card-title" style={{ marginBottom: '1rem' }}>{tr('Manage Connection', '接続管理')}</div>

            {/* Update credentials (collapsed by default) */}
            <details style={{ marginBottom: '1rem' }}>
              <summary style={{ cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '1rem' }}>
                {tr('Update credentials', '認証情報を更新')}
              </summary>
              <div style={{ display: 'grid', gap: '1rem', paddingTop: '0.5rem' }}>
                <GlassField label={tr('Channel ID', 'チャネルID')}>
                  <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} />
                </GlassField>
                <GlassField label={tr('Channel Secret', 'チャネルシークレット')}>
                  <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)} placeholder={tr('Leave blank to keep current', '空欄で現在の値を保持')} />
                </GlassField>
                <GlassField label={tr('Channel Access Token', 'チャネルアクセストークン')}>
                  <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)} placeholder={tr('Leave blank to keep current', '空欄で現在の値を保持')} />
                </GlassField>
                <UiButton variant="primary" onClick={handleSave} disabled={saving}>
                  {saving ? tr('Saving...', '保存中...') : tr('Save Changes', '変更を保存')}
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
                {deleting ? tr('Removing...', '解除中...') : tr('Disconnect', '連携解除')}
              </UiButton>
            </div>
          </GlassCard>
        </div>
      </AnimatedPage>
    )
  }

  /* ═══════════════════════════════════════════════════════════════
     Setup Wizard (not connected)
     ═══════════════════════════════════════════════════════════════ */
  return (
    <AnimatedPage className="page-body">
      <SectionHeader
        eyebrow={tr('Integrations', '連携')}
        title={tr('Connect LINE', 'LINEに接続')}
        subtitle={tr('Follow the guided steps below to connect your LINE account.', '以下のガイド手順でLINEアカウントを接続してください。')}
      />

      {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
      {success && <div style={{ marginBottom: '1.5rem', color: LINE_GREEN, fontWeight: 600 }}>{success}</div>}

      {currentStep > 0 && <StepProgress current={currentStep - 1} total={5} labels={cardStepLabels} getProgressText={getProgressText} />}

      <GlassCard>
        {/* ── Step 0: Get Started ──────────────────────────────── */}
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
                {tr('Connect your LINE account', 'LINEアカウントを接続')}
              </h3>
              <p style={{
                margin: 0, color: 'var(--text-secondary)',
                fontSize: '1rem', maxWidth: '480px', marginLeft: 'auto', marginRight: 'auto', lineHeight: 1.6,
              }}>
                {tr(
                  'We\'ll walk you through very simple steps to connect your LINE business account, so your AI Agent can reply to messages automatically.',
                  'LINEビジネスアカウントを接続する手順をわかりやすく案内します。接続後はAIエージェントが自動で返信します。',
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
                <AlertCircle size={16} color={LINE_GREEN} /> {tr('What you need before starting', '開始前に必要なもの')}
              </div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {isJa ? (
                  <>LINE Official Account（個人用LINEアプリとは別のビジネスアカウント）</>
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
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.75rem' }}>{tr('Here\'s what we\'ll do in 5 simple steps:', '5つの簡単な手順で進めます:')}</div>
              <div style={{ display: 'grid', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                {[
                  ['1', tr('Enable Messaging API (manager.line.biz)', 'Messaging APIを有効化（manager.line.biz）')],
                  ['2', tr('Turn off Auto-reply (manager.line.biz)', '自動返信をOFFにする（manager.line.biz）')],
                  ['3', tr('Copy 3 codes (Developers Console)', '3つのコードをコピー（Developers Console）')],
                  ['4', tr('Set your bot\'s address - webhook (LINE will verify)', 'ボットのWebhook URLを設定（LINE側で検証）')],
                  ['5', tr('Click "Activate" and you\'re done!', '「有効化」を押して完了')],
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
                {tr('I have a LINE Official Account - Let\'s start', 'LINE公式アカウントがあります。開始する')}
                <ChevronRight size={20} />
              </button>
              <div style={{ marginTop: '0.75rem', fontSize: '0.83rem', color: 'var(--text-secondary)' }}>
                {tr('Don\'t have one yet?', 'まだ持っていませんか？')}{' '}
                <a href="https://www.linebiz.com/jp/entry/" target="_blank" rel="noopener noreferrer"
                  style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none' }}>
                  {tr('Create it for free first', '無料で作成')} <ExternalLink size={11} style={{ display: 'inline', verticalAlign: 'middle' }} />
                </a>
              </div>
            </div>
          </div>
        )}

        {/* ── Step 1: Enable Messaging API ─────────────────────── */}
        {currentStep === 1 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>1</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Enable Messaging API', 'Messaging APIを有効化')}</h3>
            </div>

            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'rgba(6,199,85,0.07)',
              borderRadius: '10px', border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.5rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6,
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px', color: LINE_GREEN }} />
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>{tr('Important:', '重要:')}</strong> {tr('Use a computer browser - the Messaging API option is not available in the LINE mobile app.', 'PCブラウザを使用してください。LINEモバイルアプリではMessaging API設定が利用できません。')}
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
                  {' '}{tr('(the Official Account manager) and sign in', '（公式アカウントマネージャー）にログイン')}
                </li>
                <li>{tr('Click your business account name', 'ビジネスアカウント名をクリック')}</li>
                <li>{tr('Click Settings in the top-right corner', '右上の設定をクリック')}</li>
                <li>{tr('In the left menu, click "Messaging API"', '左メニューで「Messaging API」をクリック')}</li>
                <li>{tr('Click the green "Enable Messaging API" button', '緑色の「Enable Messaging API」をクリック')}</li>
                <li>{tr('Enter a Provider name (company/brand) and click OK', 'Provider名（会社/ブランド名）を入力してOK')}</li>
              </ol>

              <div style={{
                margin: '1.25rem 0 0 0',
                padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)',
                borderRadius: '10px',
                border: '1px solid var(--ui-flow-border)',
                fontSize: '0.88rem',
              }}>
                ✅ <strong>{tr('Done when:', '完了条件:')}</strong> {tr('You see a page with Channel ID and Channel Secret.', 'Channel IDとChannel Secretが表示されたら完了です。')}
              </div>
            </div>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
            }}>
              <input type="checkbox" checked={apiEnabled} onChange={(e) => setApiEnabled(e.target.checked)} />
              {tr('Messaging API enabled - I can see Channel ID and Channel Secret', 'Messaging APIを有効化し、Channel ID/Channel Secretを確認しました')}
            </label>
          </div>
        )}

        {/* ── Step 2: Turn off Auto-reply ────────────────────────── */}
        {currentStep === 2 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>2</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Turn off Auto-reply messages', '自動返信メッセージをOFFにする')}</h3>
            </div>

            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'rgba(6,199,85,0.07)',
              borderRadius: '10px', border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.5rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6,
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px', color: LINE_GREEN }} />
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>{tr('Why?', '理由:')}</strong>{' '}
                {tr('LINE sends a default "Thanks for your message!" when someone messages you. Turn it off so only your bot replies - otherwise customers get two replies.', 'LINEの初期設定ではメッセージ受信時に自動返信されます。これをOFFにして、ボットのみが返信するようにしてください。')}
              </span>
            </div>

            <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              {tr('Still in', '引き続き')}{' '}
              <a href="https://manager.line.biz/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                manager.line.biz <ExternalLink size={13} />
              </a>
              :
            </p>
            <ol style={{ margin: 0, paddingLeft: '1.4rem', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 2 }}>
              <li>{tr('Click Settings -> "Response settings" in the left menu', '左メニューの「Response settings」を開く')}</li>
              <li>{tr('Find "Auto-response messages" and turn it OFF', '「Auto-response messages」をOFFにする')}</li>
            </ol>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
              marginTop: '1rem',
            }}>
              <input type="checkbox" checked={autoReplyOff} onChange={(e) => setAutoReplyOff(e.target.checked)} />
              {tr('Auto-response messages is OFF', 'Auto-response messagesをOFFにしました')}
            </label>
          </div>
        )}

        {/* ── Step 3: Copy Credentials (must save before webhook verify) ─ */}
        {currentStep === 3 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>3</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Copy the 3 codes', '3つのコードをコピー')}</h3>
            </div>

            <p style={{ margin: '0 0 1rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              {tr('Go to', '次へアクセス')}{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                developers.line.biz/console <ExternalLink size={13} />
              </a>
              {' '}{tr('and do the following:', 'して、以下を実施してください:')}
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
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>{tr('A. Select or create a Provider', 'A. Providerを選択または作成')}</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {tr('In the left panel, you will see a list of Providers. Select an existing one, or click "Create" to make a new one.', '左側の一覧からProviderを選択するか、「Create」で新規作成します。')}
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
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>{tr('B. Select your Messaging API channel', 'B. Messaging APIチャネルを選択')}</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                {tr('Under your Provider, open the Messaging API channel you created in Step 1.', 'Step1で作成したMessaging APIチャネルを開いてください。')}
              </div>
            </div>

            <p style={{ margin: '0 0 1rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              {tr('C. Copy these 3 values from the channel page and paste them below:', 'C. チャネル画面の3項目をコピーして以下に貼り付けます:')}
            </p>

            <div style={{ display: 'grid', gap: '1.25rem', marginBottom: '0.5rem' }}>
              <GlassField
                label={tr('1. Channel ID', '1. Channel ID')}
                helper={tr('Open "Basic settings", find "Channel ID", and copy it.', '「Basic settings」で「Channel ID」を見つけてコピーします。')}
              >
                <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} placeholder={tr('Paste Channel ID', 'Channel IDを貼り付け')} />
              </GlassField>
              <GlassField
                label={tr('2. Channel Secret', '2. Channel Secret')}
                helper={tr('In "Basic settings", scroll to "Channel secret" and copy it.', '「Basic settings」の「Channel secret」をコピーします。')}
              >
                <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)} placeholder={tr('Paste Channel Secret', 'Channel Secretを貼り付け')} />
              </GlassField>
              <GlassField
                label={tr('3. Access Token', '3. Access Token')}
                helper={tr('In "Messaging API", find "Channel access token (long-lived)". Issue it if needed, then copy.', '「Messaging API」の「Channel access token (long-lived)」をコピーします。空なら先にIssueしてください。')}
              >
                <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)} placeholder={tr('Paste Access Token', 'Access Tokenを貼り付け')} />
              </GlassField>
            </div>

            <p style={{ margin: '1rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {tr('These values are saved when you click Next so webhook verification works in the next step.', '「次へ」を押した時点で保存され、次のWebhook検証に使用されます。')}
            </p>
          </div>
        )}

        {/* ── Step 4: Set Webhook URL (channel must exist for LINE verify) ─ */}
        {currentStep === 4 && (
          <div style={{ padding: '1.5rem 1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: '42px', height: '42px', borderRadius: '12px',
                background: LINE_GRADIENT,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.1rem', fontWeight: 700, color: '#fff', flexShrink: 0,
              }}>4</div>
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>{tr('Set your bot\'s address (Webhook URL)', 'ボットのWebhook URLを設定')}</h3>
            </div>

            <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              {tr('In', '次の場所で')}{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                developers.line.biz/console <ExternalLink size={13} />
              </a>
              {' '}{tr('-> your channel -> "Messaging API" tab:', '-> 対象チャネル -> 「Messaging API」タブ:')}
            </p>

            {renderLineAccountCard(
              tr('Connected LINE account', '接続中のLINEアカウント'),
              tr('Confirm this is the Official Account you want to finish setup for.', '設定を完了する対象のLINE公式アカウントか確認してください。'),
              {
                compact: true,
                background: 'rgba(6,199,85,0.08)',
                border: '1px solid rgba(6,199,85,0.22)',
                textColor: 'var(--text-primary)',
                mutedColor: 'var(--text-secondary)',
              }
            )}

            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              {tr('1. Copy this address:', '1. このURLをコピー:')}
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
                {copied ? <><Check size={16} /> {tr('Copied!', 'コピー済み')}</> : <><Copy size={16} /> {tr('Copy', 'コピー')}</>}
              </UiButton>
            </div>

            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              {tr('2. Paste into Webhook URL, click Update, turn Use webhook ON, then Verify.', '2. Webhook URLに貼り付けてUpdateし、Use webhookをONにしてVerifyします。')}
            </p>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
              marginTop: '1rem',
            }}>
              <input type="checkbox" checked={webhookSet} onChange={(e) => setWebhookSet(e.target.checked)} />
              {tr('Webhook set and Verify passed', 'Webhook設定とVerify完了')}
            </label>
          </div>
        )}

        {/* ── Step 5: Connect ──────────────────────────────────── */}
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
              {tr('Almost done! One last click...', 'もう少しで完了。最後に1クリックです。')}
            </h3>
            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              {tr('Your LINE account:', 'LINEアカウント:')} <code style={{ fontFamily: 'monospace', fontWeight: 600 }}>{lineChannelId}</code>
            </p>
            {renderLineAccountCard(
              tr('Connected LINE account', '接続中のLINEアカウント'),
              tr('This is the Official Account that will start receiving messages after activation.', '有効化後、このLINE公式アカウントでメッセージ受信が始まります。'),
              {
                compact: true,
                background: 'rgba(6,199,85,0.08)',
                border: '1px solid rgba(6,199,85,0.22)',
                textColor: 'var(--text-primary)',
                mutedColor: 'var(--text-secondary)',
              }
            )}
            <p style={{ margin: '0 0 2rem 0', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              {tr('Click the button below to activate your AI bot. After this, your bot will start replying to LINE messages automatically!', '下のボタンを押すとAIボットが有効化され、LINEメッセージへ自動返信を開始します。')}
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
                <><Loader2 size={22} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Connecting...', '接続中...')}</>
              ) : (
                <><MessageCircle size={22} /> {tr('Activate Agent', 'エージェントを有効化')}</>
              )}
            </button>
          </div>
        )}

        {/* ── Navigation Buttons ───────────────────────────────── */}
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
              <ChevronLeft size={18} /> {tr('Back', '戻る')}
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
                <><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> {tr('Saving...', '保存中...')}</>
              ) : (
                <>{tr('Next', '次へ')} <ChevronRight size={18} /></>
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
              <ChevronLeft size={18} /> {tr('Back', '戻る')}
            </button>
          </div>
        )}
      </GlassCard>
    </AnimatedPage>
  )
}
