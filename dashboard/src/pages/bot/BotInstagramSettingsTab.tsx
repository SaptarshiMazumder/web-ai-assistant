import { useCallback, useEffect, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { CheckCircle, AlertCircle, Loader2, Trash2, Zap, Instagram, ExternalLink, LogIn } from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton, GlassCard } from '../../components/ui'
import { useTranslation } from 'react-i18next'

type InstagramChannelConfig = {
  channel_id: string
  bot_id: string
  org_id: string
  ig_page_id: string
  verify_token: string
  is_active: boolean
  created_at: string
  updated_at: string
  ig_user_id?: string
  ig_username?: string
  token_expires_at?: string
  connection_method?: string
}

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

export default function BotInstagramSettingsTab() {
  const { botId } = useParams()
  const { selectedBot } = useDashboardData()
  const { getAccessTokenSilently } = useAuth0()
  const [searchParams, setSearchParams] = useSearchParams()
  const { i18n } = useTranslation()
  const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
  const isJa = lang.startsWith('ja') || lang.startsWith('jp')
  const tr = (en: string, ja: string) => (isJa ? ja : en)

  const [existing, setExisting] = useState<InstagramChannelConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [connecting, setConnecting] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [disconnecting, setDisconnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Check URL params for OAuth callback results
  useEffect(() => {
    const connected = searchParams.get('connected')
    const username = searchParams.get('username')
    const igError = searchParams.get('ig_error')

    if (connected === 'true') {
      setSuccess(
        isJa
          ? `接続完了${username ? ` @${username}` : ''}。ボットはInstagramで稼働中です。`
          : `Connected${username ? ` @${username}` : ''}! Your bot is live on Instagram.`,
      )
      // Clean URL params
      searchParams.delete('connected')
      searchParams.delete('username')
      setSearchParams(searchParams, { replace: true })
    }
    if (igError) {
      setError(`${tr('Connection failed', '接続に失敗しました')}: ${decodeURIComponent(igError)}`)
      searchParams.delete('ig_error')
      setSearchParams(searchParams, { replace: true })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

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

  const loadConfig = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/instagram-channel`)
      if (resp.status === 404) {
        setExisting(null)
        return
      }
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as InstagramChannelConfig
      setExisting(data)
    } catch (err) {
      if ((err as Error).message?.includes('404') || (err as Error).message?.includes('Not Found')) {
        setExisting(null)
      } else {
        setError((err as Error).message)
      }
    } finally {
      setLoading(false)
    }
  }, [botId, authedFetch])

  useEffect(() => {
    if (!botId) return
    setError(null)
    void loadConfig()
  }, [botId, loadConfig])

  async function handleConnect() {
    if (!botId) return
    setConnecting(true)
    setError(null)
    setSuccess(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/instagram/auth-url`)
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      const { auth_url } = (await resp.json()) as { auth_url: string }
      // Redirect to Instagram OAuth
      window.location.href = auth_url
    } catch (err) {
      setError((err as Error).message)
      setConnecting(false)
    }
  }

  async function handleTestConnection() {
    if (!botId) return
    setTesting(true)
    setTestResult(null)
    setError(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/instagram-channel/test`, {
        method: 'POST',
      })
      const data = await resp.json() as { ok: boolean; message: string }
      setTestResult(data)
    } catch (err) {
      setTestResult({ ok: false, message: (err as Error).message })
    } finally {
      setTesting(false)
    }
  }

  async function handleDisconnect() {
    if (!botId) return
    const confirmed = window.confirm(tr('Disconnect Instagram integration? Your bot will stop responding to DMs.', 'Instagram連携を解除しますか？ボットはDMに返信しなくなります。'))
    if (!confirmed) return
    setDisconnecting(true)
    setError(null)
    setSuccess(null)
    try {
      // Try OAuth disconnect endpoint first, fall back to legacy
      let resp = await authedFetch(`/v1/org/bots/${botId}/instagram/disconnect`, { method: 'POST' })
      if (resp.status === 404) {
        resp = await authedFetch(`/v1/org/bots/${botId}/instagram-channel`, { method: 'DELETE' })
      }
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      setExisting(null)
      setTestResult(null)
      setSuccess(tr('Instagram integration disconnected.', 'Instagram連携を解除しました。'))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setDisconnecting(false)
    }
  }

  if (!selectedBot || !botId) {
    return <div className="empty-panel">{tr('Select a bot to configure Instagram integration.', 'Instagram連携を設定するボットを選択してください。')}</div>
  }

  if (loading) {
    return (
      <AnimatedPage className="page-body">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', gap: '0.75rem', color: 'var(--text-secondary)' }}>
          <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
          {tr('Loading Instagram settings...', 'Instagram設定を読み込み中...')}
        </div>
      </AnimatedPage>
    )
  }

  /* ─── Connected view ─────────────────────────────────────────── */
  if (existing) {
    const isOAuth = existing.connection_method === 'oauth'
    const displayName = existing.ig_username
      ? `@${existing.ig_username}`
      : existing.ig_page_id
    const tokenExpiry = existing.token_expires_at
      ? new Date(existing.token_expires_at).toLocaleDateString(isJa ? 'ja-JP' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      : null

    return (
      <AnimatedPage className="page-body">
        <SectionHeader
          eyebrow={tr('Integrations', '連携')}
          title={tr('Instagram channel', 'Instagramチャンネル')}
          subtitle={tr('Your bot is live and responding to DMs on Instagram.', 'ボットはInstagramのDMに自動返信中です。')}
        />

        {/* Status Hero */}
        <div style={{
          background: 'linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%)',
          borderRadius: '18px',
          padding: '2rem',
          marginBottom: '2rem',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 20px 60px rgba(131, 58, 180, 0.3)',
        }}>
          <div style={{
            position: 'absolute',
            top: '-50%',
            right: '-20%',
            width: '500px',
            height: '500px',
            background: 'radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 60%)',
            borderRadius: '50%',
            pointerEvents: 'none',
          }} />

          <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
              <div style={{
                width: '60px',
                height: '60px',
                borderRadius: '16px',
                background: 'rgba(255,255,255,0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
              }}>
                <Instagram size={30} color="#fff" />
              </div>
              <div>
                <div style={{
                  fontSize: '1.4rem',
                  fontWeight: 700,
                  color: '#fff',
                  marginBottom: '0.25rem',
                }}>
                  {tr('Connected & Active', '接続済み・有効')}
                </div>
                <div style={{
                  color: 'rgba(255,255,255,0.85)',
                  fontSize: '0.95rem',
                  fontWeight: 500,
                }}>
                  {displayName && (
                    <span style={{
                      background: 'rgba(0,0,0,0.2)',
                      padding: '2px 10px',
                      borderRadius: '6px',
                      fontFamily: existing.ig_username ? 'inherit' : 'monospace',
                      fontWeight: 600,
                    }}>{displayName}</span>
                  )}
                  {' '}&bull;{' '}
                  {existing.is_active ? tr('Active', '有効') : tr('Paused', '一時停止')}
                  {isOAuth && tokenExpiry && (
                    <> &bull; {tr('Token expires', 'トークン有効期限')} {tokenExpiry}</>
                  )}
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
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: '0.95rem',
                  cursor: testing ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  transition: 'all 0.2s',
                  opacity: testing ? 0.7 : 1,
                }}
                onMouseEnter={(e) => {
                  if (!testing) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.3)'
                    e.currentTarget.style.transform = 'translateY(-2px)'
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.2)'
                  e.currentTarget.style.transform = 'translateY(0)'
                }}
              >
                {testing ? (
                  <>
                    <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                    {tr('Testing...', 'テスト中...')}
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    {tr('Test Connection', '接続テスト')}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Test Result */}
        {testResult && (
          <div style={{
            marginBottom: '1.5rem',
            color: testResult.ok ? '#833ab4' : '#e74c3c',
            fontWeight: 600,
            fontSize: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            {testResult.ok ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
            {testResult.message}
          </div>
        )}

        {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
        {success && <div style={{ marginBottom: '1.5rem', color: '#833ab4', fontWeight: 600 }}>{success}</div>}

        {/* Info + Actions */}
        <div style={{ display: 'grid', gap: '1.5rem' }}>
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1rem' }}>
              {tr('Connection Details', '接続情報')}
            </div>
            <div style={{ display: 'grid', gap: '0.75rem', fontSize: '0.95rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Method', '方式')}</span>
                <span style={{ fontWeight: 600 }}>
                  {isOAuth ? (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: '#833ab4' }}>
                      <LogIn size={14} /> {tr('OAuth (automatic)', 'OAuth（自動）')}
                    </span>
                  ) : (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                      {tr('Manual credentials', '手動認証情報')}
                    </span>
                  )}
                </span>
              </div>
              {existing.ig_username && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{tr('Account', 'アカウント')}</span>
                  <span style={{ fontWeight: 600 }}>@{existing.ig_username}</span>
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Account ID', 'アカウントID')}</span>
                <code style={{ fontSize: '0.85rem', fontFamily: 'monospace' }}>{existing.ig_user_id || existing.ig_page_id}</code>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Status', 'ステータス')}</span>
                <span style={{ fontWeight: 600, color: existing.is_active ? '#27ae60' : '#e74c3c' }}>
                  {existing.is_active ? tr('Active', '有効') : tr('Paused', '一時停止')}
                </span>
              </div>
              {isOAuth && tokenExpiry && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{tr('Token expires', 'トークン有効期限')}</span>
                  <span style={{ fontWeight: 500 }}>{tokenExpiry}</span>
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{tr('Connected', '接続日')}</span>
                <span style={{ fontWeight: 500 }}>
                  {new Date(existing.created_at).toLocaleDateString(isJa ? 'ja-JP' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
              </div>
            </div>
          </GlassCard>

          {/* Reconnect / Disconnect */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1rem' }}>
              {tr('Manage Connection', '接続管理')}
            </div>
            {isOAuth && (
              <p style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                {tr('Token auto-refreshes. If you have issues, reconnect by clicking below.', 'トークンは自動更新されます。問題がある場合は下のボタンで再接続してください。')}
              </p>
            )}
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {isOAuth && (
                <UiButton variant="primary" onClick={handleConnect} disabled={connecting}>
                  {connecting ? tr('Redirecting...', 'リダイレクト中...') : tr('Reconnect', '再接続')}
                </UiButton>
              )}
              <UiButton
                variant="secondary"
                onClick={handleDisconnect}
                disabled={disconnecting}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#e74c3c', borderColor: '#e74c3c' }}
              >
                <Trash2 size={16} />
                {disconnecting ? tr('Removing...', '解除中...') : tr('Disconnect', '連携解除')}
              </UiButton>
            </div>
          </GlassCard>
        </div>
      </AnimatedPage>
    )
  }

  /* ─── Not connected - OAuth flow ─────────────────────────────── */
  return (
    <AnimatedPage className="page-body">
      <SectionHeader
        eyebrow={tr('Integrations', '連携')}
        title={tr('Connect Instagram', 'Instagramに接続')}
        subtitle={tr('Let your AI bot reply to Instagram DMs automatically.', 'AIボットがInstagramのDMに自動返信できるようにします。')}
      />

      {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
      {success && <div style={{ marginBottom: '1.5rem', color: '#833ab4', fontWeight: 600 }}>{success}</div>}

      <GlassCard>
        {/* Hero section */}
        <div style={{
          textAlign: 'center',
          padding: '2rem 1rem',
        }}>
          <div style={{
            width: '80px',
            height: '80px',
            borderRadius: '20px',
            background: 'linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.5rem',
            boxShadow: '0 12px 40px rgba(131, 58, 180, 0.3)',
          }}>
            <Instagram size={40} color="#fff" />
          </div>

          <h3 style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            margin: '0 0 0.5rem 0',
          }}>
            {tr('Connect your Instagram account', 'Instagramアカウントを接続')}
          </h3>
          <p style={{
            margin: '0 0 2rem 0',
            color: 'var(--text-secondary)',
            fontSize: '1rem',
            maxWidth: '420px',
            marginLeft: 'auto',
            marginRight: 'auto',
            lineHeight: 1.6,
          }}>
            {tr(
              'Sign in with Instagram, approve permissions, and your bot starts replying to DMs instantly. No developer console needed.',
              'Instagramでログインして権限を許可すると、ボットがDMにすぐ自動返信を開始します。開発者コンソールは不要です。',
            )}
          </p>

          <button
            onClick={handleConnect}
            disabled={connecting}
            style={{
              background: 'linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%)',
              border: 'none',
              borderRadius: '14px',
              padding: '1rem 2.5rem',
              color: '#fff',
              fontWeight: 700,
              fontSize: '1.1rem',
              cursor: connecting ? 'not-allowed' : 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
              boxShadow: '0 8px 32px rgba(131, 58, 180, 0.35)',
              transition: 'all 0.25s',
              opacity: connecting ? 0.75 : 1,
              transform: connecting ? 'none' : 'translateY(0)',
            }}
            onMouseEnter={(e) => {
              if (!connecting) {
                e.currentTarget.style.transform = 'translateY(-3px)'
                e.currentTarget.style.boxShadow = '0 12px 40px rgba(131, 58, 180, 0.5)'
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 8px 32px rgba(131, 58, 180, 0.35)'
            }}
          >
            {connecting ? (
              <>
                <Loader2 size={22} style={{ animation: 'spin 1s linear infinite' }} />
                {tr('Redirecting to Instagram...', 'Instagramへリダイレクト中...')}
              </>
            ) : (
              <>
                <Instagram size={22} />
                {tr('Connect with Instagram', 'Instagramで接続')}
              </>
            )}
          </button>
        </div>

        {/* How it works */}
        <div style={{
          borderTop: '1px solid var(--ui-flow-border)',
          padding: '1.5rem 0 0',
          marginTop: '0.5rem',
        }}>
          <div style={{
            fontSize: '0.85rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--text-secondary)',
            marginBottom: '1.25rem',
          }}>
            {tr('How it works', '接続の流れ')}
          </div>
          <div style={{ display: 'grid', gap: '1rem' }}>
            {[
              { num: '1', text: tr('Click "Connect with Instagram" above', '上の「Instagramで接続」をクリック') },
              { num: '2', text: tr('Log in with your Instagram account and approve permissions', 'Instagramアカウントでログインして権限を許可') },
              { num: '3', text: tr('Done! Your bot starts replying to DMs automatically', '完了です。ボットがDMへ自動返信を開始します') },
            ].map((step) => (
              <div key={step.num} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '10px',
                  background: 'linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.85rem',
                  fontWeight: 700,
                  color: '#fff',
                  flexShrink: 0,
                }}>
                  {step.num}
                </div>
                <span style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>{step.text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Requirements note */}
        <div style={{
          borderTop: '1px solid var(--ui-flow-border)',
          padding: '1.25rem 0 0',
          marginTop: '1.5rem',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.75rem',
            padding: '1rem',
            background: 'var(--ui-flow-surface)',
            borderRadius: '12px',
            border: '1px solid var(--ui-flow-border)',
          }}>
            <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px', color: '#833ab4' }} />
            <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-primary)' }}>{tr('Requirements:', '要件:')}</strong>{' '}
              {tr('Your Instagram account must be a', 'Instagramアカウントは')}{' '}
              <strong>{tr('Professional account', 'プロアカウント')}</strong>{' '}
              {tr('(Business or Creator).', '（ビジネスまたはクリエイター）である必要があります。')}{' '}
              <a
                href="https://help.instagram.com/502981923235522"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#833ab4', textDecoration: 'none', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '3px' }}
              >
                {tr('Learn how to switch', '切り替え方法を見る')} <ExternalLink size={12} />
              </a>
            </div>
          </div>
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
