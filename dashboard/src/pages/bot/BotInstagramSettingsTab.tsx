import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { Check, CheckCircle, Copy, ExternalLink, AlertCircle, Loader2, Trash2, Zap, Instagram } from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton, GlassCard, GlassField } from '../../components/ui'

type InstagramChannelConfig = {
  channel_id: string
  bot_id: string
  org_id: string
  ig_page_id: string
  verify_token: string
  is_active: boolean
  created_at: string
  updated_at: string
}

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

export default function BotInstagramSettingsTab() {
  const { botId } = useParams()
  const { selectedBot } = useDashboardData()
  const { getAccessTokenSilently } = useAuth0()

  const [igPageId, setIgPageId] = useState('')
  const [appSecret, setAppSecret] = useState('')
  const [pageAccessToken, setPageAccessToken] = useState('')
  const [isActive, setIsActive] = useState(true)
  const [existing, setExisting] = useState<InstagramChannelConfig | null>(null)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [verifyTokenCopied, setVerifyTokenCopied] = useState(false)

  const webhookUrl = botId ? `${API_BASE}/webhooks/instagram/${botId}` : ''

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
      setIgPageId(data.ig_page_id)
      setIsActive(data.is_active)
    } catch (err) {
      if ((err as Error).message?.includes('404') || (err as Error).message?.includes('Not Found')) {
        setExisting(null)
      } else {
        setError((err as Error).message)
      }
    }
  }, [botId, authedFetch])

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
        ig_page_id: igPageId.trim(),
        app_secret: appSecret.trim(),
        page_access_token: pageAccessToken.trim(),
        is_active: isActive,
      }
      if (!body.ig_page_id) {
        throw new Error('Instagram Page ID is required')
      }
      if (!existing && (!body.app_secret || !body.page_access_token)) {
        throw new Error('App Secret and Page Access Token are required for initial setup')
      }
      const resp = await authedFetch(`/v1/org/bots/${botId}/instagram-channel`, {
        method: 'PUT',
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      const data = (await resp.json()) as InstagramChannelConfig
      setExisting(data)
      setAppSecret('')
      setPageAccessToken('')
      setSuccess('Connected! Your bot is live on Instagram.')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
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

  async function handleDelete() {
    if (!botId) return
    const confirmed = window.confirm('Disconnect Instagram integration?')
    if (!confirmed) return
    setDeleting(true)
    setError(null)
    setSuccess(null)
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/instagram-channel`, { method: 'DELETE' })
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}))
        throw new Error((data as { detail?: string }).detail || resp.statusText)
      }
      setExisting(null)
      setIgPageId('')
      setAppSecret('')
      setPageAccessToken('')
      setIsActive(true)
      setTestResult(null)
      setSuccess('Instagram integration disconnected.')
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

  if (!selectedBot || !botId) {
    return <div className="empty-panel">Select a bot to configure Instagram integration.</div>
  }

  /* ─── Already connected view ─────────────────────────────────── */
  if (existing) {
    return (
      <AnimatedPage className="page-body">
        <SectionHeader
          eyebrow="Integrations"
          title="Instagram channel"
          subtitle="Your bot is live and responding to DMs on Instagram."
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
                  Connected & Active
                </div>
                <div style={{
                  color: 'rgba(255,255,255,0.85)',
                  fontSize: '0.95rem',
                  fontWeight: 500,
                }}>
                  Account ID: <code style={{
                    background: 'rgba(0,0,0,0.2)',
                    padding: '2px 8px',
                    borderRadius: '6px',
                    fontFamily: 'monospace',
                  }}>{existing.ig_page_id}</code> • {existing.is_active ? 'Active' : 'Paused'}
                </div>
              </div>
            </div>

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
                  Testing...
                </>
              ) : (
                <>
                  <Zap size={18} />
                  Test Connection
                </>
              )}
            </button>
          </div>
        </div>

        {/* Test Result - Clean inline text */}
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

        {/* Cards Grid */}
        <div style={{ display: 'grid', gap: '1.5rem' }}>
          {/* Webhook URL Card */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1rem' }}>
              Webhook URL
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <code style={{
                flex: 1,
                padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)',
                borderRadius: '10px',
                fontSize: '0.9rem',
                wordBreak: 'break-all',
                border: '1.5px solid var(--ui-flow-border)',
              }}>
                {webhookUrl}
              </code>
              <UiButton
                variant={copied ? "primary" : "secondary"}
                onClick={copyWebhookUrl}
                style={{ padding: '0.85rem 1.1rem' }}
              >
                {copied ? <Check size={18} /> : <Copy size={18} />}
              </UiButton>
            </div>
          </GlassCard>

          {/* Verify Token Card */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '0.5rem' }}>
              Verify Token
            </div>
            <p style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Copy this token and paste it into Facebook's webhook configuration.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <code style={{
                flex: 1,
                padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)',
                borderRadius: '10px',
                fontSize: '0.9rem',
                wordBreak: 'break-all',
                border: '1.5px solid var(--ui-flow-border)',
                fontFamily: 'monospace',
              }}>
                {existing.verify_token}
              </code>
              <UiButton
                variant={verifyTokenCopied ? "primary" : "secondary"}
                onClick={() => {
                  navigator.clipboard.writeText(existing.verify_token)
                  setVerifyTokenCopied(true)
                  setTimeout(() => setVerifyTokenCopied(false), 2000)
                }}
                style={{ padding: '0.85rem 1.1rem' }}
              >
                {verifyTokenCopied ? <Check size={18} /> : <Copy size={18} />}
              </UiButton>
            </div>
          </GlassCard>

          {/* Update Credentials Card */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1.25rem' }}>
              Update Credentials
            </div>
            <div style={{ display: 'grid', gap: '1.25rem' }}>
              <GlassField label="Instagram Page ID">
                <input
                  type="text"
                  value={igPageId}
                  onChange={(e) => setIgPageId(e.target.value)}
                />
              </GlassField>
              <GlassField label="App Secret">
                <input
                  type="password"
                  value={appSecret}
                  onChange={(e) => setAppSecret(e.target.value)}
                  placeholder="Leave blank to keep current"
                />
              </GlassField>
              <GlassField label="Page Access Token">
                <input
                  type="password"
                  value={pageAccessToken}
                  onChange={(e) => setPageAccessToken(e.target.value)}
                  placeholder="Leave blank to keep current"
                />
              </GlassField>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <input
                  type="checkbox"
                  id="instagram-active-edit"
                  checked={isActive}
                  onChange={(e) => setIsActive(e.target.checked)}
                />
                <label htmlFor="instagram-active-edit" style={{ fontSize: '0.95rem', fontWeight: 500 }}>
                  Active
                </label>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
                <UiButton variant="primary" onClick={handleSave} disabled={saving}>
                  {saving ? 'Saving...' : 'Save Changes'}
                </UiButton>
                <UiButton
                  variant="secondary"
                  onClick={handleDelete}
                  disabled={deleting}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#e74c3c', borderColor: '#e74c3c' }}
                >
                  <Trash2 size={16} />
                  {deleting ? 'Removing...' : 'Disconnect'}
                </UiButton>
              </div>
            </div>
          </GlassCard>
        </div>
      </AnimatedPage>
    )
  }

  /* ─── Setup wizard (not yet connected) ─────────────────────── */
  return (
    <AnimatedPage className="page-body">
      <SectionHeader
        eyebrow="Integrations"
        title="Connect Instagram"
        subtitle="Step-by-step setup for instant AI responses to Instagram DMs."
      />

      {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
      {success && <div style={{ marginBottom: '1.5rem', color: '#833ab4', fontWeight: 600 }}>{success}</div>}

      {/* Step Cards */}
      <div style={{ display: 'grid', gap: '1.5rem' }}>
        {[
          {
            num: 1,
            title: 'Create or link Instagram Business Account',
            content: (
              <>
                <p style={{ margin: '0 0 1rem 0' }}>You need an Instagram Business Account linked to a Facebook Page.</p>
                <ol style={{ margin: 0, paddingLeft: '1.25rem', lineHeight: '1.8' }}>
                  <li>
                    Open the{' '}
                    <a
                      href="https://developers.facebook.com/apps"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: 'var(--ui-flow-accent)', fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                    >
                      Facebook Developers Console <ExternalLink size={14} />
                    </a>
                  </li>
                  <li>Create a new app or select an existing one</li>
                  <li>Add the <strong>Instagram</strong> product</li>
                  <li>Link your Instagram Business Account to your Facebook Page</li>
                </ol>
              </>
            ),
          },
          {
            num: 2,
            title: 'Get your credentials',
            content: (
              <div style={{ display: 'grid', gap: '1.25rem' }}>
                <p style={{ margin: 0 }}>Find these values in your Facebook App settings:</p>
                <GlassField
                  label="Instagram Page ID"
                  helper="found in Instagram settings"
                >
                  <input
                    type="text"
                    value={igPageId}
                    onChange={(e) => setIgPageId(e.target.value)}
                    placeholder="e.g. 17841400123456789"
                  />
                </GlassField>
                <GlassField
                  label="App Secret"
                  helper="from App Dashboard"
                >
                  <input
                    type="password"
                    value={appSecret}
                    onChange={(e) => setAppSecret(e.target.value)}
                    placeholder="Paste your app secret"
                  />
                </GlassField>
                <GlassField
                  label="Page Access Token"
                  helper="generate in Messenger settings"
                >
                  <input
                    type="password"
                    value={pageAccessToken}
                    onChange={(e) => setPageAccessToken(e.target.value)}
                    placeholder="Paste your page access token"
                  />
                </GlassField>
              </div>
            ),
          },
          {
            num: 3,
            title: 'Configure webhook',
            content: (
              <>
                <p style={{ margin: '0 0 1rem 0' }}>
                  In your Facebook App, go to <strong>Products → Webhooks</strong> and set up a webhook for Instagram:
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                  <code
                    style={{
                      flex: 1,
                      padding: '0.85rem 1rem',
                      background: 'var(--ui-flow-surface)',
                      borderRadius: '10px',
                      fontSize: '0.9rem',
                      wordBreak: 'break-all',
                      border: '1.5px solid var(--ui-flow-border)',
                    }}
                  >
                    {webhookUrl}
                  </code>
                  <UiButton
                    variant={copied ? "primary" : "secondary"}
                    onClick={copyWebhookUrl}
                    style={{ padding: '0.85rem 1.1rem' }}
                  >
                    {copied ? <Check size={18} /> : <Copy size={18} />}
                  </UiButton>
                </div>
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  Subscribe to <strong>messages</strong> and <strong>messaging_postbacks</strong> events.
                </p>
              </>
            ),
          },
          {
            num: 4,
            title: 'Connect',
            content: (
              <>
                <p style={{ margin: '0 0 1.25rem 0' }}>
                  Once you've completed steps 1-3, click the button below to connect your Instagram channel.
                </p>
                <UiButton variant="primary" onClick={handleSave} disabled={saving} style={{ fontSize: '1rem', padding: '0.85rem 2rem' }}>
                  {saving ? 'Connecting...' : 'Connect Instagram Channel'}
                </UiButton>
              </>
            ),
          },
        ].map((step) => (
          <GlassCard key={step.num}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
              <div
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '12px',
                  background: 'linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.1rem',
                  fontWeight: 700,
                  color: '#fff',
                  flexShrink: 0,
                  boxShadow: '0 4px 16px rgba(131, 58, 180, 0.3)',
                }}
              >
                {step.num}
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>{step.title}</div>
            </div>
            <div style={{ marginLeft: '58px', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.7' }}>
              {step.content}
            </div>
          </GlassCard>
        ))}
      </div>
    </AnimatedPage>
  )
}
