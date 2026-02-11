import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { Check, CheckCircle, Copy, ExternalLink, AlertCircle, Loader2, Trash2 } from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton } from '../../components/ui'

type LineChannelConfig = {
  channel_id: string
  bot_id: string
  org_id: string
  line_channel_id: string
  is_active: boolean
  created_at: string
  updated_at: string
}

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

/* ─── Styles ─────────────────────────────────────────────────────── */

const stepCard: React.CSSProperties = {
  border: '1px solid var(--border-color)',
  borderRadius: '10px',
  padding: '20px 24px',
  marginBottom: '16px',
  background: 'var(--bg-primary)',
}

const stepHeader: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  marginBottom: '12px',
}

const stepNumber: React.CSSProperties = {
  width: '28px',
  height: '28px',
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '13px',
  fontWeight: 600,
  flexShrink: 0,
}

const stepTitle: React.CSSProperties = {
  fontSize: '15px',
  fontWeight: 600,
}

const stepBody: React.CSSProperties = {
  marginLeft: '40px',
  color: 'var(--text-secondary)',
  fontSize: '14px',
  lineHeight: '1.7',
}

const inputLabel: React.CSSProperties = {
  display: 'block',
  marginBottom: '6px',
  fontWeight: 500,
  fontSize: '13px',
  color: 'var(--text-primary)',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '9px 12px',
  borderRadius: '6px',
  border: '1px solid var(--border-color)',
  fontSize: '14px',
  fontFamily: 'Google Sans, sans-serif',
  background: 'var(--bg-secondary)',
}

const linkStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
  color: 'var(--accent-color, #2563eb)',
  fontWeight: 500,
  textDecoration: 'none',
}

const successBanner: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
  padding: '14px 18px',
  background: 'var(--success-bg, #d4edda)',
  color: 'var(--success-text, #155724)',
  borderRadius: '8px',
  fontSize: '14px',
  marginBottom: '16px',
}

const connectedBanner: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '16px 20px',
  background: 'var(--success-bg, #d4edda)',
  color: 'var(--success-text, #155724)',
  borderRadius: '10px',
  marginBottom: '20px',
}

/* ─── Component ──────────────────────────────────────────────────── */

export default function BotLineSettingsTab() {
  const { botId } = useParams()
  const { selectedBot } = useDashboardData()
  const { getAccessTokenSilently } = useAuth0()

  const [lineChannelId, setLineChannelId] = useState('')
  const [lineChannelSecret, setLineChannelSecret] = useState('')
  const [lineAccessToken, setLineAccessToken] = useState('')
  const [isActive, setIsActive] = useState(true)
  const [existing, setExisting] = useState<LineChannelConfig | null>(null)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

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

  const loadConfig = useCallback(async () => {
    try {
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel`)
      if (resp.status === 404) {
        setExisting(null)
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
        line_channel_id: lineChannelId.trim(),
        line_channel_secret: lineChannelSecret.trim(),
        line_channel_access_token: lineAccessToken.trim(),
        is_active: isActive,
      }
      if (!body.line_channel_id) {
        throw new Error('Channel ID is required')
      }
      if (!existing && (!body.line_channel_secret || !body.line_channel_access_token)) {
        throw new Error('Channel Secret and Access Token are required for initial setup')
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
      setSuccess('LINE channel connected successfully! Your bot is now live on LINE.')
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
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel/test`, {
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
    const confirmed = window.confirm('Are you sure you want to disconnect LINE? The bot will stop responding to LINE messages.')
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
      setSuccess('LINE integration disconnected.')
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
    return <div className="empty-panel">Select a bot to configure LINE integration.</div>
  }

  /* ─── Already connected view ─────────────────────────────────── */
  if (existing) {
    return (
      <AnimatedPage className="page-body">
        <SectionHeader eyebrow="Integrations" title="LINE channel" subtitle="Control webhook health, credentials, and live status in one place." />
        <div style={connectedBanner}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <CheckCircle size={20} />
            <div>
              <strong>LINE is connected</strong>
              <div style={{ fontSize: '13px', opacity: 0.85, marginTop: '2px' }}>
                Channel ID: {existing.line_channel_id} &middot; {existing.is_active ? 'Active' : 'Paused'}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleTestConnection}
              disabled={testing}
              style={{ fontSize: '13px', padding: '6px 14px' }}
            >
              {testing ? <><Loader2 size={14} style={{ animation: 'spin 1s linear infinite', marginRight: '4px' }} /> Testing...</> : 'Test Connection'}
            </button>
          </div>
        </div>

        {testResult && (
          <div style={{
            ...successBanner,
            background: testResult.ok ? 'var(--success-bg, #d4edda)' : 'var(--error-bg, #f8d7da)',
            color: testResult.ok ? 'var(--success-text, #155724)' : 'var(--error-text, #721c24)',
          }}>
            {testResult.ok ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
            {testResult.message}
          </div>
        )}

        {error && <div className="error-message" style={{ marginBottom: '16px' }}>{error}</div>}
        {success && <div style={{ ...successBanner }}><CheckCircle size={16} />{success}</div>}

        {/* Webhook URL */}
        <div style={stepCard}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>Webhook URL</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <code style={{
              flex: 1, padding: '8px 12px', background: 'var(--bg-secondary)', borderRadius: '6px',
              fontSize: '13px', wordBreak: 'break-all', border: '1px solid var(--border-color)',
            }}>
              {webhookUrl}
            </code>
            <button onClick={copyWebhookUrl} title="Copy" style={{ minWidth: '36px', padding: '7px' }}>
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
        </div>

        {/* Update credentials */}
        <div style={stepCard}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px' }}>Update Credentials</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <label style={inputLabel}>Channel ID</label>
              <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>Channel Secret</label>
              <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)}
                placeholder="Leave blank to keep current" style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>Channel Access Token</label>
              <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)}
                placeholder="Leave blank to keep current" style={inputStyle} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" id="line-active-edit" checked={isActive} onChange={(e) => setIsActive(e.target.checked)} />
              <label htmlFor="line-active-edit" style={{ fontSize: '14px' }}>Active</label>
            </div>
            <div style={{ display: 'flex', gap: '10px', marginTop: '4px' }}>
              <UiButton variant="primary" onClick={handleSave} disabled={saving} style={{ fontSize: '13px' }}>
                {saving ? 'Saving...' : 'Save Changes'}
              </UiButton>
              <button onClick={handleDelete} disabled={deleting}
                style={{ fontSize: '13px', color: 'var(--error-text, #dc3545)', background: 'transparent', border: '1px solid var(--error-text, #dc3545)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Trash2 size={14} /> {deleting ? 'Removing...' : 'Disconnect LINE'}
              </button>
            </div>
          </div>
        </div>

        {/* Escalation info */}
        <div style={stepCard}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>Human Escalation</div>
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.7' }}>
            When a customer says <strong>"staff"</strong> or <strong>"human"</strong>, the bot steps aside.
            Your staff can reply directly in the <strong>LINE Official Account Manager</strong> app.
            Customer says <strong>"back to bot"</strong> to return to AI.
          </p>
        </div>
      </AnimatedPage>
    )
  }

  /* ─── Setup wizard (not yet connected) ─────────────────────── */
  return (
    <AnimatedPage className="page-body">
      <SectionHeader eyebrow="Integrations" title="Connect LINE" subtitle="Step-by-step channel setup for instant AI responses on LINE." />
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ margin: '0 0 6px 0', fontSize: '20px' }}>Connect LINE to your bot</h2>
        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '14px' }}>
          Follow these 4 steps to let your customers chat with your AI bot through LINE.
        </p>
      </div>

      {error && <div className="error-message" style={{ marginBottom: '16px' }}>{error}</div>}
      {success && <div style={successBanner}><CheckCircle size={16} />{success}</div>}

      {/* Step 1 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: 'var(--accent-color, #2563eb)', color: '#fff' }}>1</div>
          <div style={stepTitle}>Create a LINE Messaging API channel</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 10px 0' }}>
            If you already have a LINE Official Account, skip to step 2.
          </p>
          <ol style={{ margin: '0 0 10px 0', paddingLeft: '18px' }}>
            <li>
              Open the{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer" style={linkStyle}>
                LINE Developer Console <ExternalLink size={12} />
              </a>
            </li>
            <li>Click <strong>"Create a new provider"</strong> (or select an existing one)</li>
            <li>Click <strong>"Create a Messaging API channel"</strong></li>
            <li>Fill in your business name, description, and category</li>
          </ol>
        </div>
      </div>

      {/* Step 2 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: 'var(--accent-color, #2563eb)', color: '#fff' }}>2</div>
          <div style={stepTitle}>Copy your channel credentials</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 10px 0' }}>
            In your channel&apos;s settings, find and copy these 3 values:
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '12px' }}>
            <div>
              <label style={inputLabel}>
                Channel ID <span style={{ fontWeight: 400, color: 'var(--text-tertiary)' }}>&#8212; found under "Basic settings"</span>
              </label>
              <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)}
                placeholder="e.g. 1234567890" style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>
                Channel Secret <span style={{ fontWeight: 400, color: 'var(--text-tertiary)' }}>&#8212; found under "Basic settings"</span>
              </label>
              <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)}
                placeholder="Paste your channel secret" style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>
                Channel Access Token <span style={{ fontWeight: 400, color: 'var(--text-tertiary)' }}>&#8212; under "Messaging API", click "Issue"</span>
              </label>
              <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)}
                placeholder="Paste the long-lived token" style={inputStyle} />
            </div>
          </div>
        </div>
      </div>

      {/* Step 3 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: 'var(--accent-color, #2563eb)', color: '#fff' }}>3</div>
          <div style={stepTitle}>Set your webhook URL in LINE</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 10px 0' }}>
            Copy this URL and paste it in your channel&apos;s <strong>Messaging API &rarr; Webhook URL</strong> field. Then turn on <strong>"Use webhook"</strong>.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <code style={{
              flex: 1, padding: '10px 14px', background: 'var(--bg-secondary)', borderRadius: '6px',
              fontSize: '13px', wordBreak: 'break-all', border: '1px solid var(--border-color)',
            }}>
              {webhookUrl}
            </code>
            <button onClick={copyWebhookUrl} title="Copy webhook URL" style={{ minWidth: '40px', padding: '8px' }}>
              {copied ? <Check size={16} /> : <Copy size={16} />}
            </button>
          </div>
          <p style={{ margin: 0, fontSize: '13px' }}>
            Also go to <strong>LINE Official Account features &rarr; Response settings</strong> and enable both <strong>Webhook</strong> and <strong>Chat</strong>.
            Disable <strong>Auto-reply messages</strong>.
          </p>
        </div>
      </div>

      {/* Step 4 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: 'var(--accent-color, #2563eb)', color: '#fff' }}>4</div>
          <div style={stepTitle}>Connect</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 14px 0' }}>
            Once you&apos;ve completed steps 1-3, click the button below to connect your LINE channel.
          </p>
          <UiButton variant="primary" onClick={handleSave} disabled={saving}
            style={{ fontSize: '15px', padding: '10px 28px' }}>
            {saving ? 'Connecting...' : 'Connect LINE Channel'}
          </UiButton>
        </div>
      </div>
    </AnimatedPage>
  )
}
