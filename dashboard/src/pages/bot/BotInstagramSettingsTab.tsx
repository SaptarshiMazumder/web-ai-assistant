import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { Check, CheckCircle, Copy, ExternalLink, AlertCircle, Loader2, Trash2 } from 'lucide-react'

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
  fontFamily: 'monospace',
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
  const [copiedWebhook, setCopiedWebhook] = useState(false)
  const [copiedVerify, setCopiedVerify] = useState(false)

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
        throw new Error('Page ID is required')
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
      setSuccess('Instagram channel connected successfully! Your bot is now live on Instagram DMs.')
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
    const confirmed = window.confirm('Are you sure you want to disconnect Instagram? The bot will stop responding to Instagram DMs.')
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
    setCopiedWebhook(true)
    setTimeout(() => setCopiedWebhook(false), 2000)
  }

  function copyVerifyToken() {
    if (!existing) return
    navigator.clipboard.writeText(existing.verify_token)
    setCopiedVerify(true)
    setTimeout(() => setCopiedVerify(false), 2000)
  }

  if (!selectedBot || !botId) {
    return <div className="empty-panel">Select a bot to configure Instagram integration.</div>
  }

  /* ─── Already connected view ─────────────────────────────────── */
  if (existing) {
    return (
      <div className="page-body">
        <div style={connectedBanner}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <CheckCircle size={20} />
            <div>
              <strong>Instagram is connected</strong>
              <div style={{ fontSize: '13px', opacity: 0.85, marginTop: '2px' }}>
                Page ID: {existing.ig_page_id} &middot; {existing.is_active ? 'Active' : 'Paused'}
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <code style={{
              flex: 1, padding: '8px 12px', background: 'var(--bg-secondary)', borderRadius: '6px',
              fontSize: '13px', wordBreak: 'break-all', border: '1px solid var(--border-color)',
            }}>
              {webhookUrl}
            </code>
            <button onClick={copyWebhookUrl} title="Copy" style={{ minWidth: '36px', padding: '7px' }}>
              {copiedWebhook ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>Verify Token</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <code style={{
              flex: 1, padding: '8px 12px', background: 'var(--bg-secondary)', borderRadius: '6px',
              fontSize: '13px', wordBreak: 'break-all', border: '1px solid var(--border-color)',
            }}>
              {existing.verify_token}
            </code>
            <button onClick={copyVerifyToken} title="Copy" style={{ minWidth: '36px', padding: '7px' }}>
              {copiedVerify ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
        </div>

        {/* Update credentials */}
        <div style={stepCard}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px' }}>Update Credentials</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <label style={inputLabel}>Page ID</label>
              <input type="text" value={igPageId} onChange={(e) => setIgPageId(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>App Secret</label>
              <input type="password" value={appSecret} onChange={(e) => setAppSecret(e.target.value)}
                placeholder="Leave blank to keep current" style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>Page Access Token</label>
              <input type="password" value={pageAccessToken} onChange={(e) => setPageAccessToken(e.target.value)}
                placeholder="Leave blank to keep current" style={inputStyle} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" id="ig-active-edit" checked={isActive} onChange={(e) => setIsActive(e.target.checked)} />
              <label htmlFor="ig-active-edit" style={{ fontSize: '14px' }}>Active</label>
            </div>
            <div style={{ display: 'flex', gap: '10px', marginTop: '4px' }}>
              <button className="primary" onClick={handleSave} disabled={saving} style={{ fontSize: '13px' }}>
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
              <button onClick={handleDelete} disabled={deleting}
                style={{ fontSize: '13px', color: 'var(--error-text, #dc3545)', background: 'transparent', border: '1px solid var(--error-text, #dc3545)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Trash2 size={14} /> {deleting ? 'Removing...' : 'Disconnect Instagram'}
              </button>
            </div>
          </div>
        </div>

        {/* Escalation info */}
        <div style={stepCard}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>Human Escalation</div>
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.7' }}>
            When a customer says <strong>&quot;staff&quot;</strong> or <strong>&quot;human&quot;</strong>, the bot steps aside.
            Your staff can reply directly from <strong>Instagram</strong>.
            Customer says <strong>&quot;back to bot&quot;</strong> to return to AI.
          </p>
        </div>
      </div>
    )
  }

  /* ─── Setup wizard (not yet connected) ─────────────────────── */
  return (
    <div className="page-body">
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ margin: '0 0 6px 0', fontSize: '20px' }}>Connect Instagram to your bot</h2>
        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '14px' }}>
          Follow these 4 steps to let your customers chat with your AI bot through Instagram DMs.
        </p>
      </div>

      {error && <div className="error-message" style={{ marginBottom: '16px' }}>{error}</div>}
      {success && <div style={successBanner}><CheckCircle size={16} />{success}</div>}

      {/* Step 1 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: '#E1306C', color: '#fff' }}>1</div>
          <div style={stepTitle}>Create a Meta App with Instagram Messaging</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 10px 0' }}>
            If you already have a Meta App with Instagram Messaging enabled, skip to step 2.
          </p>
          <ol style={{ margin: '0 0 10px 0', paddingLeft: '18px' }}>
            <li>
              Go to{' '}
              <a href="https://developers.facebook.com/apps/" target="_blank" rel="noopener noreferrer" style={linkStyle}>
                Meta App Dashboard <ExternalLink size={12} />
              </a>
            </li>
            <li>Click <strong>&quot;Create App&quot;</strong> and select <strong>&quot;Business&quot;</strong> type</li>
            <li>Under <strong>Products</strong>, add <strong>&quot;Messenger&quot;</strong> and enable <strong>&quot;Instagram Messaging&quot;</strong></li>
            <li>Connect your <strong>Instagram Professional account</strong> (Business or Creator) to a Facebook Page</li>
          </ol>
        </div>
      </div>

      {/* Step 2 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: '#E1306C', color: '#fff' }}>2</div>
          <div style={stepTitle}>Copy your credentials</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 10px 0' }}>
            Find and copy these 3 values from your Meta App:
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '12px' }}>
            <div>
              <label style={inputLabel}>
                Page ID <span style={{ fontWeight: 400, color: 'var(--text-tertiary)' }}>&#8212; the Facebook Page linked to your Instagram account</span>
              </label>
              <input type="text" value={igPageId} onChange={(e) => setIgPageId(e.target.value)}
                placeholder="e.g. 123456789012345" style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>
                App Secret <span style={{ fontWeight: 400, color: 'var(--text-tertiary)' }}>&#8212; App Dashboard &rarr; Settings &rarr; Basic</span>
              </label>
              <input type="password" value={appSecret} onChange={(e) => setAppSecret(e.target.value)}
                placeholder="Paste your app secret" style={inputStyle} />
            </div>
            <div>
              <label style={inputLabel}>
                Page Access Token <span style={{ fontWeight: 400, color: 'var(--text-tertiary)' }}>&#8212; Messenger &rarr; Instagram Settings &rarr; Generate Token</span>
              </label>
              <input type="password" value={pageAccessToken} onChange={(e) => setPageAccessToken(e.target.value)}
                placeholder="Paste the page access token" style={inputStyle} />
            </div>
          </div>
        </div>
      </div>

      {/* Step 3 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: '#E1306C', color: '#fff' }}>3</div>
          <div style={stepTitle}>Set up your webhook in Meta App Dashboard</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 10px 0' }}>
            First, click <strong>&quot;Connect Instagram Channel&quot;</strong> in step 4 below to generate your Verify Token.
            Then, in your Meta App Dashboard under <strong>Messenger &rarr; Webhooks</strong>:
          </p>
          <ol style={{ margin: '0 0 10px 0', paddingLeft: '18px' }}>
            <li>Click <strong>&quot;Add Callback URL&quot;</strong></li>
            <li>Paste the <strong>Webhook URL</strong> shown below</li>
            <li>Paste the <strong>Verify Token</strong> (shown after connecting)</li>
            <li>Click <strong>&quot;Verify and Save&quot;</strong></li>
            <li>Subscribe to the <strong>&quot;messages&quot;</strong> field</li>
          </ol>
          <div style={{ marginBottom: '12px' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>Webhook URL</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <code style={{
                flex: 1, padding: '10px 14px', background: 'var(--bg-secondary)', borderRadius: '6px',
                fontSize: '13px', wordBreak: 'break-all', border: '1px solid var(--border-color)',
              }}>
                {webhookUrl}
              </code>
              <button onClick={copyWebhookUrl} title="Copy webhook URL" style={{ minWidth: '40px', padding: '8px' }}>
                {copiedWebhook ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Step 4 */}
      <div style={stepCard}>
        <div style={stepHeader}>
          <div style={{ ...stepNumber, background: '#E1306C', color: '#fff' }}>4</div>
          <div style={stepTitle}>Connect</div>
        </div>
        <div style={stepBody}>
          <p style={{ margin: '0 0 14px 0' }}>
            Once you&apos;ve completed steps 1-2, click the button below to connect. You&apos;ll then get a Verify Token for step 3.
          </p>
          <button className="primary" onClick={handleSave} disabled={saving}
            style={{ fontSize: '15px', padding: '10px 28px' }}>
            {saving ? 'Connecting...' : 'Connect Instagram Channel'}
          </button>
        </div>
      </div>
    </div>
  )
}
