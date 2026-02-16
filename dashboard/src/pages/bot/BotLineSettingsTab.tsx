import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import {
  Check, CheckCircle, Copy, ExternalLink, AlertCircle, Loader2,
  Trash2, Zap, MessageCircle, ChevronLeft, ChevronRight,
} from 'lucide-react'
import { AnimatedPage, SectionHeader, UiButton, GlassCard, GlassField } from '../../components/ui'

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

const LINE_GREEN = '#06c755'
const LINE_GRADIENT = 'linear-gradient(135deg, #06c755 0%, #00b140 100%)'

const CARD_STEP_LABELS = [
  'Enable API',
  'Auto-reply',
  'Credentials',
  'Webhook',
  'Connect',
]

/* ─── Progress Bar (for the 5 card steps only, excludes Get Started) ─── */
function StepProgress({ current, total, labels }: { current: number; total: number; labels: string[] }) {
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
        Step {current + 1} of {total} — {labels[current]}
      </div>
    </div>
  )
}

export default function BotLineSettingsTab() {
  const { botId } = useParams()
  const { selectedBot } = useDashboardData()
  const { getAccessTokenSilently } = useAuth0()

  const [lineChannelId, setLineChannelId] = useState('')
  const [lineChannelSecret, setLineChannelSecret] = useState('')
  const [lineAccessToken, setLineAccessToken] = useState('')
  const [isActive, setIsActive] = useState(true)
  const [existing, setExisting] = useState<LineChannelConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)
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

  const loadConfig = useCallback(async () => {
    setLoading(true)
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
    } finally {
      setLoading(false)
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
      if (!body.line_channel_id) throw new Error('Channel ID is required')
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
      setSuccess('Connected! Your bot is live on LINE.')
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
      const resp = await authedFetch(`/v1/org/bots/${botId}/line-channel/test`, { method: 'POST' })
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
    const confirmed = window.confirm('Disconnect LINE integration? Your bot will stop responding on LINE.')
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
    return <div className="empty-panel">Select a bot to configure LINE integration.</div>
  }

  if (loading) {
    return (
      <AnimatedPage className="page-body">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', gap: '0.75rem', color: 'var(--text-secondary)' }}>
          <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
          Loading LINE settings...
        </div>
      </AnimatedPage>
    )
  }

  /* ═══════════════════════════════════════════════════════════════
     Connected View
     ═══════════════════════════════════════════════════════════════ */
  if (existing) {
    return (
      <AnimatedPage className="page-body">
        <SectionHeader
          eyebrow="Integrations"
          title="LINE channel"
          subtitle="Your bot is live and responding to messages on LINE."
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
              <div style={{
                width: '60px', height: '60px', borderRadius: '16px',
                background: 'rgba(255,255,255,0.25)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
              }}>
                <MessageCircle size={30} color="#fff" />
              </div>
              <div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff', marginBottom: '0.25rem' }}>
                  Connected & Active
                </div>
                <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: '0.95rem', fontWeight: 500 }}>
                  Channel{' '}
                  <code style={{ background: 'rgba(0,0,0,0.2)', padding: '2px 8px', borderRadius: '6px', fontFamily: 'monospace' }}>
                    {existing.line_channel_id}
                  </code>
                  {' '}&bull;{' '}{existing.is_active ? 'Active' : 'Paused'}
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
                {testing ? (<><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Testing...</>) : (<><Zap size={18} /> Test Connection</>)}
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
            <div className="card-title" style={{ marginBottom: '1rem' }}>Connection Details</div>
            <div style={{ display: 'grid', gap: '0.75rem', fontSize: '0.95rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Channel ID</span>
                <code style={{ fontSize: '0.85rem', fontFamily: 'monospace' }}>{existing.line_channel_id}</code>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Status</span>
                <span style={{ fontWeight: 600, color: existing.is_active ? '#27ae60' : '#e74c3c' }}>
                  {existing.is_active ? 'Active' : 'Paused'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Connected</span>
                <span style={{ fontWeight: 500 }}>
                  {new Date(existing.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
              </div>
            </div>
          </GlassCard>

          {/* Webhook URL */}
          <GlassCard>
            <div className="card-title" style={{ marginBottom: '1rem' }}>Webhook URL</div>
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
            <div className="card-title" style={{ marginBottom: '1rem' }}>Manage Connection</div>

            {/* Update credentials (collapsed by default) */}
            <details style={{ marginBottom: '1rem' }}>
              <summary style={{ cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '1rem' }}>
                Update credentials
              </summary>
              <div style={{ display: 'grid', gap: '1rem', paddingTop: '0.5rem' }}>
                <GlassField label="Channel ID">
                  <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} />
                </GlassField>
                <GlassField label="Channel Secret">
                  <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)} placeholder="Leave blank to keep current" />
                </GlassField>
                <GlassField label="Channel Access Token">
                  <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)} placeholder="Leave blank to keep current" />
                </GlassField>
                <UiButton variant="primary" onClick={handleSave} disabled={saving}>
                  {saving ? 'Saving...' : 'Save Changes'}
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
                {deleting ? 'Removing...' : 'Disconnect'}
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
        eyebrow="Integrations"
        title="Connect LINE"
        subtitle="Follow the guided steps below to connect your LINE account."
      />

      {error && <div style={{ marginBottom: '1.5rem', color: '#e74c3c', fontWeight: 600 }}>{error}</div>}
      {success && <div style={{ marginBottom: '1.5rem', color: LINE_GREEN, fontWeight: 600 }}>{success}</div>}

      {currentStep > 0 && <StepProgress current={currentStep - 1} total={5} labels={CARD_STEP_LABELS} />}

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
                Connect your LINE account
              </h3>
              <p style={{
                margin: 0, color: 'var(--text-secondary)',
                fontSize: '1rem', maxWidth: '480px', marginLeft: 'auto', marginRight: 'auto', lineHeight: 1.6,
              }}>
                We'll walk you through very simple steps to connect your LINE business account, so your AI Agent can reply to messages automatically.
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
                <AlertCircle size={16} color={LINE_GREEN} /> What you need before starting
              </div>
              <div style={{ color: 'var(--text-secondary)' }}>
                A <strong>LINE Official Account</strong> — this is a <em>business</em> account (different from your personal LINE app).
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
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.75rem' }}>Here's what we'll do in 5 simple steps:</div>
              <div style={{ display: 'grid', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                {[
                  ['1', 'Enable Messaging API (manager.line.biz)'],
                  ['2', 'Turn off Auto-reply (manager.line.biz)'],
                  ['3', 'Copy 3 codes (Developers Console)'],
                  ['4', 'Set your bot\'s address — webhook (LINE will verify)'],
                  ['5', 'Click "Activate" and you\'re done!'],
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
                I have a LINE Official Account — Let's start
                <ChevronRight size={20} />
              </button>
              <div style={{ marginTop: '0.75rem', fontSize: '0.83rem', color: 'var(--text-secondary)' }}>
                Don't have one yet?{' '}
                <a href="https://www.linebiz.com/jp/entry/" target="_blank" rel="noopener noreferrer"
                  style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none' }}>
                  Create it for free first <ExternalLink size={11} style={{ display: 'inline', verticalAlign: 'middle' }} />
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
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>Enable Messaging API</h3>
            </div>

            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'rgba(6,199,85,0.07)',
              borderRadius: '10px', border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.5rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6,
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px', color: LINE_GREEN }} />
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>Important:</strong> Use a computer browser — the Messaging API option isn\'t available in the LINE mobile app.
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
                  {' '}(the Official Account manager) and sign in
                </li>
                <li>Click your business account name</li>
                <li>Click <strong>⚙️ Settings</strong> in the top-right corner</li>
                <li>In the left menu, click <strong>"Messaging API"</strong></li>
                <li>Click the green <strong>"Enable Messaging API"</strong> button</li>
                <li>Enter a Provider name (company/brand) and click <strong>OK</strong></li>
              </ol>

              <div style={{
                margin: '1.25rem 0 0 0',
                padding: '0.85rem 1rem',
                background: 'var(--ui-flow-surface)',
                borderRadius: '10px',
                border: '1px solid var(--ui-flow-border)',
                fontSize: '0.88rem',
              }}>
                ✅ <strong>Done when:</strong> You see a page with <strong>Channel ID</strong> and <strong>Channel Secret</strong>.
              </div>
            </div>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
            }}>
              <input type="checkbox" checked={apiEnabled} onChange={(e) => setApiEnabled(e.target.checked)} />
              ✓ Messaging API enabled — I can see Channel ID and Channel Secret
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
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>Turn off Auto-reply messages</h3>
            </div>

            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'rgba(6,199,85,0.07)',
              borderRadius: '10px', border: '1px solid rgba(6,199,85,0.25)',
              marginBottom: '1.5rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6,
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px', color: LINE_GREEN }} />
              <span>
                <strong style={{ color: 'var(--text-primary)' }}>Why?</strong> LINE sends a default "Thanks for your message!" when someone messages you. Turn it off so only your bot replies — otherwise customers get two replies.
              </span>
            </div>

            <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              Still in{' '}
              <a href="https://manager.line.biz/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                manager.line.biz <ExternalLink size={13} />
              </a>
              :
            </p>
            <ol style={{ margin: 0, paddingLeft: '1.4rem', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 2 }}>
              <li>Click <strong>⚙️ Settings</strong> → <strong>"Response settings"</strong> in the left menu</li>
              <li>Find <strong>"Auto-response messages"</strong> and turn it <strong>OFF</strong></li>
            </ol>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
              marginTop: '1rem',
            }}>
              <input type="checkbox" checked={autoReplyOff} onChange={(e) => setAutoReplyOff(e.target.checked)} />
              ✓ Auto-response messages is OFF
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
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>Copy the 3 codes</h3>
            </div>

            <p style={{ margin: '0 0 1rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              Go to{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                developers.line.biz/console <ExternalLink size={13} />
              </a>
              {' '}and do the following:
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
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>A. Select or create a Provider</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                In the left panel, you'll see a list of <strong>Providers</strong> (like folders for organizing your apps). Click your existing Provider, or click <strong>"Create"</strong> to make a new one (you can name it after your company).
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
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>B. Select your Messaging API channel</div>
              <div style={{ color: 'var(--text-secondary)' }}>
                Under your Provider, you'll see your <strong>Messaging API channel</strong> — this is the one you created in Step 1 when you enabled the Messaging API. Click on it to open its settings.
              </div>
            </div>

            <p style={{ margin: '0 0 1rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              C. Copy these 3 values from the channel page and paste them below:
            </p>

            <div style={{ display: 'grid', gap: '1.25rem', marginBottom: '0.5rem' }}>
              <GlassField
                label="1. Channel ID"
                helper='Click the "Basic settings" tab at the top → find "Channel ID" (a number like 2009138911) near the top → copy it'
              >
                <input type="text" value={lineChannelId} onChange={(e) => setLineChannelId(e.target.value)} placeholder="Paste Channel ID" />
              </GlassField>
              <GlassField
                label="2. Channel Secret"
                helper='Stay on the "Basic settings" tab → scroll down to "Channel secret" → click the copy button next to it'
              >
                <input type="password" value={lineChannelSecret} onChange={(e) => setLineChannelSecret(e.target.value)} placeholder="Paste Channel Secret" />
              </GlassField>
              <GlassField
                label="3. Access Token"
                helper='Click the "Messaging API" tab at the top → scroll to "Channel access token (long-lived)" → if empty, click "Issue" first → then copy the token'
              >
                <input type="password" value={lineAccessToken} onChange={(e) => setLineAccessToken(e.target.value)} placeholder="Paste Access Token" />
              </GlassField>
            </div>

            <p style={{ margin: '1rem 0 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              We save these when you click Next — that way LINE's webhook verification will work in the next step.
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
              <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 600 }}>Set your bot's address (Webhook URL)</h3>
            </div>

            <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
              In{' '}
              <a href="https://developers.line.biz/console/" target="_blank" rel="noopener noreferrer"
                style={{ color: LINE_GREEN, fontWeight: 600, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                developers.line.biz/console <ExternalLink size={13} />
              </a>
              {' '}→ your channel → <strong>"Messaging API"</strong> tab:
            </p>

            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              1. Copy this address:
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
                {copied ? <><Check size={16} /> Copied!</> : <><Copy size={16} /> Copy</>}
              </UiButton>
            </div>

            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>
              2. Paste into <strong>Webhook URL</strong> → <strong>Update</strong> → turn <strong>Use webhook</strong> ON → <strong>Verify</strong>.
            </p>

            <label style={{
              display: 'flex', alignItems: 'center', gap: '0.6rem',
              padding: '0.85rem 1rem', background: 'var(--ui-flow-surface)',
              borderRadius: '10px', border: '1px solid var(--ui-flow-border)',
              cursor: 'pointer', fontSize: '0.95rem', fontWeight: 500,
              marginTop: '1rem',
            }}>
              <input type="checkbox" checked={webhookSet} onChange={(e) => setWebhookSet(e.target.checked)} />
              ✓ Webhook set and Verify passed
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
              Almost done! One last click...
            </h3>
            <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              Your LINE account: <code style={{ fontFamily: 'monospace', fontWeight: 600 }}>{lineChannelId}</code>
            </p>
            <p style={{ margin: '0 0 2rem 0', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Click the button below to activate your AI bot. After this, your bot will start replying to LINE messages automatically!
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
                <><Loader2 size={22} style={{ animation: 'spin 1s linear infinite' }} /> Connecting...</>
              ) : (
                <><MessageCircle size={22} /> Activate My Bot</>
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
              <ChevronLeft size={18} /> Back
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
                <><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Saving...</>
              ) : (
                <>Next <ChevronRight size={18} /></>
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
              <ChevronLeft size={18} /> Back
            </button>
          </div>
        )}
      </GlassCard>
    </AnimatedPage>
  )
}
