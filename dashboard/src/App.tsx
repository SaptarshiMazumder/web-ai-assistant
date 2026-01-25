import { useEffect, useMemo, useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'

type BotSummary = {
  bot_id: string
  org_id?: string
  display_name: string
  publishable_key: string
  secret_key: string
  created_at: string
  updated_at: string
}

type BotCreateResponse = {
  bot_id: string
  display_name: string
  publishable_key: string
  secret_key: string
}

type DomainRecord = {
  bot_id: string
  hostname: string
  status: string
  verification_token: string
  verification_url?: string
  verified_at?: string | null
  created_at: string
  updated_at: string
}

type JobRecord = {
  job_id: string
  url: string
  hostname: string
  stage: string
  pages_crawled: number
  docs_count: number
  gcs_prefix: string
  last_error: string
  created_at: string
  updated_at: string
}

type IndexStatus = {
  status: string
  stage?: string
  pages_crawled?: number
  last_crawled_url?: string
  docs_count?: number
  last_error?: string
  gcs_prefix?: string
  updated_at?: string
}

type OrgSummary = {
  org_id: string
  name: string
  status: string
  created_at: string
  updated_at: string
}

type OrgMember = {
  user_id: string
  email: string
  role: string
  created_at: string
  updated_at: string
}

type TokenClaims = Record<string, unknown>

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

const terminalStages = new Set(['done', 'error', 'cancelled', 'import_submitted'])

async function fetchJson<T>(path: string, init?: RequestInit, token?: string): Promise<T> {
  const initHeaders = init?.headers
  const headerEntries =
    initHeaders instanceof Headers ? Object.fromEntries(initHeaders.entries()) : (initHeaders as Record<string, string> | undefined)
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(headerEntries || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      detail = body.detail || detail
    } catch {
      // ignore json parse error
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export default function App() {
  const [bots, setBots] = useState<BotSummary[]>([])
  const [selectedBotId, setSelectedBotId] = useState<string | null>(null)
  const [selectedBot, setSelectedBot] = useState<BotSummary | null>(null)
  const [domains, setDomains] = useState<DomainRecord[]>([])
  const [jobs, setJobs] = useState<JobRecord[]>([])
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newBotName, setNewBotName] = useState('')
  const [newDomain, setNewDomain] = useState('')
  const [crawlUrl, setCrawlUrl] = useState('')
  const [activeCrawlUrl, setActiveCrawlUrl] = useState('')
  const [orgs, setOrgs] = useState<OrgSummary[]>([])
  const [activeOrgId, setActiveOrgId] = useState<string | null>(null)
  const [orgMembers, setOrgMembers] = useState<OrgMember[]>([])
  const [newOrgName, setNewOrgName] = useState('')
  const [orgDisplayName, setOrgDisplayName] = useState('')
  const [orgDisplayNameInput, setOrgDisplayNameInput] = useState('')
  const [newMemberEmail, setNewMemberEmail] = useState('')
  const [newMemberRole, setNewMemberRole] = useState('org_admin')

  const [isSuperAdmin, setIsSuperAdmin] = useState(false)

  const {
    isAuthenticated,
    isLoading: authLoading,
    loginWithRedirect,
    logout,
    getAccessTokenSilently,
    getIdTokenClaims,
    user,
  } = useAuth0()

  const embedSnippet = useMemo(() => {
    if (!selectedBot) return ''
    return `<script async src="${API_BASE}/widget/widget.js" data-bot-key="${selectedBot.publishable_key}" data-api-base="${API_BASE}"></script>`
  }, [selectedBot])

  async function fetchAuthedJson<T>(path: string, init?: RequestInit): Promise<T> {
    const token = await getAccessTokenSilently()
    return fetchJson<T>(path, init, token)
  }

  function withOrgParam(path: string) {
    if (!activeOrgId) return path
    const suffix = `org_id=${encodeURIComponent(activeOrgId)}`
    return path.includes('?') ? `${path}&${suffix}` : `${path}?${suffix}`
  }

  async function loadBots() {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<{ bots: BotSummary[] }>(withOrgParam('/v1/org/bots'))
      setBots(data.bots)
      if (!activeOrgId && data.bots.length) {
        setActiveOrgId(data.bots[0].org_id || null)
      }
      if (data.bots.length && !selectedBotId) {
        setSelectedBotId(data.bots[0].bot_id)
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function loadBotDetail(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<{ bot: BotSummary }>(withOrgParam(`/v1/org/bots/${botId}`))
      if (!activeOrgId && data.bot.org_id) {
        setActiveOrgId(data.bot.org_id)
      }
      setSelectedBot(data.bot)
    } catch (err) {
      setError((err as Error).message)
      setSelectedBot(null)
    } finally {
      setLoading(false)
    }
  }

  async function loadDomains(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const data = await fetchAuthedJson<{ bot_id: string; domains: DomainRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/domains`)
      )
      setDomains(data.domains)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadJobs(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const data = await fetchAuthedJson<{ bot_id: string; jobs: JobRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/jobs`)
      )
      setJobs(data.jobs)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function createBot() {
    if (!newBotName.trim() || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<BotCreateResponse>(withOrgParam('/v1/org/bots'), {
        method: 'POST',
        body: JSON.stringify({ display_name: newBotName.trim() }),
      })
      setNewBotName('')
      setSelectedBotId(data.bot_id)
      await loadBots()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function loadOrgs() {
    if (!isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<{ orgs: OrgSummary[] }>('/v1/admin/orgs')
      setOrgs(data.orgs)
      if (!activeOrgId && data.orgs.length) {
        setActiveOrgId(data.orgs[0].org_id)
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadSelfOrgs() {
    if (isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<{ org_ids: string[] }>('/v1/org/self')
      const ids = data.org_ids || []
      if (!activeOrgId && ids.length) {
        setActiveOrgId(ids[0])
      }
      if (ids.length) {
        const orgList = await Promise.all(
          ids.map(async (orgId) => {
            try {
              return await fetchAuthedJson<OrgSummary>(`/v1/org/info?org_id=${encodeURIComponent(orgId)}`)
            } catch {
              return {
                org_id: orgId,
                name: orgId,
                status: 'active',
                created_at: '',
                updated_at: '',
              } as OrgSummary
            }
          })
        )
        setOrgs(orgList)
      } else {
        setOrgs([])
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadOrgInfo(orgId: string) {
    if (isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<OrgSummary>(`/v1/org/info?org_id=${encodeURIComponent(orgId)}`)
      setOrgDisplayName(data.name)
      setOrgDisplayNameInput(data.name)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function saveOrgName() {
    if (!activeOrgId || !orgDisplayNameInput.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<OrgSummary>(
        `/v1/org/name?org_id=${encodeURIComponent(activeOrgId)}`,
        {
          method: 'POST',
          body: JSON.stringify({ name: orgDisplayNameInput.trim() }),
        }
      )
      setOrgDisplayName(data.name)
      setOrgDisplayNameInput(data.name)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function createOrg() {
    if (!newOrgName.trim()) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson('/v1/admin/orgs', {
        method: 'POST',
        body: JSON.stringify({ name: newOrgName.trim() }),
      })
      setNewOrgName('')
      await loadOrgs()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function setOrgStatus(orgId: string, status: 'active' | 'disabled') {
    if (!isSuperAdmin) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson(`/v1/admin/orgs/${orgId}/${status === 'active' ? 'enable' : 'disable'}`, {
        method: 'POST',
      })
      await loadOrgs()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function loadOrgMembers(orgId: string) {
    try {
      const path = isSuperAdmin ? `/v1/admin/orgs/${orgId}/members` : `/v1/org/members?org_id=${encodeURIComponent(orgId)}`
      const data = await fetchAuthedJson<{ org_id: string; members: OrgMember[] }>(path)
      setOrgMembers(data.members)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function addOrgMember() {
    if (!activeOrgId || !newMemberEmail.trim()) return
    setLoading(true)
    setError(null)
    try {
      const path = isSuperAdmin ? `/v1/admin/orgs/${activeOrgId}/members` : `/v1/org/members?org_id=${encodeURIComponent(activeOrgId)}`
      await fetchAuthedJson(path, {
        method: 'POST',
        body: JSON.stringify({ email: newMemberEmail.trim(), role: newMemberRole }),
      })
      setNewMemberEmail('')
      await loadOrgMembers(activeOrgId)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function clearGcs() {
    if (!isSuperAdmin) return
    if (!window.confirm('Delete all crawled content from GCS? This cannot be undone.')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson('/v1/admin/reset/gcs', { method: 'POST' })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function clearRag() {
    if (!isSuperAdmin) return
    if (!window.confirm('Delete all RAG corpora? This cannot be undone.')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson('/v1/admin/reset/rag', { method: 'POST' })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function addDomain() {
    if (!selectedBot || !newDomain.trim() || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/domains`), {
        method: 'POST',
        body: JSON.stringify({ hostname: newDomain.trim() }),
      })
      setNewDomain('')
      await loadDomains(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function verifyDomain(hostname: string) {
    if (!selectedBot || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/domains/${hostname}/verify`), {
        method: 'POST',
      })
      await loadDomains(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function startCrawl() {
    if (!selectedBot || !crawlUrl.trim() || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index`), {
        method: 'POST',
        body: JSON.stringify({ url: crawlUrl.trim() }),
      })
      setActiveCrawlUrl(crawlUrl.trim())
      await refreshStatus(crawlUrl.trim())
      await loadJobs(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function cancelCrawl() {
    if (!selectedBot || !activeCrawlUrl || (isSuperAdmin && !activeOrgId)) return
    setLoading(true)
    setError(null)
    try {
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index/cancel`), {
        method: 'POST',
        body: JSON.stringify({ url: activeCrawlUrl }),
      })
      await refreshStatus(activeCrawlUrl)
      await loadJobs(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function refreshStatus(url: string) {
    if (!selectedBot || (isSuperAdmin && !activeOrgId)) return
    try {
      const path = withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index/status?url=${encodeURIComponent(url)}`)
      const status = await fetchAuthedJson<IndexStatus>(path)
      setIndexStatus(status)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    if (!isAuthenticated) return
    void getIdTokenClaims().then((claims: TokenClaims | undefined) => {
      const roles =
        (claims?.roles as string[]) ||
        (claims?.permissions as string[]) ||
        (claims && (claims['https://web-ai/roles'] as string[])) ||
        []
      const env = (import.meta as { env: Record<string, string> }).env
      const adminEmails = (env.VITE_SUPER_ADMIN_EMAILS || '')
        .split(',')
        .map((entry) => entry.trim().toLowerCase())
        .filter(Boolean)
      const claimEmail = (claims?.email as string | undefined)?.toLowerCase()
      const superAdmin =
        roles.includes('super_admin') || roles.includes('owner') || (claimEmail ? adminEmails.includes(claimEmail) : false)
      setIsSuperAdmin(superAdmin)

      const orgClaimKey = (import.meta as { env: Record<string, string> }).env.VITE_AUTH_ORG_CLAIM
      const claimValue = orgClaimKey ? (claims?.[orgClaimKey] as string | string[] | undefined) : undefined
      const orgsFromClaim =
        claimValue ||
        (claims?.org_id as string | string[] | undefined) ||
        (claims?.org as string | string[] | undefined) ||
        (claims?.organization_id as string | string[] | undefined)
      const ids = Array.isArray(orgsFromClaim) ? orgsFromClaim : orgsFromClaim ? [orgsFromClaim] : []
      if (!activeOrgId && ids.length === 1) {
        setActiveOrgId(ids[0])
      }
    })
  }, [isAuthenticated, getIdTokenClaims, activeOrgId])

  useEffect(() => {
    if (!isAuthenticated) return
    if (isSuperAdmin) {
      void loadOrgs()
    } else {
      void loadSelfOrgs()
    }
  }, [isAuthenticated, isSuperAdmin])

  useEffect(() => {
    if (!isAuthenticated || (isSuperAdmin && !activeOrgId)) return
    void loadBots()
  }, [isAuthenticated, activeOrgId, isSuperAdmin])

  useEffect(() => {
    if (activeOrgId) {
      setError(null)
    }
  }, [activeOrgId])

  useEffect(() => {
    if (!activeOrgId) return
    setSelectedBotId(null)
    setSelectedBot(null)
    setDomains([])
    setJobs([])
    setIndexStatus(null)
    setActiveCrawlUrl('')
  }, [activeOrgId])

  useEffect(() => {
    if (!activeOrgId) return
    void loadOrgMembers(activeOrgId)
  }, [isSuperAdmin, activeOrgId])

  useEffect(() => {
    if (!activeOrgId || isSuperAdmin) return
    void loadOrgInfo(activeOrgId)
  }, [activeOrgId, isSuperAdmin])

  useEffect(() => {
    if (!selectedBotId || (isSuperAdmin && !activeOrgId)) return
    void loadBotDetail(selectedBotId)
    void loadDomains(selectedBotId)
    void loadJobs(selectedBotId)
    setIndexStatus(null)
    setActiveCrawlUrl('')
  }, [selectedBotId, activeOrgId])

  useEffect(() => {
    if (!selectedBot || !activeCrawlUrl) return
    if (indexStatus?.stage && terminalStages.has(indexStatus.stage)) return
    const timer = window.setInterval(() => {
      void refreshStatus(activeCrawlUrl)
    }, 4000)
    return () => window.clearInterval(timer)
  }, [selectedBot, activeCrawlUrl, indexStatus?.stage])

  async function copySnippet() {
    if (!embedSnippet) return
    try {
      await navigator.clipboard.writeText(embedSnippet)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  function handleRefresh() {
    if (activeOrgId || !isSuperAdmin) {
      void loadBots()
      if (selectedBotId) {
        void loadBotDetail(selectedBotId)
        void loadDomains(selectedBotId)
        void loadJobs(selectedBotId)
      }
    }
    if (isSuperAdmin) {
      void loadOrgs()
      if (activeOrgId) {
        void loadOrgMembers(activeOrgId)
      }
    }
  }

  if (authLoading) {
    return (
      <div className="app-shell">
        <main className="content">
          <div className="empty-panel">Loading authentication…</div>
        </main>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="app-shell">
        <main className="content">
          <div className="empty-panel">
            <div className="title">Sign in to Web AI Admin</div>
            <button className="primary" onClick={() => loginWithRedirect()}>
              Sign in
            </button>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo-dot" />
          <div>
            <div className="brand-title">Web AI Admin</div>
            <div className="brand-subtitle">Bots & crawl control</div>
          </div>
        </div>

        <div className="section">
          <div className="section-title">Account</div>
          <div className="stack">
            <div className="muted">{user?.email || 'Signed in'}</div>
            <button className="ghost" onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}>
              Sign out
            </button>
          </div>
        </div>

        <div className="section">
          <div className="section-title">Active org</div>
          <div className="stack">
            {isSuperAdmin ? (
              <select value={activeOrgId || ''} onChange={(event) => setActiveOrgId(event.target.value)}>
                <option value="" disabled>
                  Select org
                </option>
                {orgs.map((org) => (
                  <option key={org.org_id} value={org.org_id}>
                    {org.name} ({org.org_id})
                  </option>
                ))}
              </select>
            ) : (
              <div className="muted">{orgDisplayName || activeOrgId || 'No org assigned'}</div>
            )}
          </div>
        </div>

        {isSuperAdmin && (
          <div className="section">
            <div className="section-title">Organizations</div>
            <div className="stack">
              <input
                value={newOrgName}
                onChange={(event) => setNewOrgName(event.target.value)}
                placeholder="Org name"
              />
              <button className="primary" onClick={createOrg} disabled={loading || !newOrgName.trim()}>
                Create org
              </button>
            </div>
            <div className="bot-list">
              {orgs.map((org) => (
                <div key={org.org_id} className="bot-item">
                  <div className="bot-name">{org.name}</div>
                  <div className="bot-id">{org.org_id}</div>
                  <div className="row">
                    <button className="ghost" onClick={() => setActiveOrgId(org.org_id)}>
                      Use
                    </button>
                    {org.status === 'active' ? (
                      <button className="ghost" onClick={() => setOrgStatus(org.org_id, 'disabled')}>
                        Disable
                      </button>
                    ) : (
                      <button className="ghost" onClick={() => setOrgStatus(org.org_id, 'active')}>
                        Enable
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {!orgs.length && <div className="empty">No orgs yet</div>}
            </div>
          </div>
        )}

        {!isSuperAdmin && activeOrgId && (
          <div className="section">
            <div className="section-title">Org settings</div>
            <div className="stack">
              <input
                value={orgDisplayNameInput}
                onChange={(event) => setOrgDisplayNameInput(event.target.value)}
                placeholder="Organization name"
              />
              <button className="secondary" onClick={saveOrgName} disabled={loading || !orgDisplayNameInput.trim()}>
                Save org name
              </button>
            </div>
          </div>
        )}

        {activeOrgId && (
          <div className="section">
            <div className="section-title">Org members</div>
            <div className="stack">
              <input
                value={newMemberEmail}
                onChange={(event) => setNewMemberEmail(event.target.value)}
                placeholder="user@company.com"
              />
              <select value={newMemberRole} onChange={(event) => setNewMemberRole(event.target.value)}>
                <option value="org_admin">org_admin</option>
                <option value="org_member">org_member</option>
              </select>
              <button className="secondary" onClick={addOrgMember} disabled={loading || !newMemberEmail.trim()}>
                Add member
              </button>
            </div>
            <div className="domain-list">
              {orgMembers.map((member) => (
                <div key={member.user_id} className="domain-row">
                  <div>
                    <div className="domain-host">{member.email}</div>
                    <div className="muted">{member.role}</div>
                  </div>
                </div>
              ))}
              {!orgMembers.length && <div className="empty">No members yet.</div>}
            </div>
          </div>
        )}

        {(!isSuperAdmin || activeOrgId) && (
          <div className="section">
            <div className="section-title">Create bot</div>
            <div className="stack">
              <input
                value={newBotName}
                onChange={(event) => setNewBotName(event.target.value)}
                placeholder="Bot display name"
              />
              <button className="primary" onClick={createBot} disabled={loading || !newBotName.trim()}>
                Create bot
              </button>
            </div>
          </div>
        )}

        {isSuperAdmin && (
          <div className="section">
            <div className="section-title">Danger zone</div>
            <div className="stack">
              <button className="ghost" onClick={clearGcs} disabled={loading}>
                Clear all GCS crawls
              </button>
              <button className="ghost" onClick={clearRag} disabled={loading}>
                Clear all RAG corpora
              </button>
            </div>
          </div>
        )}
        <div className="section">
          <div className="section-title">Bots</div>
          <div className="bot-list">
            {bots.map((bot) => (
              <button
                key={bot.bot_id}
                className={`bot-item ${selectedBotId === bot.bot_id ? 'active' : ''}`}
                onClick={() => setSelectedBotId(bot.bot_id)}
              >
                <div className="bot-name">{bot.display_name}</div>
                <div className="bot-id">{bot.bot_id}</div>
              </button>
            ))}
            {!bots.length && <div className="empty">No bots yet</div>}
          </div>
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <div className="title">Admin Dashboard</div>
            <div className="subtitle">Manage bots, domains, and crawls.</div>
          </div>
          <button className="ghost" onClick={handleRefresh} disabled={loading}>
            Refresh
          </button>
        </header>

        {error && <div className="alert error">{error}</div>}
        {loading && <div className="alert">Working...</div>}

        {!activeOrgId && <div className="empty-panel">Select an organization to view details.</div>}
        {activeOrgId && !selectedBot && <div className="empty-panel">Select a bot to view details.</div>}

        {selectedBot && (
          <div className="grid">
            <section className="card">
              <div className="card-title">Bot details</div>
              <div className="detail-row">
                <span>Bot ID</span>
                <code>{selectedBot.bot_id}</code>
              </div>
              <div className="detail-row">
                <span>Publishable key</span>
                <code>{selectedBot.publishable_key}</code>
              </div>
              <div className="detail-row">
                <span>Secret key</span>
                <code>{selectedBot.secret_key}</code>
              </div>
              <div className="detail-row">
                <span>Created</span>
                <span>{new Date(selectedBot.created_at).toLocaleString()}</span>
              </div>
            </section>

            <section className="card">
              <div className="card-title">Embed script</div>
              <p className="muted">Add this snippet to your client’s website.</p>
              <pre className="snippet">{embedSnippet}</pre>
              <button className="secondary" onClick={copySnippet} disabled={!embedSnippet}>
                Copy snippet
              </button>
            </section>

            <section className="card">
              <div className="card-title">Domains</div>
              <div className="stack">
                <input
                  value={newDomain}
                  onChange={(event) => setNewDomain(event.target.value)}
                  placeholder="example.com"
                />
                <button className="secondary" onClick={addDomain} disabled={loading || !newDomain.trim()}>
                  Add domain
                </button>
              </div>
              <div className="domain-list">
                {domains.map((domain) => (
                  <div key={domain.hostname} className="domain-row">
                    <div>
                      <div className="domain-host">{domain.hostname}</div>
                      <div className={`pill ${domain.status}`}>{domain.status}</div>
                    </div>
                    <div className="domain-actions">
                      <button className="ghost" onClick={() => verifyDomain(domain.hostname)}>
                        Verify
                      </button>
                      <div className="token">Token: {domain.verification_token}</div>
                    </div>
                  </div>
                ))}
                {!domains.length && <div className="empty">No domains added yet.</div>}
              </div>
            </section>

            <section className="card">
              <div className="card-title">Crawl control</div>
              <div className="stack">
                <input
                  value={crawlUrl}
                  onChange={(event) => setCrawlUrl(event.target.value)}
                  placeholder="https://example.com"
                />
                <div className="row">
                  <button className="primary" onClick={startCrawl} disabled={loading || !crawlUrl.trim()}>
                    Start crawl
                  </button>
                  <button className="ghost" onClick={cancelCrawl} disabled={loading || !activeCrawlUrl}>
                    Cancel crawl
                  </button>
                </div>
              </div>
              {indexStatus && (
                <div className="status">
                  <div className="detail-row">
                    <span>Status</span>
                    <span>{indexStatus.stage || indexStatus.status}</span>
                  </div>
                  <div className="detail-row">
                    <span>Pages crawled</span>
                    <span>{indexStatus.pages_crawled ?? '-'}</span>
                  </div>
                  <div className="detail-row">
                    <span>Docs</span>
                    <span>{indexStatus.docs_count ?? '-'}</span>
                  </div>
                  {indexStatus.last_error && <div className="alert error">{indexStatus.last_error}</div>}
                </div>
              )}
            </section>

            <section className="card">
              <div className="card-title">Recent crawl jobs</div>
              <div className="job-list">
                {jobs.map((job) => (
                  <div key={job.job_id} className="job-row">
                    <div>
                      <div className="job-url">{job.url}</div>
                      <div className="muted">Stage: {job.stage}</div>
                    </div>
                    <div className="job-meta">
                      <span>{job.pages_crawled} pages</span>
                      <span>{job.docs_count} docs</span>
                    </div>
                  </div>
                ))}
                {!jobs.length && <div className="empty">No crawl jobs yet.</div>}
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  )
}
