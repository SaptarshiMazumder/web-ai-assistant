import { useEffect, useMemo, useState } from 'react'

type BotSummary = {
  bot_id: string
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

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

const terminalStages = new Set(['done', 'error', 'cancelled', 'import_submitted'])

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const initHeaders = init?.headers
  const headerEntries =
    initHeaders instanceof Headers ? Object.fromEntries(initHeaders.entries()) : (initHeaders as Record<string, string> | undefined)
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(headerEntries || {}),
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
  const [adminKey, setAdminKey] = useState(() => {
    try {
      return window.localStorage.getItem('web-ai-admin-key') || ''
    } catch {
      return ''
    }
  })

  const embedSnippet = useMemo(() => {
    if (!selectedBot) return ''
    return `<script async src="${API_BASE}/widget/widget.js" data-bot-key="${selectedBot.publishable_key}" data-api-base="${API_BASE}"></script>`
  }, [selectedBot])

  async function loadBots() {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchJson<{ bots: BotSummary[] }>('/v1/bots', {
        headers: adminKey ? { 'X-Admin-Key': adminKey } : {},
      })
      setBots(data.bots)
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
    setLoading(true)
    setError(null)
    try {
      const data = await fetchJson<{ bot: BotSummary }>(`/v1/bots/${botId}`, {
        headers: adminKey ? { 'X-Admin-Key': adminKey } : {},
      })
      setSelectedBot(data.bot)
    } catch (err) {
      setError((err as Error).message)
      setSelectedBot(null)
    } finally {
      setLoading(false)
    }
  }

  async function loadDomains(botId: string) {
    try {
      const data = await fetchJson<{ bot_id: string; domains: DomainRecord[] }>(`/v1/bots/${botId}/domains`, {
        headers: adminKey ? { 'X-Admin-Key': adminKey } : {},
      })
      setDomains(data.domains)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadJobs(botId: string) {
    try {
      const data = await fetchJson<{ bot_id: string; jobs: JobRecord[] }>(`/v1/bots/${botId}/jobs`, {
        headers: adminKey ? { 'X-Admin-Key': adminKey } : {},
      })
      setJobs(data.jobs)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function createBot() {
    if (!newBotName.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchJson<BotCreateResponse>('/v1/bots', {
        method: 'POST',
        headers: adminKey ? { 'X-Admin-Key': adminKey } : {},
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

  async function clearGcs() {
    if (!adminKey) {
      setError('Admin key required to clear GCS data')
      return
    }
    if (!window.confirm('Delete all crawled content from GCS? This cannot be undone.')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await fetchJson('/v1/admin/reset/gcs', {
        method: 'POST',
        headers: { 'X-Admin-Key': adminKey },
      })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function clearRag() {
    if (!adminKey) {
      setError('Admin key required to clear RAG corpora')
      return
    }
    if (!window.confirm('Delete all RAG corpora? This cannot be undone.')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await fetchJson('/v1/admin/reset/rag', {
        method: 'POST',
        headers: { 'X-Admin-Key': adminKey },
      })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function addDomain() {
    if (!selectedBot || !newDomain.trim()) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/v1/bots/${selectedBot.bot_id}/domains`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${selectedBot.secret_key}` },
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
    if (!selectedBot) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/v1/bots/${selectedBot.bot_id}/domains/${hostname}/verify`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${selectedBot.secret_key}` },
      })
      await loadDomains(selectedBot.bot_id)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function startCrawl() {
    if (!selectedBot || !crawlUrl.trim()) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/v1/bots/${selectedBot.bot_id}/index`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${selectedBot.secret_key}` },
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
    if (!selectedBot || !activeCrawlUrl) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/v1/bots/${selectedBot.bot_id}/index/cancel`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${selectedBot.secret_key}` },
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
    if (!selectedBot) return
    try {
      const status = await fetchJson<IndexStatus>(
        `/v1/bots/${selectedBot.bot_id}/index/status?url=${encodeURIComponent(url)}`,
        { headers: { Authorization: `Bearer ${selectedBot.secret_key}` } }
      )
      setIndexStatus(status)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    void loadBots()
  }, [])

  useEffect(() => {
    try {
      if (adminKey) {
        window.localStorage.setItem('web-ai-admin-key', adminKey)
      } else {
        window.localStorage.removeItem('web-ai-admin-key')
      }
    } catch {
      // ignore storage errors
    }
  }, [adminKey])

  useEffect(() => {
    if (!selectedBotId) return
    void loadBotDetail(selectedBotId)
    void loadDomains(selectedBotId)
    void loadJobs(selectedBotId)
    setIndexStatus(null)
    setActiveCrawlUrl('')
  }, [selectedBotId])

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
          <div className="section-title">Admin access</div>
          <div className="stack">
            <input
              value={adminKey}
              onChange={(event) => setAdminKey(event.target.value)}
              placeholder="Admin key (X-Admin-Key)"
              type="password"
            />
          </div>
        </div>
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
          <button className="ghost" onClick={loadBots} disabled={loading}>
            Refresh
          </button>
        </header>

        {error && <div className="alert error">{error}</div>}
        {loading && <div className="alert">Working...</div>}

        {!selectedBot && <div className="empty-panel">Select a bot to view details.</div>}

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
