import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export type BotSummary = {
  bot_id: string
  org_id?: string
  display_name: string
  publishable_key: string
  secret_key: string
  created_at: string
  updated_at: string
}

export type BotCreateResponse = {
  bot_id: string
  display_name: string
  publishable_key: string
  secret_key: string
}

export type DomainRecord = {
  bot_id: string
  hostname: string
  status: string
  verification_token: string
  verification_url?: string
  verified_at?: string | null
  created_at: string
  updated_at: string
}

export type JobRecord = {
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

export type IndexStatus = {
  status: string
  stage?: string
  pages_crawled?: number
  last_crawled_url?: string
  docs_count?: number
  last_error?: string
  gcs_prefix?: string
  updated_at?: string
}

export type OrgSummary = {
  org_id: string
  name: string
  status: string
  created_at: string
  updated_at: string
}

export type OrgMember = {
  user_id: string
  email: string
  first_name?: string | null
  last_name?: string | null
  role: string
  org_id?: string
  org_name?: string
  created_at: string
  updated_at: string
}

type TokenClaims = Record<string, unknown>

type DashboardData = {
  user:
    | {
        email?: string | null
        name?: string | null
        given_name?: string | null
        family_name?: string | null
        picture?: string | null
      }
    | undefined
  logout: (options?: { logoutParams?: { returnTo?: string } }) => void
  bots: BotSummary[]
  selectedBotId: string | null
  selectedBot: BotSummary | null
  domains: DomainRecord[]
  jobs: JobRecord[]
  indexStatus: IndexStatus | null
  loading: boolean
  error: string | null
  newBotName: string
  setNewBotName: (value: string) => void
  newDomain: string
  setNewDomain: (value: string) => void
  crawlUrl: string
  setCrawlUrl: (value: string) => void
  activeCrawlUrl: string
  orgs: OrgSummary[]
  activeOrgId: string | null
  setActiveOrgId: (value: string) => void
  orgMembers: OrgMember[]
  newOrgName: string
  setNewOrgName: (value: string) => void
  orgDisplayName: string
  orgDisplayNameInput: string
  setOrgDisplayNameInput: (value: string) => void
  newMemberEmail: string
  setNewMemberEmail: (value: string) => void
  newMemberRole: string
  setNewMemberRole: (value: string) => void
  isSuperAdmin: boolean
  embedSnippet: string
  loadBots: () => Promise<void>
  loadBotDetail: (botId: string) => Promise<void>
  loadDomains: (botId: string) => Promise<void>
  loadJobs: (botId: string) => Promise<void>
  createBot: () => Promise<BotCreateResponse | null>
  loadOrgs: () => Promise<void>
  loadSelfOrgs: () => Promise<void>
  loadOrgInfo: (orgId: string) => Promise<void>
  saveOrgName: () => Promise<void>
  createOrg: () => Promise<void>
  setOrgStatus: (orgId: string, status: 'active' | 'disabled') => Promise<void>
  loadOrgMembers: (orgId: string) => Promise<void>
  addOrgMember: () => Promise<void>
  clearGcs: () => Promise<void>
  clearRag: () => Promise<void>
  addDomain: () => Promise<void>
  verifyDomain: (hostname: string) => Promise<void>
  startCrawl: () => Promise<void>
  cancelCrawl: () => Promise<void>
  refreshStatus: (url: string) => Promise<void>
  copySnippet: () => Promise<void>
  refreshAll: () => void
  setSelectedBotId: (value: string | null) => void
}

const DashboardDataContext = createContext<DashboardData | undefined>(undefined)

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin
const ALL_ORGS_ID = "__all__"
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

export function DashboardDataProvider({ children }: { children: React.ReactNode }) {
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

  const { getAccessTokenSilently, getIdTokenClaims, user, logout, isAuthenticated } = useAuth0()

  const embedSnippet = useMemo(() => {
    if (!selectedBot) return ''
    return `<script async src="${API_BASE}/widget/widget.js" data-bot-key="${selectedBot.publishable_key}" data-api-base="${API_BASE}"></script>`
  }, [selectedBot])

  async function fetchAuthedJson<T>(path: string, init?: RequestInit): Promise<T> {
    const token = await getAccessTokenSilently()
    return fetchJson<T>(path, init, token)
  }

  function withOrgParam(path: string, orgIdOverride?: string | null) {
    const orgId = orgIdOverride ?? activeOrgId
    if (!orgId || orgId === ALL_ORGS_ID) return path
    const suffix = `org_id=${encodeURIComponent(orgId)}`
    return path.includes('?') ? `${path}&${suffix}` : `${path}?${suffix}`
  }

  async function loadBots() {
    if (isSuperAdmin && !activeOrgId) return
    setLoading(true)
    setError(null)
    try {
      if (isSuperAdmin && activeOrgId === ALL_ORGS_ID) {
        const orgIds = orgs.map((org) => org.org_id)
        const results = await Promise.all(
          orgIds.map(async (orgId) => {
            try {
              const data = await fetchAuthedJson<{ bots: BotSummary[] }>(withOrgParam('/v1/org/bots', orgId))
              return data.bots
            } catch {
              return []
            }
          })
        )
        const merged = results.flat()
        const unique = Array.from(new Map(merged.map((bot) => [bot.bot_id, bot])).values())
        setBots(unique)
      } else {
        const data = await fetchAuthedJson<{ bots: BotSummary[] }>(withOrgParam('/v1/org/bots'))
        setBots(data.bots)
        if (!activeOrgId && data.bots.length) {
          setActiveOrgId(data.bots[0].org_id || null)
        }
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
      const orgOverride = activeOrgId === ALL_ORGS_ID ? null : activeOrgId
      const data = await fetchAuthedJson<{ bot: BotSummary }>(withOrgParam(`/v1/org/bots/${botId}`, orgOverride))
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
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<{ bot_id: string; domains: DomainRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/domains`, orgOverride)
      )
      setDomains(data.domains)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function loadJobs(botId: string) {
    if (isSuperAdmin && !activeOrgId) return
    try {
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const data = await fetchAuthedJson<{ bot_id: string; jobs: JobRecord[] }>(
        withOrgParam(`/v1/org/bots/${botId}/jobs`, orgOverride)
      )
      setJobs(data.jobs)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function createBot(): Promise<BotCreateResponse | null> {
    if (!newBotName.trim()) return null
    if (isSuperAdmin && (!activeOrgId || activeOrgId === ALL_ORGS_ID)) {
      setError("Select an organization to create a bot")
      return null
    }
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAuthedJson<BotCreateResponse>(withOrgParam('/v1/org/bots'), {
        method: 'POST',
        body: JSON.stringify({ display_name: newBotName.trim() }),
      })
      setNewBotName('')
      await loadBots()
      return data
    } catch (err) {
      setError((err as Error).message)
      return null
    } finally {
      setLoading(false)
    }
  }

  async function loadOrgs() {
    if (!isSuperAdmin) return
    try {
      const data = await fetchAuthedJson<{ orgs: OrgSummary[] }>('/v1/admin/orgs')
      setOrgs(data.orgs)
      if (!activeOrgId) {
        setActiveOrgId(ALL_ORGS_ID)
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
      const data = await fetchAuthedJson<OrgSummary>(`/v1/org/name?org_id=${encodeURIComponent(activeOrgId)}`, {
        method: 'POST',
        body: JSON.stringify({ name: orgDisplayNameInput.trim() }),
      })
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
      if (isSuperAdmin && orgId === ALL_ORGS_ID) {
        let orgList = orgs
        if (!orgList.length) {
          const data = await fetchAuthedJson<{ orgs: OrgSummary[] }>('/v1/admin/orgs')
          orgList = data.orgs || []
          setOrgs(orgList)
        }
        if (!orgList.length) {
          setOrgMembers([])
          return
        }
        const responses = await Promise.all(
          orgList.map(async (org) => {
            try {
              const data = await fetchAuthedJson<{ org_id: string; members: OrgMember[] }>(`/v1/admin/orgs/${org.org_id}/members`)
              return (data.members || []).map((member) => ({
                ...member,
                org_id: org.org_id,
                org_name: org.name,
              }))
            } catch {
              return [] as OrgMember[]
            }
          })
        )
        const merged = responses.flat()
        setOrgMembers(merged)
        return
      }
      const path = isSuperAdmin ? `/v1/admin/orgs/${orgId}/members` : `/v1/org/members?org_id=${encodeURIComponent(orgId)}`
      const data = await fetchAuthedJson<{ org_id: string; members: OrgMember[] }>(path)
      if (isSuperAdmin) {
        const org = orgs.find((entry) => entry.org_id === orgId)
        setOrgMembers(
          (data.members || []).map((member) => ({
            ...member,
            org_id: orgId,
            org_name: org?.name,
          }))
        )
      } else {
        setOrgMembers(data.members)
      }
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
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/domains`, orgOverride), {
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
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/domains/${hostname}/verify`, orgOverride), {
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
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index`, orgOverride), {
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
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      await fetchAuthedJson(withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index/cancel`, orgOverride), {
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
      const orgOverride = selectedBot?.org_id && activeOrgId === ALL_ORGS_ID ? selectedBot.org_id : activeOrgId
      const path = withOrgParam(`/v1/org/bots/${selectedBot.bot_id}/index/status?url=${encodeURIComponent(url)}`, orgOverride)
      const status = await fetchAuthedJson<IndexStatus>(path)
      setIndexStatus(status)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function copySnippet() {
    if (!embedSnippet) return
    try {
      await navigator.clipboard.writeText(embedSnippet)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  function refreshAll() {
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
  }, [isAuthenticated, activeOrgId, isSuperAdmin, orgs.length])

  useEffect(() => {
    if (activeOrgId) {
      setError(null)
    }
  }, [activeOrgId])

  useEffect(() => {
    if (!activeOrgId) return
    setBots([])
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

  const value: DashboardData = {
    user,
    logout,
    bots,
    selectedBotId,
    selectedBot,
    domains,
    jobs,
    indexStatus,
    loading,
    error,
    newBotName,
    setNewBotName,
    newDomain,
    setNewDomain,
    crawlUrl,
    setCrawlUrl,
    activeCrawlUrl,
    orgs,
    activeOrgId,
    setActiveOrgId,
    orgMembers,
    newOrgName,
    setNewOrgName,
    orgDisplayName,
    orgDisplayNameInput,
    setOrgDisplayNameInput,
    newMemberEmail,
    setNewMemberEmail,
    newMemberRole,
    setNewMemberRole,
    isSuperAdmin,
    embedSnippet,
    loadBots,
    loadBotDetail,
    loadDomains,
    loadJobs,
    createBot,
    loadOrgs,
    loadSelfOrgs,
    loadOrgInfo,
    saveOrgName,
    createOrg,
    setOrgStatus,
    loadOrgMembers,
    addOrgMember,
    clearGcs,
    clearRag,
    addDomain,
    verifyDomain,
    startCrawl,
    cancelCrawl,
    refreshStatus,
    copySnippet,
    refreshAll,
    setSelectedBotId,
  }

  return React.createElement(DashboardDataContext.Provider, { value }, children)
}

export function useDashboardData() {
  const context = useContext(DashboardDataContext)
  if (!context) {
    throw new Error('useDashboardData must be used within DashboardDataProvider')
  }
  return context
}
