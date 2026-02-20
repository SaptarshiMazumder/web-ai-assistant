import { useCallback, useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import { Check, Search, X, Plus, Trash2, Globe } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassField, SectionHeader, UiButton } from '../../components/ui'

const API_BASE = (import.meta as { env: Record<string, string> }).env.VITE_API_BASE || window.location.origin

type Persona = {
  id: string
  name: string
  category: string
  description: string
  emoji: string
  system_prompt: string
  is_custom?: boolean
}

type AgentConfig = {
  model_id?: string | null
  instructions?: string | null
  temperature?: number | null
  persona_id?: string | null
  custom_personas?: Persona[]
}

function withOrg(path: string, orgId: string | null): string {
  if (!orgId || orgId === '__all__') return path
  const suffix = `org_id=${encodeURIComponent(orgId)}`
  return path.includes('?') ? `${path}&${suffix}` : `${path}?${suffix}`
}

const ALL_TAB = 'All'
const CUSTOM_CATEGORY = 'Custom'
const DEFAULT_PERSONA_ID = 'default-assistant'
const CUSTOM_PERSONA_EMOJI = '💬'

export default function BotPersonaTab() {
  const { botId } = useParams()
  const { getAccessTokenSilently } = useAuth0()
  const { selectedBot, activeOrgId } = useDashboardData()

  const [builtinPersonas, setBuiltinPersonas] = useState<Persona[]>([])
  const [customPersonas, setCustomPersonas] = useState<Persona[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [activeCategory, setActiveCategory] = useState(ALL_TAB)
  const [selectedPersonaId, setSelectedPersonaId] = useState<string>(DEFAULT_PERSONA_ID)
  const [savedPersonaId, setSavedPersonaId] = useState<string>(DEFAULT_PERSONA_ID)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [applyingPersonaId, setApplyingPersonaId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  // Preview modal state (click-to-open, centered)
  const [previewPersona, setPreviewPersona] = useState<Persona | null>(null)

  // Custom persona creation modal
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDescription, setNewDescription] = useState('')
  const [newSystemPrompt, setNewSystemPrompt] = useState('')
  const [creatingSaving, setCreatingSaving] = useState(false)
  const [generatingPrompt, setGeneratingPrompt] = useState(false)

  const categoryTabsRef = useRef<HTMLDivElement>(null)

  // All personas merged
  const personas = [...customPersonas.map((p) => ({ ...p, is_custom: true })), ...builtinPersonas]

  // Load personas catalog
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/v1/personas`)
        if (!res.ok) throw new Error(res.statusText)
        const data = await res.json() as { personas: Persona[]; categories: string[] }
        if (!cancelled) {
          setBuiltinPersonas(data.personas)
          setCategories([ALL_TAB, ...data.categories])
        }
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [])

  // Load current agent config to get saved persona_id + custom personas
  useEffect(() => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    let cancelled = false
    void (async () => {
      try {
        const token = await getAccessTokenSilently()
        const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
        const res = await fetch(`${API_BASE}${path}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (!res.ok) throw new Error(res.statusText)
        const data = await res.json() as AgentConfig
        if (!cancelled) {
          const pid = (data.persona_id || DEFAULT_PERSONA_ID).trim()
          setSelectedPersonaId(pid)
          setSavedPersonaId(pid)
          setCustomPersonas((data.custom_personas || []).map((p) => ({ ...p, category: CUSTOM_CATEGORY })))
        }
      } catch {
        // ignore
      }
    })()
    return () => { cancelled = true }
  }, [botId, activeOrgId, getAccessTokenSilently])

  // Add "Custom" to categories when custom personas exist
  const allCategories = customPersonas.length > 0 && !categories.includes(CUSTOM_CATEGORY)
    ? [ALL_TAB, CUSTOM_CATEGORY, ...categories.filter((c) => c !== ALL_TAB)]
    : categories

  useEffect(() => {
    if (!personas.length) return
    if (!personas.some((p) => p.id === selectedPersonaId)) {
      setSelectedPersonaId(DEFAULT_PERSONA_ID)
    }
    if (!personas.some((p) => p.id === savedPersonaId)) {
      setSavedPersonaId(DEFAULT_PERSONA_ID)
    }
  }, [personas, selectedPersonaId, savedPersonaId])

  const handleSelectPersona = useCallback((personaId: string) => {
    setSelectedPersonaId(personaId)
  }, [])

  const handleSave = useCallback(async (personaIdToSave?: string) => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    const nextPersonaId = (personaIdToSave || selectedPersonaId || DEFAULT_PERSONA_ID).trim()
    if (!nextPersonaId) return
    setSelectedPersonaId(nextPersonaId)
    setSaving(true)
    setApplyingPersonaId(nextPersonaId)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)

      const getRes = await fetch(`${API_BASE}${path}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const existing = getRes.ok ? (await getRes.json() as AgentConfig) : {}

      const payload: AgentConfig = {
        model_id: existing.model_id || undefined,
        temperature: existing.temperature ?? undefined,
        persona_id: nextPersonaId,
        custom_personas: customPersonas.length > 0 ? customPersonas : undefined,
      }

      const res = await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error('Failed to save')
      setSavedPersonaId(nextPersonaId)
    } catch {
      // ignore
    } finally {
      setApplyingPersonaId(null)
      setSaving(false)
    }
  }, [botId, activeOrgId, selectedPersonaId, customPersonas, getAccessTokenSilently])

  // Create custom persona
  const handleCreateCustom = async () => {
    if (!newName.trim() || !newSystemPrompt.trim()) return
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    setCreatingSaving(true)
    try {
      const customId = `custom-${Date.now()}`
      const newPersona: Persona = {
        id: customId,
        name: newName.trim(),
        emoji: CUSTOM_PERSONA_EMOJI,
        description: newDescription.trim() || `Custom persona: ${newName.trim()}`,
        system_prompt: newSystemPrompt.trim(),
        category: CUSTOM_CATEGORY,
      }
      const updatedCustom = [...customPersonas, newPersona]

      // Save to agent config
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
      const getRes = await fetch(`${API_BASE}${path}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const existing = getRes.ok ? (await getRes.json() as AgentConfig) : {}

      await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          ...existing,
          custom_personas: updatedCustom,
        }),
      })

      setCustomPersonas(updatedCustom)
      setShowCreateModal(false)
      setNewName('')
      setNewDescription('')
      setNewSystemPrompt('')
    } catch {
      // ignore
    } finally {
      setCreatingSaving(false)
    }
  }

  // Generate system prompt from website using RAG, then auto-create + auto-apply the persona
  const handleGenerateFromWebsite = async () => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    setGeneratingPrompt(true)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/generate-default-prompt`, activeOrgId)
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to generate')
      const data = await res.json() as { prompt: string; business_name: string }
      if (!data.prompt) return

      // Auto-create custom persona (fixed ID so re-generating replaces it)
      const customId = `custom-website-${botId}`
      const personaName = data.business_name ? `${data.business_name} Assistant` : 'Website Assistant'
      const newPersona: Persona = {
        id: customId,
        name: personaName,
        emoji: CUSTOM_PERSONA_EMOJI,
        description: `Auto-generated from ${data.business_name || 'website'} trained content.`,
        system_prompt: data.prompt,
        category: CUSTOM_CATEGORY,
      }

      // Replace any previous auto-generated website persona, keep other custom ones
      const updatedCustom = [newPersona, ...customPersonas.filter((p) => p.id !== customId)]

      // Save custom personas + apply the new one in a single agent-config update
      const configPath = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
      const getRes = await fetch(`${API_BASE}${configPath}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const existing = getRes.ok ? (await getRes.json() as AgentConfig) : {}

      await fetch(`${API_BASE}${configPath}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ ...existing, custom_personas: updatedCustom, persona_id: customId }),
      })

      setCustomPersonas(updatedCustom)
      setSelectedPersonaId(customId)
      setSavedPersonaId(customId)

      // Pre-fill modal fields too in case user wants to edit before creating another
      setNewSystemPrompt(data.prompt)
      if (!newName.trim()) setNewName(personaName)
    } catch {
      // ignore
    } finally {
      setGeneratingPrompt(false)
    }
  }

  // Delete custom persona
  const handleDeleteCustom = async (personaId: string) => {
    if (!botId || !activeOrgId || activeOrgId === '__all__') return
    const updatedCustom = customPersonas.filter((p) => p.id !== personaId)
    try {
      const token = await getAccessTokenSilently()
      const path = withOrg(`/v1/org/bots/${botId}/agent-config`, activeOrgId)
      const getRes = await fetch(`${API_BASE}${path}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const existing = getRes.ok ? (await getRes.json() as AgentConfig) : {}

      await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          ...existing,
          custom_personas: updatedCustom.length > 0 ? updatedCustom : undefined,
          persona_id: existing.persona_id === personaId ? DEFAULT_PERSONA_ID : (existing.persona_id || DEFAULT_PERSONA_ID),
        }),
      })

      setCustomPersonas(updatedCustom)
      if (selectedPersonaId === personaId) {
        setSelectedPersonaId(DEFAULT_PERSONA_ID)
        setSavedPersonaId(DEFAULT_PERSONA_ID)
      }
    } catch {
      // ignore
    }
  }

  // Filter personas by category and search
  const filteredPersonas = personas.filter((p) => {
    const matchesCategory = activeCategory === ALL_TAB || p.category === activeCategory
    const matchesSearch =
      !searchQuery.trim() ||
      p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.category.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  if (!botId || !selectedBot) {
    return <div className="empty-panel">Select a bot to configure personas.</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow="Personality"
        title="Choose a persona for your AI"
        subtitle="Select a personality that defines how your agent communicates. This shapes the tone, style, and character of every response."
      />

      {/* Search + Actions row */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', alignItems: 'stretch' }}>
        <div className="persona-search-bar" style={{ flex: 1, marginBottom: 0 }}>
          <Search size={16} className="persona-search-icon" />
          <input
            type="text"
            className="persona-search-input"
            placeholder="Search personas..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button className="persona-search-clear" onClick={() => setSearchQuery('')}>
              <X size={14} />
            </button>
          )}
        </div>
        <UiButton
          variant="ghost"
          onClick={() => void handleGenerateFromWebsite()}
          disabled={generatingPrompt}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', whiteSpace: 'nowrap', flexShrink: 0 }}
        >
          <Globe size={16} />
          {generatingPrompt ? 'Generating...' : 'Generate from website'}
        </UiButton>
        <UiButton
          variant="primary"
          onClick={() => setShowCreateModal(true)}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', whiteSpace: 'nowrap', flexShrink: 0 }}
        >
          <Plus size={16} />
          Create Custom
        </UiButton>
      </div>

      {/* Category tabs (scrollable) */}
      <div className="persona-category-tabs" ref={categoryTabsRef}>
        {allCategories.map((cat) => (
          <button
            key={cat}
            className={`persona-category-tab ${activeCategory === cat ? 'is-active' : ''}`}
            onClick={() => setActiveCategory(cat)}
          >
            {cat}
            <span className="persona-category-count">
              {cat === ALL_TAB
                ? personas.length
                : personas.filter((p) => p.category === cat).length}
            </span>
          </button>
        ))}
      </div>

      {/* Persona grid */}
      {loading ? (
        <div className="persona-loading">Loading personas...</div>
      ) : (
        <div className="persona-grid">
          {filteredPersonas.map((p) => {
            const isSelected = selectedPersonaId === p.id
            const isSaved = savedPersonaId === p.id
            const isCustom = p.is_custom
            return (
              <div
                key={p.id}
                className={`persona-card ${isSelected ? 'persona-card--selected' : ''} ${isSaved ? 'persona-card--saved' : ''}`}
                onClick={() => handleSelectPersona(p.id)}
              >
                <div className="persona-card-header">
                  <span className="persona-card-emoji">{p.emoji}</span>
                  <div className="persona-card-meta">
                    <h4 className="persona-card-name">{p.name}</h4>
                    <span className="persona-card-category">
                      {p.category}
                      {isCustom && <span className="persona-custom-badge">Custom</span>}
                    </span>
                  </div>
                  {isSelected && (
                    <div className="persona-card-check">
                      <Check size={16} />
                    </div>
                  )}
                </div>
                <p className="persona-card-desc">{p.description}</p>
                <div className="persona-card-footer">
                  <div className="persona-card-footer-actions">
                    <button
                      className="persona-card-preview-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        setPreviewPersona(p)
                      }}
                    >
                      Preview prompt
                    </button>
                    <button
                      type="button"
                      className={`persona-card-apply-btn ${isSaved ? 'is-active' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        void handleSave(p.id)
                      }}
                      disabled={saving}
                    >
                      {saving && applyingPersonaId === p.id ? 'Applying...' : isSaved ? 'Applied' : 'Apply persona'}
                    </button>
                  </div>
                  {isCustom && (
                    <button
                      className="persona-card-delete-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        void handleDeleteCustom(p.id)
                      }}
                      title="Delete custom persona"
                    >
                      <Trash2 size={13} />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {filteredPersonas.length === 0 && !loading && (
        <div className="persona-empty">
          No personas match your search. Try a different term or category.
        </div>
      )}

      {/* Preview modal (centered, click-to-open) */}
      {previewPersona && (
        <div className="persona-modal-overlay" onClick={() => setPreviewPersona(null)}>
          <div className="persona-modal" onClick={(e) => e.stopPropagation()}>
            <div className="persona-preview-header">
              <span className="persona-preview-emoji">{previewPersona.emoji}</span>
              <div>
                <h4 className="persona-preview-name">{previewPersona.name}</h4>
                <span className="persona-preview-cat">{previewPersona.category}</span>
              </div>
              <button className="persona-preview-close" onClick={() => setPreviewPersona(null)}>
                <X size={16} />
              </button>
            </div>
            <p className="persona-preview-desc">{previewPersona.description}</p>
            <div className="persona-preview-prompt-label">System Prompt</div>
            <div className="persona-preview-prompt">{previewPersona.system_prompt}</div>
            <div className="persona-preview-actions">
              <UiButton
                variant="primary"
                onClick={() => {
                  handleSelectPersona(previewPersona.id)
                  setPreviewPersona(null)
                }}
              >
                Use this persona
              </UiButton>
            </div>
          </div>
        </div>
      )}

      {/* Create custom persona modal */}
      {showCreateModal && (
        <div className="persona-modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="persona-modal persona-modal--create" onClick={(e) => e.stopPropagation()}>
            <div className="persona-preview-header">
              <div>
                <h4 className="persona-preview-name">Create Custom Persona</h4>
                <span className="persona-preview-cat">Design your own personality</span>
              </div>
              <button className="persona-preview-close" onClick={() => setShowCreateModal(false)}>
                <X size={16} />
              </button>
            </div>

            <div className="persona-create-form">
              <GlassField label="Name" className="persona-create-field" style={{ maxWidth: '100%' }}>
                <input
                  type="text"
                  placeholder="e.g. Friendly Sales Rep"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  maxLength={60}
                />
              </GlassField>

              <GlassField label="Description (optional)" className="persona-create-field" style={{ maxWidth: '100%' }}>
                <input
                  type="text"
                  placeholder="Short description of this persona"
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  maxLength={200}
                />
              </GlassField>

              <GlassField label="System Prompt" className="persona-create-field" style={{ maxWidth: '100%' }}>
                <textarea
                  placeholder="Write the system prompt that defines this persona's behavior, tone, and style..."
                  value={newSystemPrompt}
                  onChange={(e) => setNewSystemPrompt(e.target.value)}
                  rows={6}
                />
              </GlassField>
            </div>

            <div className="persona-preview-actions">
              <UiButton variant="ghost" onClick={() => setShowCreateModal(false)}>
                Cancel
              </UiButton>
              <UiButton
                variant="primary"
                onClick={() => void handleCreateCustom()}
                disabled={creatingSaving || !newName.trim() || !newSystemPrompt.trim()}
              >
                {creatingSaving ? 'Creating...' : (
                  <><Plus size={16} /> Create Persona</>
                )}
              </UiButton>
            </div>
          </div>
        </div>
      )}
    </AnimatedPage>
  )
}
