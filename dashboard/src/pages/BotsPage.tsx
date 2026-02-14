import { MoreVertical, Pencil, Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, SectionHeader, StatusDot, UiButton } from '../components/ui'

export default function BotsPage() {
  const { bots, loading, isSuperAdmin, activeOrgId, deleteBot, renameBot } = useDashboardData()
  const navigate = useNavigate()
  const [selectedBotIds, setSelectedBotIds] = useState<Set<string>>(new Set())
  const [openMenuBotId, setOpenMenuBotId] = useState<string | null>(null)

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">Select an organization to view bots.</div>
  }

  if (!loading && bots.length === 0) {
    return <Navigate to="/create-bot" replace />
  }

  const canCreateBot = !isSuperAdmin || (activeOrgId && activeOrgId !== '__all__')
  const selectedCount = selectedBotIds.size
  const allSelected = useMemo(
    () => bots.length > 0 && bots.every((bot) => selectedBotIds.has(bot.bot_id)),
    [bots, selectedBotIds]
  )

  const toggleBotSelected = (botId: string) => {
    setSelectedBotIds((prev) => {
      const next = new Set(prev)
      if (next.has(botId)) next.delete(botId)
      else next.add(botId)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelectedBotIds(new Set())
      return
    }
    setSelectedBotIds(new Set(bots.map((b) => b.bot_id)))
  }

  const handleDeleteSelected = async () => {
    if (!selectedCount) return
    const ok = window.confirm(`Delete ${selectedCount} selected bot${selectedCount > 1 ? 's' : ''}? This cannot be undone.`)
    if (!ok) return
    const ids = Array.from(selectedBotIds)
    for (const botId of ids) {
      await deleteBot(botId)
    }
    setSelectedBotIds(new Set())
  }

  const handleRenameBot = async (botId: string, currentName: string) => {
    const nextName = window.prompt('Enter new bot name:', currentName)?.trim()
    if (!nextName || nextName === currentName) return
    await renameBot(botId, nextName)
    setOpenMenuBotId(null)
  }

  const handleDeleteOne = async (botId: string, name: string) => {
    const ok = window.confirm(`Delete "${name}"? This cannot be undone.`)
    if (!ok) return
    await deleteBot(botId)
    setSelectedBotIds((prev) => {
      const next = new Set(prev)
      next.delete(botId)
      return next
    })
    setOpenMenuBotId(null)
  }

  return (
    <AnimatedPage className="page">
      <PageHeader title="Bots" />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow="Agents"
          title="Build, launch, and scale your bot fleet"
          subtitle="Every card is a live workspace with direct access to settings, analytics, and training."
        />

        {bots.length === 0 ? (
          <GlassCard>
            <EmptyState
              title="No bots yet"
              description="Start with one beautiful assistant and expand into a full AI team."
              action={
                <UiButton variant="primary" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
                  Create your first bot
                </UiButton>
              }
            />
          </GlassCard>
        ) : (
          <>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '0.75rem',
                marginBottom: '0.75rem',
              }}
            >
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', color: 'var(--flow-muted)' }}>
                <input type="checkbox" checked={allSelected} onChange={toggleSelectAll} />
                Select all
              </label>
              {selectedCount > 0 && (
                <UiButton variant="ghost" onClick={() => void handleDeleteSelected()} style={{ display: 'inline-flex', gap: '0.4rem' }}>
                  <Trash2 size={16} />
                  Delete selected ({selectedCount})
                </UiButton>
              )}
            </div>

            <div className="bot-card-grid">
              <button className="cta-create-bot ui-glass-card" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
                <div className="cta-create-bot-icon-wrap">
                  <Plus size={24} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>Add Bot</div>
                  <div style={{ fontSize: '0.85rem', opacity: 0.9, marginTop: '2px' }}>Build a new AI agent</div>
                </div>
              </button>

              {bots.map((bot) => (
                <div
                  key={bot.bot_id}
                  className="ui-glass-card bot-card-modern"
                  role="button"
                  tabIndex={0}
                  onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') navigate(`/bots/${bot.bot_id}/overview`)
                  }}
                  style={{ textAlign: 'left', position: 'relative', cursor: 'pointer' }}
                >
                  <div className="bot-card-title-row">
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.55rem', minWidth: 0 }}>
                      <input
                        type="checkbox"
                        checked={selectedBotIds.has(bot.bot_id)}
                        onChange={() => toggleBotSelected(bot.bot_id)}
                        onClick={(e) => e.stopPropagation()}
                        aria-label={`Select ${bot.display_name}`}
                      />
                      <div className="list-title" style={{ minWidth: 0 }}>{bot.display_name}</div>
                    </div>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                      <StatusDot tone="success" />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setOpenMenuBotId((prev) => (prev === bot.bot_id ? null : bot.bot_id))
                        }}
                        aria-label={`Open options for ${bot.display_name}`}
                        className="bot-card-menu-trigger"
                      >
                        <MoreVertical size={16} />
                      </button>
                    </div>
                  </div>
                  <div className="bot-card-id">{bot.bot_id}</div>
                  {openMenuBotId === bot.bot_id && (
                    <div
                      onClick={(e) => e.stopPropagation()}
                      className="bot-card-actions-menu"
                    >
                      <button
                        type="button"
                        onClick={() => void handleRenameBot(bot.bot_id, bot.display_name)}
                        style={{
                          width: '100%',
                          border: 'none',
                          background: 'transparent',
                          textAlign: 'left',
                          cursor: 'pointer',
                          padding: '8px 10px',
                          borderRadius: 8,
                          display: 'inline-flex',
                          gap: '0.45rem',
                          alignItems: 'center',
                        }}
                      >
                        <Pencil size={14} />
                        Rename
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDeleteOne(bot.bot_id, bot.display_name)}
                        style={{
                          width: '100%',
                          border: 'none',
                          background: 'transparent',
                          textAlign: 'left',
                          cursor: 'pointer',
                          padding: '8px 10px',
                          borderRadius: 8,
                          color: '#dc2626',
                          display: 'inline-flex',
                          gap: '0.45rem',
                          alignItems: 'center',
                        }}
                      >
                        <Trash2 size={14} />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </AnimatedPage>
  )
}
