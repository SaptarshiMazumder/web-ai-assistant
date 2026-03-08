import { MoreVertical, Pencil, Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, SectionHeader, StatusDot, UiButton } from '../components/ui'
import { useTranslation } from 'react-i18next'
import { useDialog } from '../contexts/DialogContext'

export default function BotsPage() {
  const { bots, loading, isSuperAdmin, activeOrgId, deleteBot, renameBot } = useDashboardData()
  const { t } = useTranslation()
  const dialog = useDialog()
  const navigate = useNavigate()
  const [selectedBotIds, setSelectedBotIds] = useState<Set<string>>(new Set())
  const [openMenuBotId, setOpenMenuBotId] = useState<string | null>(null)

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">{t('botsPage.selectOrgPrompt', 'Select an organization to view bots.')}</div>
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
    const ok = await dialog.confirm({
      title: t('botsPage.deleteMultipleConfirm', 'Delete {{count}} selected bot(s)? This cannot be undone.', { count: selectedCount }),
      confirmLabel: t('common.delete', 'Delete'),
      cancelLabel: t('common.cancel', 'Cancel'),
      tone: 'danger',
    })
    if (!ok) return
    const ids = Array.from(selectedBotIds)
    for (const botId of ids) {
      await deleteBot(botId)
    }
    setSelectedBotIds(new Set())
  }

  const handleRenameBot = async (botId: string, currentName: string) => {
    const nextName = await dialog.prompt({
      title: t('botsPage.enterNewBotName', 'Enter new bot name:'),
      label: t('botSettings.botName', 'Bot Name'),
      placeholder: t('botSettings.enterBotName', 'Enter bot name'),
      defaultValue: currentName,
      confirmLabel: t('botsPage.rename', 'Rename'),
      cancelLabel: t('common.cancel', 'Cancel'),
      required: true,
    })
    const trimmedName = nextName?.trim()
    if (!trimmedName || trimmedName === currentName) return
    await renameBot(botId, trimmedName)
    setOpenMenuBotId(null)
  }

  const handleDeleteOne = async (botId: string, name: string) => {
    const ok = await dialog.confirm({
      title: t('botsPage.deleteOneConfirm', 'Delete "{{name}}"? This cannot be undone.', { name }),
      confirmLabel: t('common.delete', 'Delete'),
      cancelLabel: t('common.cancel', 'Cancel'),
      tone: 'danger',
    })
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
      <PageHeader title={t('botsPage.title', 'Bots')} />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow={t('botsPage.agentsTab', 'Agents')}
          title={t('botsPage.buildScale', 'Build, launch, and scale your bot fleet')}
          subtitle={t('botsPage.buildScaleSubtitle', 'Every card is a live workspace with direct access to settings, analytics, and training.')}
        />

        {bots.length === 0 ? (
          <GlassCard>
            <EmptyState
              title={t('botsPage.noBotsYet', 'No bots yet')}
              description={t('botsPage.startWithOne', 'Start with one beautiful assistant and expand into a full AI team.')}
              action={
                <UiButton variant="primary" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
                  {t('botsPage.createFirstBot', 'Create your first bot')}
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
                {t('botsPage.selectAll', 'Select all')}
              </label>
              {selectedCount > 0 && (
                <UiButton variant="ghost" onClick={() => void handleDeleteSelected()} style={{ display: 'inline-flex', gap: '0.4rem' }}>
                  <Trash2 size={16} />
                  {t('botsPage.deleteSelected', 'Delete selected ({{count}})', { count: selectedCount })}
                </UiButton>
              )}
            </div>

            <div className="bot-card-grid">
              <button className="cta-create-bot ui-glass-card" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
                <div className="cta-create-bot-icon-wrap">
                  <Plus size={24} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{t('botsPage.addBot', 'Add Bot')}</div>
                  <div style={{ fontSize: '0.85rem', opacity: 0.9, marginTop: '2px' }}>{t('botsPage.buildNewAgent', 'Build a new AI agent')}</div>
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
                        {t('botsPage.rename', 'Rename')}
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
                        {t('botsPage.delete', 'Delete')}
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
