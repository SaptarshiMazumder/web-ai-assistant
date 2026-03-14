import { MoreVertical, Pencil, Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, SectionHeader, StatusDot, UiButton } from '../components/ui'
import { useTranslation } from 'react-i18next'
import { useDialog } from '../contexts/DialogContext'

const BOT_CARD_PALETTES = [
  'bot-workspace-card--palette-0',
  'bot-workspace-card--palette-1',
  'bot-workspace-card--palette-2',
  'bot-workspace-card--palette-3',
  'bot-workspace-card--palette-4',
  'bot-workspace-card--palette-5',
] as const

function stableHash(input: string): number {
  let hash = 0
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0
  }
  return hash
}

function paletteClassForBot(botId: string): string {
  return BOT_CARD_PALETTES[stableHash(botId) % BOT_CARD_PALETTES.length]
}

function shortenBotId(botId: string): string {
  if (botId.length <= 18) return botId
  return `${botId.slice(0, 10)}...${botId.slice(-6)}`
}

export default function BotsPage() {
  const { bots, loading, botsLoadedOnce, isSuperAdmin, activeOrgId, deleteBot, renameBot, loadBots } = useDashboardData()
  const { t, i18n } = useTranslation()
  const dialog = useDialog()
  const navigate = useNavigate()
  const [selectedBotIds, setSelectedBotIds] = useState<Set<string>>(new Set())
  const [openMenuBotId, setOpenMenuBotId] = useState<string | null>(null)

  if (isSuperAdmin && !activeOrgId) {
    return <div className="empty-panel">{t('botsPage.selectOrgPrompt', 'Select an organization to view bots.')}</div>
  }

  if (botsLoadedOnce && !loading && bots.length === 0) {
    return <Navigate to="/create-bot" replace />
  }

  const canCreateBot = !isSuperAdmin || (activeOrgId && activeOrgId !== '__all__')
  const selectedCount = selectedBotIds.size
  const createdDateFormatter = useMemo(
    () =>
      new Intl.DateTimeFormat(i18n.language || undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      }),
    [i18n.language]
  )
  const allSelected = useMemo(
    () => bots.length > 0 && bots.every((bot) => selectedBotIds.has(bot.bot_id)),
    [bots, selectedBotIds]
  )
  const latestBotId = useMemo(() => {
    if (bots.length === 0) return null
    let latest = bots[0]
    let latestTs = new Date(bots[0].created_at).getTime()
    for (const bot of bots.slice(1)) {
      const ts = new Date(bot.created_at).getTime()
      if (Number.isNaN(ts)) continue
      if (Number.isNaN(latestTs) || ts > latestTs) {
        latest = bot
        latestTs = ts
      }
    }
    return latest.bot_id
  }, [bots])

  const formatCreatedDate = (value: string) => {
    const parsed = new Date(value)
    if (Number.isNaN(parsed.getTime())) return t('common.notAvailable', 'Not available')
    return createdDateFormatter.format(parsed)
  }

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
    console.log('Starting bulk deletion of', ids.length, 'bots:', ids)
    for (const botId of ids) {
      const success = await deleteBot(botId, true)
      console.log('Deleted bot', botId, ':', success)
    }
    console.log('Deletion loop complete, calling loadBots()')
    await loadBots()
    console.log('loadBots() complete')
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
            <div className="bots-toolbar">
              <label className="bots-select-all">
                <input type="checkbox" checked={allSelected} onChange={toggleSelectAll} />
                {t('botsPage.selectAll', 'Select all')}
              </label>
              {selectedCount > 0 && (
                <UiButton variant="ghost" onClick={() => void handleDeleteSelected()} className="bots-delete-selected">
                  <Trash2 size={16} />
                  {t('botsPage.deleteSelected', 'Delete selected ({{count}})', { count: selectedCount })}
                </UiButton>
              )}
            </div>

            <div className="bot-card-grid">
              <button className="bot-create-card" onClick={() => navigate('/create-bot')} disabled={!canCreateBot || loading}>
                <div className="bot-create-card__icon" aria-hidden="true">
                  <Plus size={24} />
                </div>
                <div className="bot-create-card__content">
                  <div className="bot-create-card__title">{t('botsPage.addBot', 'Add Bot')}</div>
                  <div className="bot-create-card__subtitle">{t('botsPage.buildNewAgent', 'Build a new AI agent')}</div>
                </div>
              </button>

              {bots.map((bot) => (
                <div
                  key={bot.bot_id}
                  className={`bot-workspace-card ${bot.bot_id === latestBotId ? 'bot-workspace-card--latest' : paletteClassForBot(bot.bot_id)}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      navigate(`/bots/${bot.bot_id}/overview`)
                    }
                  }}
                >
                  <div className="bot-workspace-card__header">
                    <div className="bot-workspace-card__selection">
                      <input
                        type="checkbox"
                        checked={selectedBotIds.has(bot.bot_id)}
                        onChange={() => toggleBotSelected(bot.bot_id)}
                        onClick={(e) => e.stopPropagation()}
                        aria-label={`Select ${bot.display_name}`}
                      />
                    </div>
                    <div className="bot-workspace-card__header-actions">
                      <span className="bot-workspace-card__status-pill">
                        <StatusDot tone="success" />
                        {t('botsPage.liveStatus', 'Live')}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setOpenMenuBotId((prev) => (prev === bot.bot_id ? null : bot.bot_id))
                        }}
                        aria-label={t('botsPage.openActions', 'Open options for {{name}}', { name: bot.display_name })}
                        className="bot-card-menu-trigger"
                      >
                        <MoreVertical size={16} />
                      </button>
                    </div>
                  </div>
                  <div className="bot-workspace-card__title" title={bot.display_name}>{bot.display_name}</div>
                  <div className="bot-workspace-card__meta">
                    <div className="bot-workspace-card__meta-row">
                      <span className="bot-workspace-card__meta-label">{t('botsPage.botIdLabel', 'Bot ID')}</span>
                      <span className="bot-workspace-card__meta-value">{shortenBotId(bot.bot_id)}</span>
                    </div>
                    <div className="bot-workspace-card__meta-row">
                      <span className="bot-workspace-card__meta-label">{t('botsPage.createdLabel', 'Created')}</span>
                      <span className="bot-workspace-card__meta-value">{formatCreatedDate(bot.created_at)}</span>
                    </div>
                  </div>
                  <div className="bot-workspace-card__actions" onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      className="bot-workspace-card__pill"
                      onClick={() => navigate(`/bots/${bot.bot_id}/overview`)}
                    >
                      {t('botsPage.quickOverview', 'Overview')}
                    </button>
                    <button
                      type="button"
                      className="bot-workspace-card__pill"
                      onClick={() => navigate(`/bots/${bot.bot_id}/knowledge`)}
                    >
                      {t('botsPage.quickSources', 'Sources')}
                    </button>
                  </div>
                  {openMenuBotId === bot.bot_id && (
                    <div onClick={(e) => e.stopPropagation()} className="bot-card-actions-menu">
                      <button
                        type="button"
                        onClick={() => void handleRenameBot(bot.bot_id, bot.display_name)}
                        className="bot-card-action-item"
                      >
                        <Pencil size={14} />
                        {t('botsPage.rename', 'Rename')}
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDeleteOne(bot.bot_id, bot.display_name)}
                        className="bot-card-action-item bot-card-action-item--danger"
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
