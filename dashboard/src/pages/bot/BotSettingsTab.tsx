import { Trash2, AlertTriangle, ShieldAlert, Edit2, Check, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

export default function BotSettingsTab() {
  const { t } = useTranslation()
  const { selectedBot, deleteBot, renameBot, saveWidgetConfig, selectedBotWidgetConfig, loading, error } = useDashboardData()
  const navigate = useNavigate()

  const [isEditing, setIsEditing] = useState(false)
  const [botName, setBotName] = useState(selectedBot?.display_name || '')
  const [saveSuccess, setSaveSuccess] = useState(false)

  // Update local state when selectedBot changes
  useEffect(() => {
    if (selectedBot && !isEditing) {
      setBotName(selectedBot.display_name)
    }
  }, [selectedBot, isEditing])

  const handleDelete = async () => {
    if (!selectedBot) return

    const confirmed = window.confirm(
      t('botSettings.deleteConfirm', 'Are you sure you want to delete "{{name}}"? This action cannot be undone and will delete all associated data including domains, knowledge base, and crawl jobs.', { name: selectedBot.display_name })
    )

    if (!confirmed) return

    const success = await deleteBot(selectedBot.bot_id)
    if (success) {
      navigate('/bots')
    }
  }

  const handleSave = async () => {
    if (!selectedBot || !botName.trim()) return

    const trimmed = botName.trim()
    const success = await renameBot(selectedBot.bot_id, trimmed)
    if (success) {
      // Sync widget title to match new bot name
      const existingConfig = selectedBotWidgetConfig ?? {}
      await saveWidgetConfig(selectedBot.bot_id, { ...existingConfig, title: trimmed })
      setIsEditing(false)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    }
  }

  const handleCancel = () => {
    setBotName(selectedBot?.display_name || '')
    setIsEditing(false)
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow={t('botSettings.configurationEyebrow', 'Configuration')}
        title={t('botSettings.title', 'Bot settings')}
        subtitle={t('botSettings.configurationSubtitle', 'Manage your bot configuration and take destructive actions when needed.')}
      />

      {/* Bot Info Section */}
      <GlassCard style={{ marginBottom: '1.5rem' }}>
        <div className="card-title" style={{ marginBottom: '1.25rem' }}>
          {t('botSettings.botInfo', 'Bot Information')}
        </div>

        {!isEditing ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--ui-flow-muted)', marginBottom: '0.35rem' }}>
                {t('botSettings.botName', 'Bot Name')}
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                {selectedBot?.display_name || t('botSettings.unnamedBot', 'Unnamed Bot')}
              </div>
            </div>
            <UiButton
              variant="secondary"
              onClick={() => setIsEditing(true)}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              disabled={loading}
            >
              <Edit2 size={16} />
              {t('botSettings.rename', 'Rename')}
            </UiButton>
          </div>
        ) : (
          <div>
            <div className="flow-field">
              <label className="flow-field-label">
                {t('botSettings.botName', 'Bot Name')}
              </label>
              <div className="flow-field-input-wrap">
                <input
                  type="text"
                  value={botName}
                  onChange={(e) => setBotName(e.target.value)}
                  placeholder={t('botSettings.enterBotName', 'Enter bot name')}
                  autoFocus
                  onBlur={(e) => {
                    // Prevent accidental cancel if clicking buttons
                    if (e.relatedTarget instanceof HTMLButtonElement) return;
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSave()
                    if (e.key === 'Escape') handleCancel()
                  }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
              <UiButton
                variant="primary"
                onClick={handleSave}
                disabled={loading || !botName.trim() || botName.trim() === selectedBot?.display_name}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <Check size={16} />
                {loading ? t('botSettings.saving', 'Saving...') : t('botSettings.save', 'Save')}
              </UiButton>
              <UiButton
                variant="secondary"
                onClick={handleCancel}
                disabled={loading}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <X size={16} />
                {t('botSettings.cancel', 'Cancel')}
              </UiButton>
            </div>
          </div>
        )}

        {saveSuccess && !isEditing && (
          <div style={{ marginTop: '0.75rem', color: '#10b981', fontSize: '0.9rem', fontWeight: 500 }}>
            {t('botSettings.botNameUpdated', '✓ Bot name updated successfully')}
          </div>
        )}

        {error && isEditing && (
          <div style={{ marginTop: '0.75rem', color: '#ef4444', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <AlertTriangle size={15} />
            {error}
          </div>
        )}
      </GlassCard>

      <GlassCard>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444' }}>
          <ShieldAlert size={18} />
          {t('botSettings.dangerZone', 'Danger zone')}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <p style={{ margin: '0 0 8px 0', color: 'var(--ui-flow-muted)' }}>
              {t('botSettings.deleteDescription', 'Permanently delete this bot. This will remove all associated data including:')}
            </p>
            <ul style={{ margin: '0 0 16px 0', paddingLeft: '20px', color: 'var(--ui-flow-muted)', lineHeight: 1.8 }}>
              <li>{t('botSettings.deleteListConfig', 'Bot configuration')}</li>
              <li>{t('botSettings.deleteListDomains', 'All domains and verifications')}</li>
              <li>{t('botSettings.deleteListKnowledge', 'Knowledge base and crawled content')}</li>
              <li>{t('botSettings.deleteListHistory', 'All crawl jobs and history')}</li>
            </ul>
          </div>
          <UiButton
            variant="primary"
            className="danger"
            onClick={handleDelete}
            disabled={loading || !selectedBot}
            style={{ alignSelf: 'flex-start', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
          >
            <Trash2 size={16} />
            {t('botSettings.deleteBot', 'Delete Bot')}
          </UiButton>
          {error && !isEditing && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#ef4444', fontSize: '0.9rem' }}>
              <AlertTriangle size={15} />
              {error}
            </div>
          )}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
