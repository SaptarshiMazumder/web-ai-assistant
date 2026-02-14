import { Trash2, AlertTriangle, ShieldAlert, Edit2, Check, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

export default function BotSettingsTab() {
  const { selectedBot, deleteBot, renameBot, loading, error } = useDashboardData()
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
      `Are you sure you want to delete "${selectedBot.display_name}"? This action cannot be undone and will delete all associated data including domains, knowledge base, and crawl jobs.`
    )

    if (!confirmed) return

    const success = await deleteBot(selectedBot.bot_id)
    if (success) {
      navigate('/bots')
    }
  }

  const handleSave = async () => {
    if (!selectedBot || !botName.trim()) return

    const success = await renameBot(selectedBot.bot_id, botName.trim())
    if (success) {
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
        eyebrow="Configuration"
        title="Bot settings"
        subtitle="Manage your bot configuration and take destructive actions when needed."
      />

      {/* Bot Info Section */}
      <GlassCard style={{ marginBottom: '1.5rem' }}>
        <div className="card-title" style={{ marginBottom: '1.25rem' }}>
          Bot Information
        </div>

        {!isEditing ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--ui-flow-muted)', marginBottom: '0.35rem' }}>
                Bot Name
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                {selectedBot?.display_name || 'Unnamed Bot'}
              </div>
            </div>
            <UiButton
              variant="secondary"
              onClick={() => setIsEditing(true)}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              disabled={loading}
            >
              <Edit2 size={16} />
              Rename
            </UiButton>
          </div>
        ) : (
          <div>
            <div className="flow-field">
              <label className="flow-field-label">
                Bot Name
              </label>
              <div className="flow-field-input-wrap">
                <input
                  type="text"
                  value={botName}
                  onChange={(e) => setBotName(e.target.value)}
                  placeholder="Enter bot name"
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
                {loading ? 'Saving...' : 'Save'}
              </UiButton>
              <UiButton
                variant="secondary"
                onClick={handleCancel}
                disabled={loading}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <X size={16} />
                Cancel
              </UiButton>
            </div>
          </div>
        )}

        {saveSuccess && !isEditing && (
          <div style={{ marginTop: '0.75rem', color: '#10b981', fontSize: '0.9rem', fontWeight: 500 }}>
            ✓ Bot name updated successfully
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
          Danger zone
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <p style={{ margin: '0 0 8px 0', color: 'var(--ui-flow-muted)' }}>
              Permanently delete this bot. This will remove all associated data including:
            </p>
            <ul style={{ margin: '0 0 16px 0', paddingLeft: '20px', color: 'var(--ui-flow-muted)', lineHeight: 1.8 }}>
              <li>Bot configuration</li>
              <li>All domains and verifications</li>
              <li>Knowledge base and crawled content</li>
              <li>All crawl jobs and history</li>
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
            Delete Bot
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
