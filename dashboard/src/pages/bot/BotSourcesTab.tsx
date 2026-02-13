import { Link, useParams } from 'react-router-dom'
import { Database, Plus } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

function sourceTypeLabel(type: string): string {
  const t = (type || '').toLowerCase()
  if (t === 'url') return 'URL'
  if (t === 'drive') return 'Drive'
  if (t === 'docs') return 'Doc'
  return type || '—'
}

function sourceSummary(source: { type: string; config: Record<string, unknown>; display_name?: string | null }): string {
  if (source.display_name) return source.display_name
  if (source.type === 'url' && typeof source.config?.url === 'string') return source.config.url
  if (source.type === 'drive' && typeof source.config?.folder_id === 'string') return `Drive folder: ${source.config.folder_id}`
  if (source.type === 'docs' && typeof source.config?.doc_id === 'string') return `Doc: ${source.config.doc_id}`
  return source.type || '—'
}

export default function BotSourcesTab() {
  const { botId } = useParams()
  const { selectedBot, sources } = useDashboardData()

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to view sources.</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        eyebrow="Data"
        title="Training sources"
        subtitle="Every source (URL, Drive, Docs, etc.) this bot learns from."
        action={
          <Link to={botId ? `/bots/${botId}/sources/new` : '#'}>
            <UiButton variant="primary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
              <Plus size={16} />
              Add source
            </UiButton>
          </Link>
        }
      />

      <GlassCard>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Database size={16} style={{ color: 'var(--ui-flow-accent)' }} />
          Sources ({sources.length})
        </div>
        <div className="list">
          {sources.map((s) => (
            <div key={s.source_id} className="list-row">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <span className="source-type-badge" data-type={s.type.toLowerCase()}>
                  {sourceTypeLabel(s.type)}
                </span>
                <div className="list-title" style={{ wordBreak: 'break-all' }}>{sourceSummary(s)}</div>
              </div>
            </div>
          ))}
          {!sources.length && <div className="muted" style={{ padding: '1rem 0' }}>No sources yet. Add sources from the Knowledge tab or use the button above.</div>}
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
