import { Link, useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'

function sourceTypeLabel(type: string): string {
  const t = (type || '').toLowerCase()
  if (t === 'url') return 'URL'
  if (t === 'drive') return 'Drive'
  if (t === 'docs') return 'Google Docs'
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
    <section className="card">
      <div className="card-title">Sources</div>
      <p className="card-subtitle" style={{ marginTop: 0, marginBottom: '1rem' }}>
        Every source (URL, Drive, Docs, etc.) this bot learns from.
      </p>
      <div style={{ marginBottom: '1rem' }}>
        <Link to={botId ? `/bots/${botId}/sources/new` : '#'} className="primary">
          + Add source
        </Link>
      </div>
      <div className="list">
        {sources.map((s) => (
          <div key={s.source_id} className="list-row">
            <div>
              <span className="source-type-badge" data-type={s.type.toLowerCase()} style={{ marginRight: '0.5rem' }}>
                {sourceTypeLabel(s.type)}
              </span>
              <div className="list-title" style={{ wordBreak: 'break-all' }}>{sourceSummary(s)}</div>
            </div>
          </div>
        ))}
        {!sources.length && <div className="empty">No sources yet. Add sources from the Knowledge tab.</div>}
      </div>
    </section>
  )
}
