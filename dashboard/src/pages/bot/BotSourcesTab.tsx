import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotSourcesTab() {
  const { selectedBot, jobs } = useDashboardData()

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to view sources.</div>
  }

  return (
    <section className="card">
      <div className="card-title">Recent crawl jobs</div>
      <div className="list">
        {jobs.map((job) => (
          <div key={job.job_id} className="list-row">
            <div>
              <div className="list-title">{job.url}</div>
              <div className="muted">Stage: {job.stage}</div>
            </div>
            <div className="list-meta">
              <span>{job.pages_crawled} pages</span>
              <span>{job.docs_count} docs</span>
            </div>
          </div>
        ))}
        {!jobs.length && <div className="empty">No crawl jobs yet.</div>}
      </div>
    </section>
  )
}
