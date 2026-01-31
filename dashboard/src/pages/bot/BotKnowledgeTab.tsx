import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotKnowledgeTab() {
  const {
    selectedBot,
    domains,
    newDomain,
    setNewDomain,
    addDomain,
    verifyDomain,
    crawlUrl,
    setCrawlUrl,
    startCrawl,
    cancelCrawl,
    activeCrawlUrl,
    indexStatus,
    loading,
  } = useDashboardData()

  if (!selectedBot) {
    return <div className="empty-panel">Select a bot to manage knowledge.</div>
  }

  return (
    <div className="card-grid">
      <section className="card">
        <div className="card-title">Domains</div>
        <div className="stack">
          <input value={newDomain} onChange={(event) => setNewDomain(event.target.value)} placeholder="example.com" />
          <button className="secondary" onClick={addDomain} disabled={loading || !newDomain.trim()}>
            Add domain
          </button>
        </div>
        <div className="list">
          {domains.map((domain) => (
            <div key={domain.hostname} className="list-row">
              <div>
                <div className="list-title">{domain.hostname}</div>
                <div className={`pill ${domain.status}`}>{domain.status}</div>
              </div>
              <div className="row">
                <button className="ghost" onClick={() => verifyDomain(domain.hostname)} disabled={loading}>
                  Verify
                </button>
                <div className="token">Token: {domain.verification_token}</div>
              </div>
            </div>
          ))}
          {!domains.length && <div className="empty">No domains added yet.</div>}
        </div>
      </section>

      <section className="card">
        <div className="card-title">Crawl control</div>
        <div className="stack">
          <input value={crawlUrl} onChange={(event) => setCrawlUrl(event.target.value)} placeholder="https://example.com" />
          <div className="row">
            <button className="primary" onClick={startCrawl} disabled={loading || !crawlUrl.trim()}>
              Start crawl
            </button>
            <button className="ghost" onClick={cancelCrawl} disabled={loading || !activeCrawlUrl}>
              Cancel crawl
            </button>
          </div>
        </div>
        {indexStatus && (
          <div className="status">
            <div className="detail-row">
              <span>Status</span>
              <span>{indexStatus.stage || indexStatus.status}</span>
            </div>
            <div className="detail-row">
              <span>Pages crawled</span>
              <span>{indexStatus.pages_crawled ?? '-'}</span>
            </div>
            <div className="detail-row">
              <span>Docs</span>
              <span>{indexStatus.docs_count ?? '-'}</span>
            </div>
            {indexStatus.last_error && <div className="alert error">{indexStatus.last_error}</div>}
          </div>
        )}
      </section>
    </div>
  )
}
