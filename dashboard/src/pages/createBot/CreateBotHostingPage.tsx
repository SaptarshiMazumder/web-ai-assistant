import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotHostingPage() {
  const navigate = useNavigate()
  const { step2, flow } = useCreateBotFlow()
  const {
    contentHosting,
    setContentHosting,
    websiteUrl,
    setWebsiteUrl,
    isDiscovering,
    localError,
    setLocalError,
    discoverUrls,
    stopDiscovery,
  } = step2

  const [continuing, setContinuing] = useState(false)

  const canContinue =
    !!contentHosting &&
    !(contentHosting === 'own' && !websiteUrl.trim()) &&
    !isDiscovering &&
    !continuing

  const handleContinue = async () => {
    if (!contentHosting || !flow.nextPath || !canContinue) return
    setContinuing(true)
    setLocalError(null)
    try {
      if (contentHosting === 'own') {
        const ok = await discoverUrls()
        if (!ok) return
      }
      navigate(flow.nextPath)
    } finally {
      setContinuing(false)
    }
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Do you have your own website?</div>
        <div className="card-subtitle">Pick what best describes your business.</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem' }}>
        <button
          type="button"
          onClick={() => setContentHosting('own')}
          style={{
            textAlign: 'left',
            padding: '1.25rem',
            borderRadius: '16px',
            border: `2px solid ${contentHosting === 'own' ? '#6366f1' : '#e2e8f0'}`,
            background: contentHosting === 'own' ? 'rgba(99,102,241,0.06)' : '#fff',
          }}
        >
          <div style={{ fontWeight: 600, fontSize: '1.05rem', color: '#0f172a', marginBottom: '0.25rem' }}>
            Yes — I have my own website
          </div>
        </button>

        <button
          type="button"
          onClick={() => setContentHosting('shared')}
          style={{
            textAlign: 'left',
            padding: '1.25rem',
            borderRadius: '16px',
            border: `2px solid ${contentHosting === 'shared' ? '#6366f1' : '#e2e8f0'}`,
            background: contentHosting === 'shared' ? 'rgba(99,102,241,0.06)' : '#fff',
          }}
        >
          <div style={{ fontWeight: 600, fontSize: '1.05rem', color: '#0f172a', marginBottom: '0.25rem' }}>
            Not really — I use a website service
          </div>
        </button>
      </div>

      {contentHosting === 'own' && (
        <div>
          <div className="card-title">Your website link</div>
          <div className="card-subtitle">Paste your main website address.</div>
          <input
            type="url"
            value={websiteUrl}
            onChange={(e) => setWebsiteUrl(e.target.value)}
            placeholder="https://yourwebsite.com"
            style={{ width: '100%' }}
          />
        </div>
      )}

      {localError && (
        <div className="alert error">
          <div style={{ whiteSpace: 'pre-line' }}>{localError}</div>
        </div>
      )}

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        {isDiscovering ? (
          <button type="button" className="primary" onClick={stopDiscovery}>
            Stop
          </button>
        ) : (
          <button type="button" className="primary" onClick={() => void handleContinue()} disabled={!canContinue}>
            {continuing ? 'Continuing…' : 'Continue'}
          </button>
        )}
      </div>
    </div>
  )
}

