import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon, StopIcon } from './DiscoveryIcons'

export default function CreateBotDetailsPage() {
  const navigate = useNavigate()
  const { step1, flow } = useCreateBotFlow()
  const { botName, setBotName, websiteUrl, setWebsiteUrl, discoveryMethod, setDiscoveryMethod, isDiscovering, localError, setLocalError, discoverUrls, stopDiscovery } = step1
  const [starting, setStarting] = useState(false)

  const handleContinue = async () => {
    if (starting || isDiscovering) return
    setStarting(true)
    try {
      const ok = await discoverUrls()
      if (ok && flow.nextPath) {
        navigate(flow.nextPath)
      }
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Name your bot</div>
        <div className="card-subtitle">Give your chatbot a friendly, customer-facing name.</div>
        <input value={botName} onChange={(event) => setBotName(event.target.value)} placeholder="Web AI Assistant" />
      </div>

      <div>
        <div className="card-title">Website to learn from</div>
        <div className="card-subtitle">We will scan this site and suggest pages to include.</div>
        <input value={websiteUrl} onChange={(event) => setWebsiteUrl(event.target.value)} placeholder="https://yourwebsite.com" />
      </div>

      <div>
        <div className="card-title">Discovery Method</div>
        <div className="card-subtitle">Choose how to discover URLs from the website.</div>
        <div className="design-form-radio-group" style={{ marginTop: '8px' }}>
          <label className="design-form-radio-card">
            <input
              type="radio"
              name="discoveryMethod"
              value="auto"
              checked={discoveryMethod === 'auto'}
              onChange={(e) => setDiscoveryMethod(e.target.value)}
            />
            <span>Automatic (Recommended)</span>
          </label>
          <label className="design-form-radio-card">
            <input
              type="radio"
              name="discoveryMethod"
              value="sitemap"
              checked={discoveryMethod === 'sitemap'}
              onChange={(e) => setDiscoveryMethod(e.target.value)}
            />
            <span>Sitemap Only</span>
          </label>
        </div>
        <div style={{ marginTop: '8px', fontSize: '14px', color: '#666' }}>
          {discoveryMethod === 'auto' 
            ? 'Automatically discovers URLs by crawling the website (works even if sitemap is protected)'
            : 'Uses sitemap.xml from robots.txt (faster but may miss URLs if sitemap is incomplete or protected)'}
        </div>
      </div>

      {localError && (
        <div className="alert error">
          <div style={{ whiteSpace: 'pre-line', marginBottom: discoveryMethod === 'sitemap' ? '12px' : '0' }}>
            {localError}
          </div>
          {discoveryMethod === 'sitemap' && localError.includes('sitemap') && (
            <div style={{ marginTop: '12px' }}>
              <button
                className="secondary"
                onClick={() => {
                  setDiscoveryMethod('auto')
                  setLocalError(null)
                }}
                style={{ marginRight: '8px' }}
              >
                Switch to Automatic (Recommended)
              </button>
            </div>
          )}
        </div>
      )}

      <div className="flow-actions">
        {isDiscovering ? (
          <button type="button" className="primary" onClick={stopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <StopIcon />
            Stop
          </button>
        ) : (
          <button type="button" className="primary" onClick={() => void handleContinue()} disabled={starting} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <PlayIcon />
            {starting ? 'Starting…' : 'Start bot training'}
          </button>
        )}
      </div>
    </div>
  )
}
