import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Globe, Share2 } from 'lucide-react'
import { UiButton } from '../../components/ui'
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
        <div className="card-title">Where does your content live?</div>
        <div className="card-subtitle">This helps us find and learn from your business information.</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem' }}>
        <button
          type="button"
          className={`flow-hosting-card ${contentHosting === 'own' ? 'selected' : ''}`}
          onClick={() => setContentHosting('own')}
        >
          <div style={{ marginBottom: '0.75rem', position: 'relative', zIndex: 1 }}>
            <div style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: contentHosting === 'own' ? 'var(--flow-accent)' : 'var(--flow-accent-soft)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background 0.2s ease',
            }}>
              <Globe size={20} strokeWidth={1.8} color={contentHosting === 'own' ? '#fff' : 'var(--flow-accent)'} />
            </div>
          </div>
          <div className="flow-hosting-card-title">I have my own website</div>
          <div className="flow-hosting-card-desc">We'll scan your site and find pages automatically.</div>
        </button>

        <button
          type="button"
          className={`flow-hosting-card ${contentHosting === 'shared' ? 'selected' : ''}`}
          onClick={() => setContentHosting('shared')}
        >
          <div style={{ marginBottom: '0.75rem', position: 'relative', zIndex: 1 }}>
            <div style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: contentHosting === 'shared' ? 'var(--flow-accent)' : 'var(--flow-accent-soft)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background 0.2s ease',
            }}>
              <Share2 size={20} strokeWidth={1.8} color={contentHosting === 'shared' ? '#fff' : 'var(--flow-accent)'} />
            </div>
          </div>
          <div className="flow-hosting-card-title">I use a website service</div>
          <div className="flow-hosting-card-desc">You'll add links and upload files manually.</div>
        </button>
      </div>

      {contentHosting === 'own' && (
        <div className="flow-field">
          <label className="flow-field-label">Website URL</label>
          <div className="flow-field-input-wrap">
            <input
              type="url"
              value={websiteUrl}
              onChange={(e) => setWebsiteUrl(e.target.value)}
              placeholder="https://yourwebsite.com"
            />
          </div>
          <span className="flow-field-helper">We will discover pages from this domain.</span>
        </div>
      )}

      {localError && (
        <div className="alert error">
          <div style={{ whiteSpace: 'pre-line' }}>{localError}</div>
        </div>
      )}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        {isDiscovering ? (
          <UiButton variant="primary" onClick={stopDiscovery}>
            Stop
          </UiButton>
        ) : (
          <UiButton variant="primary" onClick={() => void handleContinue()} disabled={!canContinue}>
            {continuing ? 'Scanning...' : 'Continue'}
          </UiButton>
        )}
      </div>
    </div>
  )
}
