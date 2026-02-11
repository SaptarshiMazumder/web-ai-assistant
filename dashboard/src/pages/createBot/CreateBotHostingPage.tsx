import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotHostingPage() {
  const navigate = useNavigate()
  const { step2, flow } = useCreateBotFlow()
  const { websiteUrl, setWebsiteUrl, localError, setLocalError } = step2
  const [continuing, setContinuing] = useState(false)
  const canContinue = !!websiteUrl.trim() && !continuing

  const handleContinue = async () => {
    if (!flow.nextPath || !canContinue) return
    setContinuing(true)
    setLocalError(null)
    try {
      const raw = websiteUrl.trim()
      if (!raw) return
      try {
        const withProtocol = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`
        const parsed = new URL(withProtocol)
        if (!/^https?:$/i.test(parsed.protocol)) {
          setLocalError('Enter a valid website URL.')
          return
        }
      } catch {
        setLocalError('Enter a valid website URL.')
        return
      }
      navigate(flow.nextPath)
    } finally {
      setContinuing(false)
    }
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Website URL</div>
        <div className="card-subtitle">Enter your website or section URL. You can use a subpath like `example.com/hotel/tokyo`.</div>
      </div>

      <div className="flow-field">
        <label className="flow-field-label">Website URL</label>
        <div className="flow-field-input-wrap">
          <input
            type="url"
            value={websiteUrl}
            onChange={(e) => setWebsiteUrl(e.target.value)}
            placeholder="https://yourwebsite.com or https://yourwebsite.com/section/"
          />
        </div>
        <span className="flow-field-helper">We’ll discover pages from this URL scope in the next step.</span>
      </div>

      {localError && (
        <div className="alert error">
          <div style={{ whiteSpace: 'pre-line' }}>{localError}</div>
        </div>
      )}

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        <UiButton variant="primary" onClick={() => void handleContinue()} disabled={!canContinue}>
          {continuing ? 'Checking...' : 'Continue'}
        </UiButton>
      </div>
    </div>
  )
}
