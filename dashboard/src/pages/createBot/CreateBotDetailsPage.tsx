import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotDetailsPage() {
  const navigate = useNavigate()
  const { botName, setBotName, websiteUrl, setWebsiteUrl, isDiscovering, localError, discoverUrls } = useCreateBotFlow()

  const handleContinue = async () => {
    const ok = await discoverUrls()
    if (ok) {
      navigate('/create-bot/urls')
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

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <button className="primary" onClick={handleContinue} disabled={isDiscovering}>
          {isDiscovering ? 'Finding URLs...' : 'Start bot training'}
        </button>
      </div>
    </div>
  )
}
