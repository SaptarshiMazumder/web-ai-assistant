import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotUrlsPage() {
  const navigate = useNavigate()
  const {
    discoveredUrls,
    selectedUrls,
    normalizedWebsiteUrl,
    toggleUrl,
    selectAll,
    deselectAll,
    localError,
    startTraining,
  } = useCreateBotFlow()

  useEffect(() => {
    if (!discoveredUrls.length) {
      navigate('/create-bot')
    }
  }, [discoveredUrls.length, navigate])

  const handleStartTraining = async () => {
    const botId = await startTraining()
    if (botId) {
      navigate('/create-bot/progress')
    }
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Select URLs to train on</div>
        <div className="card-subtitle">
          We found {discoveredUrls.length} pages on {normalizedWebsiteUrl}. Choose the ones your bot should learn from.
        </div>
      </div>

      <div className="flow-toolbar">
        <button className="secondary" onClick={selectAll}>
          Select all
        </button>
        <button className="ghost" onClick={deselectAll}>
          Deselect all
        </button>
        <div className="muted">{selectedUrls.length} selected</div>
      </div>

      <div className="url-list">
        {discoveredUrls.map((url) => (
          <label key={url} className={`url-row ${selectedUrls.includes(url) ? 'selected' : ''}`}>
            <input type="checkbox" checked={selectedUrls.includes(url)} onChange={() => toggleUrl(url)} />
            <span>{url}</span>
          </label>
        ))}
      </div>

      {localError && <div className="alert error">{localError}</div>}

      <div className="flow-actions">
        <button className="secondary" onClick={() => navigate('/create-bot')}>
          Back
        </button>
        <button className="primary" onClick={handleStartTraining}>
          Start training
        </button>
      </div>
    </div>
  )
}
