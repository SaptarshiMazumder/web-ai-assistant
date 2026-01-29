import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotWidgetPage() {
  const navigate = useNavigate()
  const { step3, step4, flow } = useCreateBotFlow()
  const { botId } = step3
  const {
    widgetPosition,
    setWidgetPosition,
    widgetPrimaryColor,
    setWidgetPrimaryColor,
    widgetTitle,
    setWidgetTitle,
    widgetSize,
    setWidgetSize,
  } = step4

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  const handleContinue = () => {
    if (flow.nextPath) navigate(flow.nextPath)
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Design the chat widget</div>
        <div className="card-subtitle">
          Customize how the widget appears on your website. You can change these later in bot settings.
        </div>
      </div>

      <div>
        <div className="card-title">Position</div>
        <div className="card-subtitle">Where the chat bubble appears on the page.</div>
        <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="radio"
              name="widgetPosition"
              value="bottom-right"
              checked={widgetPosition === 'bottom-right'}
              onChange={() => setWidgetPosition('bottom-right')}
            />
            <span>Bottom right</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="radio"
              name="widgetPosition"
              value="bottom-left"
              checked={widgetPosition === 'bottom-left'}
              onChange={() => setWidgetPosition('bottom-left')}
            />
            <span>Bottom left</span>
          </label>
        </div>
      </div>

      <div>
        <div className="card-title">Primary color</div>
        <div className="card-subtitle">Accent color for the widget (header, buttons).</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '8px' }}>
          <input
            type="color"
            value={widgetPrimaryColor}
            onChange={(e) => setWidgetPrimaryColor(e.target.value)}
            style={{ width: '44px', height: '36px', padding: 0, border: '1px solid #ccc', cursor: 'pointer' }}
          />
          <input
            type="text"
            value={widgetPrimaryColor}
            onChange={(e) => setWidgetPrimaryColor(e.target.value)}
            placeholder="#6366f1"
            style={{ width: '120px', padding: '8px 12px' }}
          />
        </div>
      </div>

      <div>
        <div className="card-title">Widget title</div>
        <div className="card-subtitle">Label shown in the widget header.</div>
        <input
          type="text"
          value={widgetTitle}
          onChange={(e) => setWidgetTitle(e.target.value)}
          placeholder="Chat"
          style={{ marginTop: '8px', maxWidth: '280px' }}
        />
      </div>

      <div>
        <div className="card-title">Size</div>
        <div className="card-subtitle">Widget window size.</div>
        <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
          {(['small', 'medium', 'large'] as const).map((size) => (
            <label key={size} style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="radio"
                name="widgetSize"
                value={size}
                checked={widgetSize === size}
                onChange={() => setWidgetSize(size)}
              />
              <span style={{ textTransform: 'capitalize' }}>{size}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="flow-actions">
        <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </button>
        <button type="button" className="primary" onClick={handleContinue}>
          Continue
        </button>
      </div>
    </div>
  )
}
