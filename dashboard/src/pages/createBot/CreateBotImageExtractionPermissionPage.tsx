import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { SectionHeader, UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotImageExtractionPermissionPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { step2, flow } = useCreateBotFlow()

  return (
    <div className="flow-panel-body">
      <SectionHeader
        title={t('createBot.imageExtractionPermissionTitle', 'Image extraction')}
        subtitle={t(
          'createBot.imageExtractionPermissionSubtitle',
          'Images will be extracted automatically after training.'
        )}
      />

      <section className="ui-glass-card">
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
          }}
        >
          <div style={{ fontWeight: 600, color: 'var(--flow-text)' }}>
            {t('createBot.allowAutoImageExtraction', 'Extract images automatically')}
          </div>
          <label className="toggle" style={{ marginTop: 0 }}>
            <input
              type="checkbox"
              aria-label={t('createBot.allowAutoImageExtraction', 'Extract images automatically')}
              checked={step2.allowAutoImageExtraction}
              onChange={(event) => step2.setAllowAutoImageExtraction(event.target.checked)}
            />
            <span className="toggle-slider" />
          </label>
        </div>
      </section>

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          {t('common.back', 'Back')}
        </UiButton>
        <UiButton variant="primary" onClick={() => flow.nextPath && navigate(flow.nextPath)}>
          {t('common.continue', 'Continue')}
        </UiButton>
      </div>
    </div>
  )
}
