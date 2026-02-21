import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, UiButton } from '../components/ui'
import { LanguageSelector } from '../components/LanguageSelector'
import { useTranslation } from 'react-i18next'

export default function SettingsPage() {
  const { t } = useTranslation()

  return (
    <AnimatedPage className="page">
      <PageHeader title={t('settings.title', 'Settings')} />
      <div className="page-body page-body-narrow" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <GlassCard>
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem', color: 'var(--flow-heading)' }}>
              {t('settings.language', 'Language')}
            </h3>
            <p style={{ margin: '0 0 1rem 0', color: 'var(--flow-muted)', fontSize: '0.9rem' }}>
              {t('settings.languageDescription', 'Choose your preferred language for the dashboard interface.')}
            </p>
            <LanguageSelector />
          </div>
        </GlassCard>

        <GlassCard>
          <EmptyState
            title={t('settings.personalSettingsHub', 'Personal settings hub')}
            description={t('settings.personalSettingsDesc', 'Notification tuning, workspace defaults, and automation preferences are being redesigned.')}
            action={<UiButton variant="ghost">{t('settings.previewFutureControls', 'Preview future controls')}</UiButton>}
          />
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
