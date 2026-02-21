import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, UiButton } from '../components/ui'
import { useTranslation } from 'react-i18next'

export default function BillingPage() {
  const { t } = useTranslation()
  return (
    <AnimatedPage className="page">
      <PageHeader title={t('billingPage.title', 'Billing and subscriptions')} />
      <div className="page-body page-body-narrow">
        <GlassCard>
          <EmptyState
            title={t('billingPage.cockpitComing', 'Billing cockpit is coming')}
            description={t('billingPage.cockpitDesc', 'Advanced plans, invoices, and usage controls will launch here with the same warm dashboard styling.')}
            action={<UiButton variant="secondary">{t('billingPage.notifyMe', 'Notify me when live')}</UiButton>}
          />
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
