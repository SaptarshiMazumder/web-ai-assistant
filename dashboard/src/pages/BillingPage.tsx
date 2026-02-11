import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, UiButton } from '../components/ui'

export default function BillingPage() {
  return (
    <AnimatedPage className="page">
      <PageHeader title="Billing and subscriptions" />
      <div className="page-body page-body-narrow">
        <GlassCard>
          <EmptyState
            title="Billing cockpit is coming"
            description="Advanced plans, invoices, and usage controls will launch here with the same warm dashboard styling."
            action={<UiButton variant="secondary">Notify me when live</UiButton>}
          />
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
