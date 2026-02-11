import PageHeader from '../components/PageHeader'
import { AnimatedPage, EmptyState, GlassCard, UiButton } from '../components/ui'

export default function SettingsPage() {
  return (
    <AnimatedPage className="page">
      <PageHeader title="Settings" />
      <div className="page-body page-body-narrow">
        <GlassCard>
          <EmptyState
            title="Personal settings hub"
            description="Notification tuning, workspace defaults, and automation preferences are being redesigned."
            action={<UiButton variant="ghost">Preview future controls</UiButton>}
          />
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
