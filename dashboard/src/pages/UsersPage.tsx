import PageHeader from '../components/PageHeader'
import { AnimatedPage, GlassCard, EmptyState, SectionHeader } from '../components/ui'

export default function UsersPage() {
  return (
    <AnimatedPage className="page">
      <PageHeader title="Users" />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow="Access"
          title="User management"
          subtitle="Manage user-level controls and access policies."
        />
        <GlassCard>
          <EmptyState
            title="User controls in progress"
            description="Fine-grained user access policies, activity logs, and role management will launch here."
          />
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
