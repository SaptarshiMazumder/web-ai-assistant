import PageHeader from '../components/PageHeader'
import { AnimatedPage, GlassCard, EmptyState, SectionHeader } from '../components/ui'

export default function DomainPage() {
  return (
    <AnimatedPage className="page">
      <PageHeader title="Domain" />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow="Infrastructure"
          title="Domain management"
          subtitle="Verify and manage custom domains for your bots."
        />
        <GlassCard>
          <EmptyState
            title="Domain controls coming soon"
            description="Custom domain verification, DNS management, and SSL configuration will be available here."
          />
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
