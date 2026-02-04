import PageHeader from '../components/PageHeader'

export default function BillingPage() {
  return (
    <div className="page">
      <PageHeader title="Billing and subscriptions" />
      <div className="page-body page-body-narrow">
        <section className="card">
          <div className="card-title">Billing</div>
          <p className="muted">Billing and subscription management will be available here.</p>
        </section>
      </div>
    </div>
  )
}
