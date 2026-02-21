import { LogOut, Mail, User } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useDashboardData } from '../hooks/useDashboardData'
import PageHeader from '../components/PageHeader'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../components/ui'

export default function AccountPage() {
  const { t } = useTranslation()
  const { user, logout } = useDashboardData()

  return (
    <AnimatedPage className="page narrow">
      <PageHeader title={t('account.title', 'Account')} />
      <div className="page-body page-body-narrow">
        <SectionHeader
          eyebrow={t('account.profileEyebrow', 'Profile')}
          title={t('account.yourAccount', 'Your account')}
          subtitle={t('account.profileSubtitle', 'Manage identity, access, and workspace presence.')}
        />
        <GlassCard>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <User size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            {t('account.accountDetails', 'Account details')}
          </div>
          <div className="detail-row">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Mail size={14} style={{ color: 'var(--ui-flow-muted)' }} />
              {t('account.emailLabel', 'Email')}
            </span>
            <span>{user?.email || t('account.notAvailable', 'Not available')}</span>
          </div>
          <div style={{ marginTop: '1rem' }}>
            <UiButton
              variant="ghost"
              onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
            >
              <LogOut size={15} />
              {t('account.signOut', 'Sign out')}
            </UiButton>
          </div>
        </GlassCard>
      </div>
    </AnimatedPage>
  )
}
