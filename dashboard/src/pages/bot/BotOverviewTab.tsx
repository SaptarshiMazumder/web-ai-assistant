import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Check, Clock, ExternalLink, Globe, Key } from 'lucide-react'
import { LineIcon } from '../../assets/icons/LineIcon'
import { useDashboardData, type LineChannelRecord, type LineChannelTestResult } from '../../hooks/useDashboardData'
import DashboardAnalytics from '../../components/DashboardAnalytics'
import { GlassCard } from '../../components/ui'
import { useTranslation } from 'react-i18next'

type SetupIndicator = {
  id: string
  label: string
  done: boolean
  route: string
}

type InstallTone = 'live' | 'pending' | 'not-deployed' | 'error'

function installToneClassName(tone: InstallTone) {
  if (tone === 'error') return 'overview-status overview-status--error'
  return `overview-status overview-status--${tone}`
}

export default function BotOverviewTab() {
  const { t, i18n } = useTranslation()
  const { botId } = useParams()
  const {
    selectedBot,
    domains,
    getBotOverviewSetup,
    getLineChannel,
    testLineChannel,
  } = useDashboardData()

  const [setupIndicators, setSetupIndicators] = useState<SetupIndicator[]>([])
  const [lineChannel, setLineChannel] = useState<LineChannelRecord | null>(null)
  const [lineTest, setLineTest] = useState<LineChannelTestResult | null>(null)
  const [lineLoading, setLineLoading] = useState(false)

  useEffect(() => {
    if (!botId || !selectedBot) {
      setSetupIndicators([])
      return
    }
    let cancelled = false
    void getBotOverviewSetup(botId, i18n.language || undefined, selectedBot.org_id).then((sections) => {
      if (!cancelled) {
        setSetupIndicators(sections)
      }
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botId, selectedBot?.org_id, i18n.language])

  useEffect(() => {
    if (!botId || !selectedBot) {
      setLineChannel(null)
      setLineTest(null)
      setLineLoading(false)
      return
    }
    let cancelled = false
    setLineLoading(true)
    void getLineChannel(botId, selectedBot.org_id).then(async (channel) => {
      if (cancelled) return
      setLineChannel(channel)
      if (!channel) {
        setLineTest(null)
        setLineLoading(false)
        return
      }
      const testResult = await testLineChannel(botId, selectedBot.org_id)
      if (cancelled) return
      setLineTest(testResult)
      setLineLoading(false)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botId, selectedBot?.org_id])

  if (!selectedBot || !botId) {
    return <div className="empty-panel">{t('botOverview.selectBot', 'Select a bot to view overview details.')}</div>
  }

  const notAvailable = t('botOverview.notAvailable', 'Not available')
  const formatDateTime = (value?: string | null) => (value ? new Date(value).toLocaleString(i18n.language || undefined) : notAvailable)
  const verifiedDomains = domains.filter((domain) => !!domain.verified_at)
  const pendingDomains = domains.filter((domain) => !domain.verified_at)
  const latestVerifiedDomainAt = verifiedDomains
    .map((domain) => domain.verified_at || '')
    .filter(Boolean)
    .sort()
    .at(-1) || null

  const websiteTone: InstallTone = verifiedDomains.length > 0 ? 'live' : pendingDomains.length > 0 ? 'pending' : 'not-deployed'
  const websiteStatusLabel = verifiedDomains.length > 0
    ? t('botOverview.statusActive', 'Active')
    : pendingDomains.length > 0
      ? t('botOverview.statusPending', 'Pending')
      : t('botOverview.statusNotConnected', 'Not connected')
  const websiteSummary = verifiedDomains.length > 0
    ? t('botOverview.websiteActiveSummary', 'Live on {{count}} verified website(s)', { count: verifiedDomains.length })
    : pendingDomains.length > 0
      ? t('botOverview.websitePendingSummary', '{{count}} domain(s) waiting for verification', { count: pendingDomains.length })
      : t('botOverview.websiteNotConnectedSummary', 'No verified website installation yet')
  const websitePrimaryDomain = verifiedDomains[0]?.hostname || pendingDomains[0]?.hostname || notAvailable

  const lineTone: InstallTone = lineLoading
    ? 'pending'
    : !lineChannel
      ? 'not-deployed'
      : lineTest && lineTest.ok === false
        ? 'error'
        : lineChannel.is_active
          ? 'live'
          : 'pending'
  const lineStatusLabel = lineLoading
    ? t('botOverview.statusChecking', 'Checking...')
    : !lineChannel
      ? t('botOverview.statusNotConnected', 'Not connected')
      : lineTest && lineTest.ok === false
        ? t('botOverview.statusError', 'Error')
        : lineChannel.is_active
          ? t('botOverview.statusActive', 'Active')
          : t('botOverview.statusPaused', 'Paused')
  const lineSummary = lineLoading
    ? t('botOverview.lineCheckingSummary', 'Checking live LINE connection')
    : !lineChannel
      ? t('botOverview.lineNotConnectedSummary', 'No LINE channel connected yet')
      : lineTest && lineTest.ok === false
        ? t('botOverview.lineErrorSummary', 'LINE connection needs attention')
        : lineChannel.is_active
          ? t('botOverview.lineActiveSummary', 'Connected and responding on LINE')
          : t('botOverview.linePausedSummary', 'Connected on LINE but currently paused')
  const lineAccountName = lineTest?.display_name || lineTest?.basic_id || lineChannel?.line_channel_id || notAvailable

  return (
    <>
      <DashboardAnalytics
        botId={selectedBot.bot_id}
        setupPills={
          <div className="summary-pills-inner">
            {setupIndicators.map((ind) => (
              <Link
                key={ind.id}
                to={`/bots/${botId}/${ind.route}`}
                className="summary-pill"
                title={ind.done
                  ? t('botOverview.setupDoneTitle', '{{label}} is set up', { label: ind.label })
                  : t('botOverview.setupPendingTitle', 'Set up {{label}}', { label: ind.label })}
              >
                {ind.done ? (
                  <span className="summary-pill-icon summary-pill-icon--check" aria-hidden>
                    <Check size={13} strokeWidth={2.5} />
                  </span>
                ) : (
                  <span className="summary-pill-icon summary-pill-icon--setup" aria-hidden title={t('botOverview.pending', 'Pending')}>
                    <Clock size={13} strokeWidth={2.5} />
                  </span>
                )}
                <span className="summary-pill-label">{ind.label}</span>
              </Link>
            ))}
          </div>
        }
      />

      <div className="card-grid" style={{ marginTop: 16 }}>
        <GlassCard>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Key size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            {t('botOverview.botDetailsTitle', 'Bot details')}
          </div>
          <div className="detail-row">
            <span>{t('botOverview.botId', 'Bot ID')}</span>
            <code>{selectedBot.bot_id}</code>
          </div>
          <div className="detail-row">
            <span>{t('botOverview.publishableKey', 'Publishable key')}</span>
            <code>{selectedBot.publishable_key}</code>
          </div>
          <div className="detail-row">
            <span>{t('botOverview.secretKey', 'Secret key')}</span>
            <code>{selectedBot.secret_key}</code>
          </div>
          <div className="detail-row">
            <span>{t('botOverview.created', 'Created')}</span>
            <span>{new Date(selectedBot.created_at).toLocaleString(i18n.language || undefined)}</span>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Globe size={16} style={{ color: 'var(--ui-flow-accent)' }} />
            {t('botOverview.installStatusTitle', 'Install status')}
          </div>
          <p className="muted" style={{ marginBottom: '1rem' }}>
            {t('botOverview.installStatusSubtitle', 'Live channel status for Website and LINE.')}
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            <div style={{ border: '1px solid var(--ui-flow-border)', borderRadius: 16, padding: '1rem', background: 'var(--ui-flow-surface)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
                  <Globe size={16} style={{ color: 'var(--ui-flow-accent)', flexShrink: 0 }} />
                  <span style={{ fontWeight: 600 }}>{t('botOverview.websiteLabel', 'Website')}</span>
                </div>
                <span className={installToneClassName(websiteTone)}>{websiteStatusLabel}</span>
              </div>
              <p style={{ margin: '0 0 0.85rem 0', color: 'var(--text-secondary)', fontSize: '0.92rem', lineHeight: 1.5 }}>
                {websiteSummary}
              </p>
              <div className="detail-row">
                <span>{t('botOverview.domain', 'Domain')}</span>
                <span>{websitePrimaryDomain}</span>
              </div>
              <div className="detail-row">
                <span>{t('botOverview.connectedDomains', 'Connected domains')}</span>
                <span>{verifiedDomains.length}</span>
              </div>
              <div className="detail-row">
                <span>{t('botOverview.pendingDomains', 'Pending domains')}</span>
                <span>{pendingDomains.length}</span>
              </div>
              <div className="detail-row">
                <span>{t('botOverview.lastVerified', 'Last verified')}</span>
                <span>{formatDateTime(latestVerifiedDomainAt)}</span>
              </div>
              <Link
                to={`/bots/${botId}/website`}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.85rem', textDecoration: 'none', fontWeight: 600, color: 'var(--ui-flow-accent)' }}
              >
                {t('botOverview.manageWebsite', 'Manage website install')}
                <ExternalLink size={14} />
              </Link>
            </div>

            <div style={{ border: '1px solid var(--ui-flow-border)', borderRadius: 16, padding: '1rem', background: 'var(--ui-flow-surface)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
                  <LineIcon size={16} style={{ color: 'var(--ui-flow-accent)', flexShrink: 0 }} />
                  <span style={{ fontWeight: 600 }}>{t('botOverview.lineLabel', 'LINE')}</span>
                </div>
                <span className={installToneClassName(lineTone)}>{lineStatusLabel}</span>
              </div>
              <p style={{ margin: '0 0 0.85rem 0', color: 'var(--text-secondary)', fontSize: '0.92rem', lineHeight: 1.5 }}>
                {lineSummary}
              </p>
              <div className="detail-row">
                <span>{t('botOverview.account', 'Account')}</span>
                <span>{lineAccountName}</span>
              </div>
              <div className="detail-row">
                <span>{t('botOverview.channelId', 'Channel ID')}</span>
                <span>{lineChannel?.line_channel_id || notAvailable}</span>
              </div>
              <div className="detail-row">
                <span>{t('botOverview.basicId', 'Basic ID')}</span>
                <span>{lineTest?.basic_id || notAvailable}</span>
              </div>
              <div className="detail-row">
                <span>{t('botOverview.lastUpdated', 'Last updated')}</span>
                <span>{formatDateTime(lineChannel?.updated_at)}</span>
              </div>
              {lineTest?.message ? (
                <div
                  style={{
                    marginTop: '0.85rem',
                    padding: '0.75rem 0.85rem',
                    borderRadius: 12,
                    background: lineTest.ok ? 'rgba(34, 197, 94, 0.08)' : 'rgba(239, 68, 68, 0.08)',
                    color: lineTest.ok ? '#166534' : '#b91c1c',
                    fontSize: '0.88rem',
                    lineHeight: 1.45,
                  }}
                >
                  {lineTest.message}
                </div>
              ) : null}
              <Link
                to={`/bots/${botId}/line`}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.85rem', textDecoration: 'none', fontWeight: 600, color: 'var(--ui-flow-accent)' }}
              >
                {t('botOverview.manageLine', 'Manage LINE')}
                <ExternalLink size={14} />
              </Link>
            </div>
          </div>
        </GlassCard>
      </div>
    </>
  )
}
