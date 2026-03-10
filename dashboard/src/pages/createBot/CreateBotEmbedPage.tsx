import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CheckCircle2, ChevronLeft, Globe, MessageCircle } from 'lucide-react'
import { FlowIcon } from '../../components/FlowIcon'
import { UiButton } from '../../components/ui'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'
import { LineIcon } from '../../assets/icons/LineIcon'
import BotLineSettingsTab from '../bot/BotLineSettingsTab'

type InstallView = 'hub' | 'website' | 'line'

export default function CreateBotEmbedPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const {
    buildEmbedSnippet,
    copySnippet,
    domains,
    loadDomains,
    getLineChannel,
  } = useDashboardData()
  const { step3, flow } = useCreateBotFlow()
  const { botId } = step3
  const [copied, setCopied] = useState(false)
  const [installView, setInstallView] = useState<InstallView>('hub')
  const [websiteInstalledManual, setWebsiteInstalledManual] = useState(false)
  const [lineInstalled, setLineInstalled] = useState(false)

  const snippet = useMemo(() => buildEmbedSnippet(), [buildEmbedSnippet])
  const websiteInstalled = useMemo(
    () => websiteInstalledManual || domains.some((domain) => Boolean(domain.verified_at)),
    [websiteInstalledManual, domains]
  )

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  useEffect(() => {
    if (!botId) return
    let cancelled = false
    void loadDomains(botId)
    void getLineChannel(botId).then((channel) => {
      if (cancelled) return
      setLineInstalled(Boolean(channel?.is_active))
    })
    return () => {
      cancelled = true
    }
  }, [botId, loadDomains, getLineChannel])

  const handleCopy = async () => {
    await copySnippet(snippet)
    setCopied(true)
    setTimeout(() => setCopied(false), 2500)
  }

  const goToOverview = () => {
    if (botId) {
      navigate(`/bots/${botId}/overview`)
    } else {
      navigate('/bots')
    }
  }

  if (installView === 'website') {
    return (
      <div className="flow-panel-body">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            type="button"
            onClick={() => setInstallView('hub')}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', border: 'none', background: 'transparent', color: 'var(--flow-muted)', cursor: 'pointer', padding: 0 }}
          >
            <ChevronLeft size={16} />
            {t('createBot.backToInstallHub', 'Back to install')}
          </button>
        </div>

        <div>
          <div className="card-title">{t('createBot.websiteInstallTitle', 'Website installation')}</div>
          <div className="card-subtitle">
            {t(
              'createBot.websiteInstallInstructions',
              'Copy this snippet and paste it right before the closing </body> tag on your website.'
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
          <pre className="snippet" style={{ margin: 0, flex: 1, minWidth: 0 }}>
            {snippet}
          </pre>
          <button
            type="button"
            onClick={() => void handleCopy()}
            disabled={!snippet}
            title={copied ? t('createBot.copied', 'Copied!') : t('createBot.copyToClipboard', 'Copy to clipboard')}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              border: '1px solid var(--flow-border, #f2d8d2)',
              background: copied ? 'var(--flow-accent-soft, #fff1ef)' : 'var(--flow-surface, #ffffff)',
              color: copied ? 'var(--flow-accent, #e4587a)' : 'var(--flow-muted, #7e5a70)',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              flexShrink: 0,
            }}
          >
            <FlowIcon name={copied ? 'check' : 'content_copy'} size="sm" />
          </button>
        </div>
        <div className="muted" style={{ fontSize: '0.85rem' }}>
          {t(
            'createBot.embedUpdateHint',
            'If you update the widget design later, changes will appear on your website automatically.'
          )}
        </div>
        <div className="flow-actions">
          <UiButton variant="secondary" onClick={() => setInstallView('hub')}>
            {t('createBot.skipForNow', 'Skip for now')}
          </UiButton>
          <UiButton
            variant="primary"
            onClick={() => {
              setWebsiteInstalledManual(true)
              setInstallView('hub')
            }}
          >
            {t('createBot.markWebsiteInstalled', 'Done for website')}
          </UiButton>
        </div>
      </div>
    )
  }

  if (installView === 'line') {
    return (
      <div className="flow-panel-body">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <button
            type="button"
            onClick={() => setInstallView('hub')}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', border: 'none', background: 'transparent', color: 'var(--flow-muted)', cursor: 'pointer', padding: 0 }}
          >
            <ChevronLeft size={16} />
            {t('createBot.backToInstallHub', 'Back to install')}
          </button>
        </div>
        {botId ? (
          <BotLineSettingsTab
            botIdOverride={botId}
            onConnected={() => {
              setLineInstalled(true)
              setInstallView('hub')
            }}
          />
        ) : null}
      </div>
    )
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">{t('createBot.installChannelsTitle', 'Install your AI agent')}</div>
        <div className="card-subtitle">
          {t('createBot.installChannelsSubtitle', 'Choose a platform card to open setup flow.')}
        </div>
      </div>

      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
        <button
          type="button"
          onClick={() => {
            setInstallView('website')
          }}
          style={{
            textAlign: 'left',
            border: websiteInstalled ? '2px solid #22c55e' : '1px solid var(--flow-border, #f2d8d2)',
            borderRadius: 16,
            background: websiteInstalled ? 'rgba(34,197,94,0.08)' : 'var(--flow-surface, #fff)',
            padding: '1rem',
            cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.7rem' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: 'var(--flow-text)' }}>
              <Globe size={18} />
              {t('createBot.websiteCardTitle', 'Website')}
            </div>
            {websiteInstalled ? <CheckCircle2 size={18} color="#22c55e" /> : null}
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--flow-muted)', lineHeight: 1.5 }}>
            {websiteInstalled
              ? t('createBot.websiteInstalledStatus', 'Installed')
              : t('createBot.websiteNotInstalledStatus', 'Not installed')}
          </div>
        </button>

        <button
          type="button"
          onClick={() => setInstallView('line')}
          style={{
            textAlign: 'left',
            border: lineInstalled ? '2px solid #22c55e' : '1px solid var(--flow-border, #f2d8d2)',
            borderRadius: 16,
            background: lineInstalled ? 'rgba(34,197,94,0.08)' : 'var(--flow-surface, #fff)',
            padding: '1rem',
            cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.7rem' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: 'var(--flow-text)' }}>
              <LineIcon size={18} />
              {t('createBot.lineCardTitle', 'LINE')}
            </div>
            {lineInstalled ? <CheckCircle2 size={18} color="#22c55e" /> : null}
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--flow-muted)', lineHeight: 1.5 }}>
            {lineInstalled
              ? t('createBot.lineInstalledStatus', 'Installed')
              : t('createBot.lineNotInstalledStatus', 'Not installed')}
          </div>
        </button>
      </div>

      <div style={{ border: '1px dashed var(--flow-border, #f2d8d2)', borderRadius: 12, padding: '0.9rem 1rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', color: 'var(--flow-muted)', fontSize: '0.9rem' }}>
        <MessageCircle size={16} />
        {t('createBot.installOptionalHint', 'You can always set this up later.')}
      </div>

      <div className="flow-actions">
        <UiButton variant="secondary" onClick={goToOverview}>
          {t('createBot.skipForNow', 'Skip for now')}
        </UiButton>
        <UiButton
          variant="primary"
          onClick={goToOverview}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <FlowIcon name="celebration" filled size="sm" />
          {t('createBot.finishSetup', 'Finish setup')}
        </UiButton>
      </div>
    </div>
  )
}
