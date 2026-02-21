import { useTranslation } from 'react-i18next'

export default function BotNotificationsTab() {
    const { t } = useTranslation()

    return (
        <div className="ui-glass-card">
            <h3>{t('botNotifications.title', 'Notifications')}</h3>
            <p className="muted">
                {t('botNotifications.subtitle', 'View your agent notifications and alerts here.')}
            </p>
            <p className="muted" style={{ marginTop: '1rem' }}>
                {t('botNotifications.noNewNotifications', 'No new notifications.')}
            </p>
        </div>
    )
}
