import type { EscalationRecord } from '../hooks/useDashboardData'

type Translator = (en: string, ja: string) => string

export function isLineEscalation(record: Pick<EscalationRecord, 'visitor_email'> | null | undefined) {
  return /^line:/i.test(record?.visitor_email || '')
}

export function isInstagramEscalation(record: Pick<EscalationRecord, 'visitor_email'> | null | undefined) {
  return /^(instagram|ig):/i.test(record?.visitor_email || '')
}

function preferredDisplayName(record: Pick<EscalationRecord, 'visitor_name' | 'title'> | null | undefined) {
  return (record?.visitor_name || record?.title || '').trim()
}

export function getEscalationTitle(record: EscalationRecord, tr: Translator) {
  const displayName = preferredDisplayName(record)
  if (displayName) return displayName
  if (isLineEscalation(record)) return tr('LINE customer', 'LINE顧客')
  if (isInstagramEscalation(record)) return tr('Instagram customer', 'Instagram顧客')
  return record.site_title || record.site_url || record.session_id
}

export function getEscalationSubtitle(record: EscalationRecord, tr: Translator) {
  if (isLineEscalation(record)) return tr('LINE user', 'LINEユーザー')
  if (isInstagramEscalation(record)) return tr('Instagram user', 'Instagramユーザー')
  return record.visitor_email || record.site_url || record.session_id
}

export function getEscalationContact(record: EscalationRecord | null, tr: Translator) {
  if (!record) {
    return {
      label: tr('Customer', '顧客'),
      value: '',
    }
  }
  const displayName = preferredDisplayName(record)
  if (isLineEscalation(record)) {
    return {
      label: tr('LINE user', 'LINEユーザー'),
      value: displayName || tr('LINE customer', 'LINE顧客'),
    }
  }
  if (isInstagramEscalation(record)) {
    return {
      label: tr('Instagram user', 'Instagramユーザー'),
      value: displayName || tr('Instagram customer', 'Instagram顧客'),
    }
  }
  return {
    label: tr('Email', 'メール'),
    value: record.visitor_email || '',
  }
}
