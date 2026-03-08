export type MessageLinkPart =
  | { type: 'text'; content: string }
  | { type: 'link'; text: string; url: string }

const URL_SAFE_CHAR_RE = /^[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]$/

function friendlyLabelFromUrl(url: string): string {
  try {
    const parsed = new URL(url)
    const path = (parsed.pathname || '/').replace(/\/$/, '').replace(/^\//, '')
    if (path) {
      const last = path.split('/').pop() || ''
      const cleaned = last.replace(/-/g, ' ').replace(/_/g, ' ')
      return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : parsed.hostname
    }
    return parsed.hostname
  } catch {
    return ''
  }
}

function isPdfLocalUrl(url: string): boolean {
  try {
    return new URL(url).hostname.toLowerCase() === 'pdf.local'
  } catch {
    return false
  }
}

function isGenericLinkText(label: string): boolean {
  const normalized = (label || '').trim().toLowerCase().replace(/\s+/g, ' ')
  if (!normalized) return true
  if (/^sources?$/.test(normalized) || normalized === 'link' || normalized === 'here') return true
  if (normalized.includes('http://') || normalized.includes('https://')) return true
  if (normalized.startsWith('/') && !normalized.includes(' ')) return true
  if (/^[a-z0-9.-]+\.[a-z]{2,}(\/.*)?$/.test(normalized) && !normalized.includes(' ')) return true
  return false
}

function isUrlChar(ch: string): boolean {
  return !!ch && ch.charCodeAt(0) <= 127 && URL_SAFE_CHAR_RE.test(ch)
}

function countChar(text: string, target: string): number {
  let count = 0
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] === target) count += 1
  }
  return count
}

function trimUrlSuffix(rawToken: string): { url: string; trailing: string } {
  let value = rawToken || ''
  let trailing = ''
  while (value) {
    const tail = value[value.length - 1]
    if (/[.,;:!?]/.test(tail) || tail === '"' || tail === '\'') {
      trailing = `${tail}${trailing}`
      value = value.slice(0, -1)
      continue
    }
    const opener = tail === ')' ? '(' : tail === ']' ? '[' : tail === '}' ? '{' : ''
    if (opener && countChar(value, opener) < countChar(value, tail)) {
      trailing = `${tail}${trailing}`
      value = value.slice(0, -1)
      continue
    }
    break
  }
  return { url: value, trailing }
}

function isValidHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

function pushTextPart(parts: MessageLinkPart[], content: string): void {
  if (!content) return
  const last = parts.length ? parts[parts.length - 1] : null
  if (last?.type === 'text') {
    last.content += content
    return
  }
  parts.push({ type: 'text', content })
}

function parseMarkdownLinkAt(text: string, start: number): { start: number; end: number; label: string; url: string } | null {
  if (start < 0 || start >= text.length || text[start] !== '[') return null
  const closeBracket = text.indexOf(']', start + 1)
  if (closeBracket < 0 || closeBracket + 1 >= text.length || text[closeBracket + 1] !== '(') return null
  let depth = 1
  for (let i = closeBracket + 2; i < text.length; i += 1) {
    const ch = text[i]
    if (ch === '(') depth += 1
    else if (ch === ')') {
      depth -= 1
      if (depth === 0) {
        return {
          start,
          end: i + 1,
          label: text.slice(start + 1, closeBracket),
          url: text.slice(closeBracket + 2, i),
        }
      }
    }
  }
  return null
}

function findNextMarkdownLink(text: string, fromIndex: number): { start: number; end: number; label: string; url: string } | null {
  let cursor = fromIndex
  while (cursor < text.length) {
    const openBracket = text.indexOf('[', cursor)
    if (openBracket < 0) return null
    const parsed = parseMarkdownLinkAt(text, openBracket)
    if (parsed) return parsed
    cursor = openBracket + 1
  }
  return null
}

function consumeUrlToken(
  text: string,
  start: number,
  explicitLabel: string,
): { end: number; link: Extract<MessageLinkPart, { type: 'link' }>; trailing: string } | null {
  let end = start
  while (end < text.length && isUrlChar(text[end])) end += 1
  if (end <= start) return null
  const trimmed = trimUrlSuffix(text.slice(start, end))
  if (!isValidHttpUrl(trimmed.url)) return null
  let label = (explicitLabel || '').trim()
  if (!label || isGenericLinkText(label)) {
    label = friendlyLabelFromUrl(trimmed.url) || 'this page'
  }
  return {
    end,
    link: { type: 'link', text: label, url: trimmed.url },
    trailing: trimmed.trailing,
  }
}

function appendParsedText(parts: MessageLinkPart[], content: string): void {
  let cursor = 0
  while (cursor < content.length) {
    const match = /https?:\/\//i.exec(content.slice(cursor))
    if (!match) {
      pushTextPart(parts, content.slice(cursor))
      return
    }
    const urlStart = cursor + match.index
    if (urlStart > cursor) pushTextPart(parts, content.slice(cursor, urlStart))
    const consumed = consumeUrlToken(content, urlStart, '')
    if (!consumed) {
      pushTextPart(parts, content.slice(urlStart, urlStart + 1))
      cursor = urlStart + 1
      continue
    }
    if (!isPdfLocalUrl(consumed.link.url)) {
      parts.push(consumed.link)
    }
    if (consumed.trailing) pushTextPart(parts, consumed.trailing)
    cursor = consumed.end
  }
}

export function parseMessageLinks(text: string): MessageLinkPart[] | null {
  if (!text) return null
  const parts: MessageLinkPart[] = []
  let cursor = 0
  while (cursor < text.length) {
    const markdown = findNextMarkdownLink(text, cursor)
    if (!markdown) {
      appendParsedText(parts, text.slice(cursor))
      break
    }
    if (markdown.start > cursor) appendParsedText(parts, text.slice(cursor, markdown.start))
    const consumed = consumeUrlToken(markdown.url.trim(), 0, markdown.label)
    if (!consumed) {
      pushTextPart(parts, text.slice(markdown.start, markdown.end))
    } else {
      if (!isPdfLocalUrl(consumed.link.url)) {
        parts.push(consumed.link)
      }
      const suffixText = `${consumed.trailing || ''}${markdown.url.trim().slice(consumed.end)}`
      if (suffixText) pushTextPart(parts, suffixText)
    }
    cursor = markdown.end
  }
  return parts.length ? parts : null
}
