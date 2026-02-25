export type UrlCategory = {
  name: string
  path: string
  urls: string[]
  children: Map<string, UrlCategory>
  level: number
}

/** Key for the "root-level pages" category so they're under one collapsible parent. */
const ROOT_PAGES_KEY = '\0root-pages'

type ParsedSegment = {
  key: string
  label: string
}

function safeDecode(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function normalizeSegmentKey(value: string): string {
  return safeDecode(value).trim().toLowerCase()
}

function segmentLabel(value: string): string {
  const decoded = safeDecode(value).trim()
  return decoded || value
}

/** Normalize URL key for dedupe/mapping: path '' and '/' are treated the same. */
export function getNormalizedUrlKey(url: string): string {
  try {
    const u = new URL(url)
    const normalizedPath = (u.pathname || '') === '' ? '/' : u.pathname
    if (normalizedPath === '/') return `${u.origin}/${u.search || ''}`
    return `${u.origin}${normalizedPath}${u.search || ''}`
  } catch {
    return (url || '').trim()
  }
}

/** Returns the path-style label for a category (e.g. "/", "/articles", "/articles/bmr"). */
export function getCategoryDisplayPath(category: UrlCategory): string {
  if (category.path === '' || category.path === ROOT_PAGES_KEY) return '/'
  return '/' + category.path
}

/** Returns all category paths that can be expanded (for "expand all" / default open). */
export function getAllExpandablePaths(category: UrlCategory): string[] {
  const paths: string[] = []
  for (const child of category.children.values()) {
    paths.push(child.path)
    paths.push(...getAllExpandablePaths(child))
  }
  return paths
}

function ensureRootPagesCategory(root: UrlCategory): UrlCategory {
  if (!root.children.has(ROOT_PAGES_KEY)) {
    root.children.set(ROOT_PAGES_KEY, {
      name: 'Main',
      path: ROOT_PAGES_KEY,
      urls: [],
      children: new Map(),
      level: root.level + 1,
    })
  }
  return root.children.get(ROOT_PAGES_KEY)!
}

function parsePathSegments(urlObj: URL): ParsedSegment[] {
  return (urlObj.pathname || '')
    .split('/')
    .map((seg) => seg.trim())
    .filter((seg) => seg.length > 0)
    .map((seg) => ({ key: normalizeSegmentKey(seg), label: segmentLabel(seg) }))
    .filter((seg) => seg.key.length > 0)
}

/**
 * Heuristic to identify non-meaningful path prefixes (ID-like/hash-like)
 * so grouping can happen on stable folders like /blog, /coupon, etc.
 */
function isLikelyDynamicSegment(segment: string): boolean {
  const s = segment.toLowerCase()
  if (!s) return true
  if (s.length >= 8 && /[a-z]/.test(s) && /\d/.test(s)) return true
  if (/^\d{4,}$/.test(s)) return true
  if (/^[a-f0-9]{12,}$/.test(s)) return true
  if (/^[a-z0-9_-]{20,}$/.test(s) && !s.includes('-')) return true
  if (/^(id|uid|pid|sid|gid|cid|rid)[-_]?[a-z0-9]{4,}$/.test(s)) return true
  return false
}

/** Skip noisy leading IDs, but always keep at least one segment. */
function stripNoisyLeadingSegments(segments: ParsedSegment[]): ParsedSegment[] {
  if (segments.length <= 1) return segments
  let start = 0
  while (start < segments.length - 1 && isLikelyDynamicSegment(segments[start].key)) {
    start += 1
  }
  return segments.slice(start)
}

/**
 * Group by folder-like path:
 * - single segment path (e.g. /pricing) -> group "pricing"
 * - multi-segment path (e.g. /blog/post-1) -> group "blog"
 * - deeper paths (e.g. /a/b/c) -> group "a/b"
 */
function getGroupingSegments(segments: ParsedSegment[]): ParsedSegment[] {
  if (segments.length === 0) return []
  if (segments.length === 1) return segments
  return segments.slice(0, -1)
}

function getOrCreateChild(parent: UrlCategory, segment: ParsedSegment): UrlCategory {
  const existing = parent.children.get(segment.key)
  if (existing) return existing
  const child: UrlCategory = {
    name: segment.label,
    path: parent.path ? `${parent.path}/${segment.key}` : segment.key,
    urls: [],
    children: new Map(),
    level: parent.level + 1,
  }
  parent.children.set(segment.key, child)
  return child
}

export function categorizeUrls(urls: string[], _baseUrl: string): UrlCategory {
  const root: UrlCategory = {
    name: 'Main',
    path: '',
    urls: [],
    children: new Map(),
    level: 0,
  }

  // Dedupe so https://example.com and https://example.com/ count as one
  const seen = new Set<string>()
  const deduped = urls.filter((u) => {
    const k = getNormalizedUrlKey(u)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })

  for (const url of deduped) {
    try {
      const urlObj = new URL(url)
      const path = urlObj.pathname || '/'
      if (path === '/' || path === '') {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      const parsed = parsePathSegments(urlObj)
      if (parsed.length === 0) {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      const meaningful = stripNoisyLeadingSegments(parsed)
      if (meaningful.length === 0) {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      const groupSegments = getGroupingSegments(meaningful)
      if (groupSegments.length === 0) {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      let current = root
      for (const segment of groupSegments) {
        current = getOrCreateChild(current, segment)
      }
      current.urls.push(url)
    } catch {
      continue
    }
  }

  return root
}

export function getAllUrlsFromCategory(category: UrlCategory): string[] {
  const urls: string[] = [...category.urls]
  for (const child of category.children.values()) {
    urls.push(...getAllUrlsFromCategory(child))
  }
  return urls
}

export function getCategoryUrlCount(category: UrlCategory): number {
  let count = category.urls.length
  for (const child of category.children.values()) {
    count += getCategoryUrlCount(child)
  }
  return count
}
