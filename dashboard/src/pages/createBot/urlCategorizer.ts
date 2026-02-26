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

/** Normalize URL key for dedupe/mapping: ignores trailing slash differences on non-root paths. */
export function getNormalizedUrlKey(url: string): string {
  try {
    const u = new URL(url)
    const rawPath = (u.pathname || '') === '' ? '/' : u.pathname
    const normalizedPath = rawPath === '/' ? '/' : (rawPath.replace(/\/+$/, '') || '/')
    if (normalizedPath === '/') return `${u.origin}/${u.search || ''}`
    return `${u.origin}${normalizedPath}${u.search || ''}`
  } catch {
    return (url || '').trim()
  }
}

function hasPathTrailingSlash(url: string): boolean {
  try {
    return new URL(url).pathname.endsWith('/')
  } catch {
    return (url || '').trim().endsWith('/')
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
 * If URL shares the full base path prefix, anchor grouping at the base page leaf.
 * Example:
 *   base: /tokyo/A1304/A130401/13224546
 *   url : /tokyo/A1304/A130401/13224546/peripheral_map
 *   -> start from 13224546/peripheral_map
 */
function anchorSegmentsByBasePath(
  urlObj: URL,
  segments: ParsedSegment[],
  baseUrlObj: URL | null,
  baseSegments: ParsedSegment[]
): { segments: ParsedSegment[]; anchored: boolean } {
  if (!baseUrlObj || baseSegments.length === 0) return { segments, anchored: false }
  if (urlObj.origin.toLowerCase() !== baseUrlObj.origin.toLowerCase()) return { segments, anchored: false }
  if (segments.length < baseSegments.length) return { segments, anchored: false }

  for (let i = 0; i < baseSegments.length; i += 1) {
    if (segments[i]?.key !== baseSegments[i]?.key) {
      return { segments, anchored: false }
    }
  }

  const anchorStart = Math.max(0, baseSegments.length - 1)
  return { segments: segments.slice(anchorStart), anchored: true }
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
  let baseUrlObj: URL | null = null
  let baseSegments: ParsedSegment[] = []
  try {
    baseUrlObj = new URL(_baseUrl)
    baseSegments = parsePathSegments(baseUrlObj)
  } catch {
    baseUrlObj = null
    baseSegments = []
  }

  const root: UrlCategory = {
    name: 'Main',
    path: '',
    urls: [],
    children: new Map(),
    level: 0,
  }

  // Dedupe equivalent URLs (including slash/no-slash variants); prefer trailing-slash form.
  const dedupedByKey = new Map<string, string>()
  const dedupeOrder: string[] = []
  for (const u of urls) {
    const k = getNormalizedUrlKey(u)
    if (!k) continue
    const existing = dedupedByKey.get(k)
    if (!existing) {
      dedupedByKey.set(k, u)
      dedupeOrder.push(k)
      continue
    }
    if (!hasPathTrailingSlash(existing) && hasPathTrailingSlash(u)) {
      dedupedByKey.set(k, u)
    }
  }
  const deduped = dedupeOrder
    .map((k) => dedupedByKey.get(k))
    .filter((u): u is string => !!u)

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

      const { segments: baseAnchored, anchored } = anchorSegmentsByBasePath(urlObj, parsed, baseUrlObj, baseSegments)
      const meaningful = anchored ? baseAnchored : stripNoisyLeadingSegments(baseAnchored)
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
