export type UrlCategory = {
  name: string
  path: string
  urls: string[]
  children: Map<string, UrlCategory>
  level: number
}

/** Key for the "root-level pages" category so they're under one collapsible parent. */
const ROOT_PAGES_KEY = '\0root-pages'

/** Normalize URL for dedupe: path '' and '/' are the same (e.g. example.com and example.com/). */
function normalizeUrlKey(url: string): string {
  try {
    const u = new URL(url)
    const path = u.pathname
    if (path === '' || path === '/') {
      return u.origin + '/' + (u.search || '')
    }
    return url.split('#')[0]
  } catch {
    return url
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
    const k = normalizeUrlKey(u)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })

  for (const url of deduped) {
    try {
      const urlObj = new URL(url)
      const path = urlObj.pathname
      // All root-like URLs (path '' or '/', including with query) go under "/" so no URL is parentless
      if (path === '/' || path === '') {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      const segments = path.split('/').filter(s => s.length > 0)
      if (segments.length === 0) {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      // Single-segment paths (e.g. /about, /adherence-neutral) → under "/" as well
      if (segments.length === 1) {
        ensureRootPagesCategory(root).urls.push(url)
        continue
      }

      // Multi-segment paths: one parent per first segment (e.g. workouts, articles), URLs directly under it
      const firstSegment = segments[0]
      if (!root.children.has(firstSegment)) {
        root.children.set(firstSegment, {
          name: firstSegment,
          path: firstSegment,
          urls: [],
          children: new Map(),
          level: root.level + 1,
        })
      }
      root.children.get(firstSegment)!.urls.push(url)
    } catch (e) {
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
