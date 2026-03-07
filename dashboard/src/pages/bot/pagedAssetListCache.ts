const CACHE_TTL_MS = 2 * 60 * 1000
const CACHE_PREFIX = 'dashboard.paged_assets.'

export type PagedListCacheEntry<T> = {
  savedAt: number
  etag?: string
  payload: T
}

export function buildPagedListCacheKey(scope: string, botId: string, pageSize: number, offset: number): string {
  return `${CACHE_PREFIX}${scope}:${botId}:${pageSize}:${offset}`
}

export function readPagedListCache<T>(cacheKey: string): PagedListCacheEntry<T> | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = window.sessionStorage.getItem(cacheKey)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<PagedListCacheEntry<T>>
    const savedAt = Number(parsed.savedAt || 0)
    if (!savedAt || !parsed.payload) return null
    if (Date.now() - savedAt > CACHE_TTL_MS) {
      window.sessionStorage.removeItem(cacheKey)
      return null
    }
    return {
      savedAt,
      etag: typeof parsed.etag === 'string' ? parsed.etag : undefined,
      payload: parsed.payload as T,
    }
  } catch {
    return null
  }
}

export function writePagedListCache<T>(cacheKey: string, entry: PagedListCacheEntry<T>): void {
  if (typeof window === 'undefined') return
  try {
    window.sessionStorage.setItem(cacheKey, JSON.stringify(entry))
  } catch {
    // best-effort cache
  }
}

