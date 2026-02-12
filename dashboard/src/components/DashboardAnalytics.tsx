import React, { useEffect, useMemo, useState } from 'react'
import { useDashboardData, type AnalyticsSummary, type AnalyticsTimeseries, type TopSources, type Topics } from '../hooks/useDashboardData'
import { MetricCard, SectionHeader } from './ui'
import { AlertTriangle, MessageCircle, MessagesSquare, RotateCw, Smile, UserPlus } from 'lucide-react'

function toDayString(d: Date) {
  const yyyy = d.getFullYear()
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd}`
}

function daysAgo(n: number) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d
}

type PresetRange = '1d' | '7d' | '1m' | 'custom'

function resolvePreset(preset: PresetRange, custom: { fromDay: string; toDay: string }) {
  const today = toDayString(new Date())
  if (preset === '1d') return { from_day: today, to_day: today }
  if (preset === '7d') return { from_day: toDayString(daysAgo(6)), to_day: today }
  if (preset === '1m') return { from_day: toDayString(daysAgo(29)), to_day: today }
  return { from_day: custom.fromDay, to_day: custom.toDay }
}

function linePathSmooth(points: Array<{ x: number; y: number }>) {
  if (points.length <= 1) return ''
  const pts = points
  let d = `M ${pts[0].x} ${pts[0].y}`
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i]
    const p1 = pts[i]
    const p2 = pts[i + 1]
    const p3 = pts[i + 2] || p2

    // Catmull-Rom to cubic Bezier conversion
    const c1x = p1.x + (p2.x - p0.x) / 6
    const c1y = p1.y + (p2.y - p0.y) / 6
    const c2x = p2.x - (p3.x - p1.x) / 6
    const c2y = p2.y - (p3.y - p1.y) / 6
    d += ` C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2.x} ${p2.y}`
  }
  return d
}

const CHART_PAD_LEFT = 36
const CHART_PAD_RIGHT = 12
const CHART_PAD_TOP = 8
const CHART_PAD_BOTTOM = 28
const CHART_HEIGHT = 320

function LineChartWithAxes({
  labels,
  values,
  stroke,
}: {
  labels: string[]
  values: number[]
  stroke: string
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const CHART_WIDTH = 500
  if (!values.length || values.length !== labels.length) return null
  const max = Math.max(...values, 1)
  const min = Math.min(...values, 0)
  const range = max - min || 1
  const plotW = CHART_WIDTH - CHART_PAD_LEFT - CHART_PAD_RIGHT
  const plotH = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM
  const pts = values.map((v, i) => {
    const x = CHART_PAD_LEFT + (i * plotW) / Math.max(values.length - 1, 1)
    const y = CHART_PAD_TOP + ((max - v) / range) * plotH
    return { x, y }
  })
  const d = linePathSmooth(pts)

  const yTicks = 5
  const yTickValues: number[] = []
  for (let i = 0; i <= yTicks; i++) {
    yTickValues.push(Math.round(min + (range * i) / yTicks))
  }

  const xStep = Math.max(1, Math.floor(labels.length / 6))
  const xTickIndices: number[] = []
  for (let i = 0; i < labels.length; i += xStep) xTickIndices.push(i)
  if (labels.length > 0 && xTickIndices[xTickIndices.length - 1] !== labels.length - 1) {
    xTickIndices.push(labels.length - 1)
  }

  return (
    <div style={{ flex: 1, minHeight: CHART_HEIGHT, width: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} width="100%" height="100%" style={{ display: 'block', minHeight: CHART_HEIGHT }} preserveAspectRatio="xMidYMid meet">
      {yTickValues.map((val) => {
        const y = CHART_PAD_TOP + ((max - val) / range) * plotH
        return (
          <g key={val}>
            <line x1={CHART_PAD_LEFT} y1={y} x2={CHART_PAD_LEFT + plotW} y2={y} stroke="#e2e8f0" strokeWidth="1" strokeDasharray="2,2" />
            <text x={CHART_PAD_LEFT - 6} y={y} textAnchor="end" dominantBaseline="middle" fontSize="9" fill="#64748b">{val}</text>
          </g>
        )
      })}
      {xTickIndices.map((idx) => {
        const x = CHART_PAD_LEFT + (idx * plotW) / Math.max(labels.length - 1, 1)
        const label = labels[idx]
        const short = label ? label.slice(5) : '' // MM-DD
        return (
          <text key={idx} x={x} y={CHART_HEIGHT - 8} textAnchor="middle" fontSize="9" fill="#64748b">{short}</text>
        )
      })}
      <path d={d} fill="none" stroke={stroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {/* Interactive clickable points */}
      {pts.map((pt, idx) => (
        <g key={idx}>
          {/* Invisible larger hit area for easier clicking */}
          <circle
            cx={pt.x}
            cy={pt.y}
            r={12}
            fill="transparent"
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => setHoveredIndex(idx)}
            onMouseLeave={() => setHoveredIndex(null)}
            onClick={() => setHoveredIndex(hoveredIndex === idx ? null : idx)}
          />
          {/* Visible point */}
          <circle
            cx={pt.x}
            cy={pt.y}
            r={hoveredIndex === idx ? 6 : 4}
            fill={hoveredIndex === idx ? stroke : 'white'}
            stroke={stroke}
            strokeWidth="2"
            style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
          />
          {/* Tooltip */}
          {hoveredIndex === idx && (
            <g>
              <rect
                x={pt.x - 35}
                y={pt.y - 38}
                width={70}
                height={28}
                rx={4}
                fill="#1e293b"
              />
              <text x={pt.x} y={pt.y - 27} textAnchor="middle" fontSize="10" fill="#94a3b8">
                {labels[idx]?.slice(5) || ''}
              </text>
              <text x={pt.x} y={pt.y - 15} textAnchor="middle" fontSize="12" fontWeight="600" fill="white">
                {values[idx]}
              </text>
            </g>
          )}
        </g>
      ))}
    </svg>
    </div>
  )
}

function RangeControls({
  preset,
  setPreset,
  custom,
  setCustom,
}: {
  preset: PresetRange
  setPreset: (p: PresetRange) => void
  custom: { fromDay: string; toDay: string }
  setCustom: (v: { fromDay: string; toDay: string }) => void
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [showCustomPicker, setShowCustomPicker] = useState(false)
  const dropdownRef = React.useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
        setShowCustomPicker(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const presetLabels: Record<PresetRange, string> = {
    '1d': 'Today',
    '7d': 'Last 7 days',
    '1m': 'Last 30 days',
    'custom': 'Custom range',
  }

  const handlePresetSelect = (p: PresetRange) => {
    if (p === 'custom') {
      setShowCustomPicker(true)
    } else {
      setPreset(p)
      setIsOpen(false)
      setShowCustomPicker(false)
    }
  }

  const applyCustomRange = () => {
    setPreset('custom')
    setIsOpen(false)
    setShowCustomPicker(false)
  }

  return (
    <div className="range-dropdown" ref={dropdownRef}>
      <button
        type="button"
        className="range-dropdown-trigger"
        onClick={() => {
          setIsOpen(!isOpen)
          if (!isOpen) setShowCustomPicker(preset === 'custom')
        }}
      >
        <span>{presetLabels[preset]}</span>
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: 6 }}>
          <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      {isOpen && (
        <div className="range-dropdown-menu">
          {!showCustomPicker ? (
            <>
              {(['1d', '7d', '1m'] as const).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={`range-dropdown-item ${preset === id ? 'active' : ''}`}
                  onClick={() => handlePresetSelect(id)}
                >
                  {presetLabels[id]}
                </button>
              ))}
              <button
                type="button"
                className={`range-dropdown-item ${preset === 'custom' ? 'active' : ''}`}
                onClick={() => handlePresetSelect('custom')}
              >
                {presetLabels['custom']}
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: 'auto' }}>
                  <path d="M4.5 3L7.5 6L4.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </>
          ) : (
            <div className="range-dropdown-custom">
              <button
                type="button"
                className="range-dropdown-back"
                onClick={() => setShowCustomPicker(false)}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M7.5 9L4.5 6L7.5 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span>Back</span>
              </button>
              <div className="range-dropdown-custom-fields">
                <label>
                  <span>From</span>
                  <input 
                    type="date" 
                    value={custom.fromDay} 
                    onChange={(e) => setCustom({ ...custom, fromDay: e.target.value })} 
                  />
                </label>
                <label>
                  <span>To</span>
                  <input 
                    type="date" 
                    value={custom.toDay} 
                    onChange={(e) => setCustom({ ...custom, toDay: e.target.value })} 
                  />
                </label>
              </div>
              <button
                type="button"
                className="range-dropdown-apply"
                onClick={applyCustomRange}
              >
                Apply
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/** Small pie chart: positive (green) vs negative (red) CSAT */
function CSATPie({ positive, negative }: { positive: number; negative: number }) {
  const total = positive + negative
  const size = 80
  const r = 32
  const cx = size / 2
  const cy = size / 2
  if (total === 0) {
    return null // Return null when empty, will show "0" in the metric value instead
  }
  const posAngle = (positive / total) * 360
  const negAngle = 360 - posAngle
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const x1 = cx + r * Math.cos(toRad(-90))
  const y1 = cy + r * Math.sin(toRad(-90))
  const x2 = cx + r * Math.cos(toRad(-90 + posAngle))
  const y2 = cy + r * Math.sin(toRad(-90 + posAngle))
  const large1 = posAngle > 180 ? 1 : 0
  const large2 = negAngle > 180 ? 1 : 0
  const posPath = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large1} 1 ${x2} ${y2} Z`
  const x3 = cx + r * Math.cos(toRad(-90 + posAngle))
  const y3 = cy + r * Math.sin(toRad(-90 + posAngle))
  const x4 = cx + r * Math.cos(toRad(-90 + 360))
  const y4 = cy + r * Math.sin(toRad(-90 + 360))
  const negPath = `M ${cx} ${cy} L ${x3} ${y3} A ${r} ${r} 0 ${large2} 1 ${x4} ${y4} Z`
  return (
    <div className="summary-feedback-pie" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={posPath} fill="#22c55e" />
        <path d={negPath} fill="#ef4444" />
      </svg>
      <div className="summary-feedback-legend">
        <span className="summary-feedback-legend-pos">+{positive}</span>
        <span className="summary-feedback-legend-neg">−{negative}</span>
      </div>
    </div>
  )
}

type Props = { botId: string | null; setupPills?: React.ReactNode }

type DashboardAnalyticsCacheEntry = {
  updatedAt: number
  summary: AnalyticsSummary | null
  unresolvedEscalations: number | null
  convSeries: AnalyticsTimeseries | null
  escSeries: AnalyticsTimeseries | null
  sources: TopSources | null
  topics: Topics | null
}

const ANALYTICS_STALE_MS = 45_000
const DASHBOARD_ANALYTICS_CACHE = new Map<string, DashboardAnalyticsCacheEntry>()

export default function DashboardAnalytics({ botId, setupPills }: Props) {
  const { getAnalyticsSummary, getAnalyticsTimeseries, getAnalyticsTopSources, getAnalyticsTopics, getEscalationCounts, recomputeAnalytics } =
    useDashboardData()
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [unresolvedEscalations, setUnresolvedEscalations] = useState<number | null>(null)
  const [convSeries, setConvSeries] = useState<AnalyticsTimeseries | null>(null)
  const [escSeries, setEscSeries] = useState<AnalyticsTimeseries | null>(null)
  const [sources, setSources] = useState<TopSources | null>(null)
  const [topics, setTopics] = useState<Topics | null>(null)

  const [convPreset, setConvPreset] = useState<PresetRange>('7d')
  const [convCustom, setConvCustom] = useState<{ fromDay: string; toDay: string }>(() => ({
    fromDay: toDayString(daysAgo(29)),
    toDay: toDayString(new Date()),
  }))

  const [escPreset, setEscPreset] = useState<PresetRange>('7d')
  const [escCustom, setEscCustom] = useState<{ fromDay: string; toDay: string }>(() => ({
    fromDay: toDayString(daysAgo(29)),
    toDay: toDayString(new Date()),
  }))

  useEffect(() => {
    if (!botId) return
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botId, convPreset, convCustom.fromDay, convCustom.toDay, escPreset, escCustom.fromDay, escCustom.toDay])

  async function load(options?: { recompute?: boolean; forceNetwork?: boolean }) {
    if (!botId) return
    const convWindow = resolvePreset(convPreset, convCustom)
    const escWindow = resolvePreset(escPreset, escCustom)
    const cacheKey = [
      botId,
      convPreset,
      convWindow.from_day,
      convWindow.to_day,
      escPreset,
      escWindow.from_day,
      escWindow.to_day,
    ].join('|')

    const cached = DASHBOARD_ANALYTICS_CACHE.get(cacheKey)
    if (cached) {
      setSummary(cached.summary)
      setConvSeries(cached.convSeries)
      setEscSeries(cached.escSeries)
      setSources(cached.sources)
      setTopics(cached.topics)
      setUnresolvedEscalations(cached.unresolvedEscalations)

      const isFresh = Date.now() - cached.updatedAt < ANALYTICS_STALE_MS
      if (!options?.recompute && !options?.forceNetwork && isFresh) {
        return
      }
    }

    setLoading(options?.recompute === true || !cached)
    try {
      if (options?.recompute) {
        const recomputeStart = convWindow.from_day <= escWindow.from_day ? convWindow.from_day : escWindow.from_day
        const recomputeEnd = convWindow.to_day >= escWindow.to_day ? convWindow.to_day : escWindow.to_day
        await recomputeAnalytics(botId, { from_day: recomputeStart, to_day: recomputeEnd })
      }

      const [s, ts, src, t, counts] = await Promise.all([
        getAnalyticsSummary(botId, { from_day: convWindow.from_day, to_day: convWindow.to_day }),
        getAnalyticsTimeseries(botId, { from_day: convWindow.from_day, to_day: convWindow.to_day }),
        getAnalyticsTopSources(botId, { from_day: convWindow.from_day, to_day: convWindow.to_day, limit: 10 }),
        getAnalyticsTopics(botId, { from_day: convWindow.from_day, to_day: convWindow.to_day, limit: 20 }),
        getEscalationCounts(botId),
      ])
      setSummary(s)
      setConvSeries(ts)
      setSources(src)
      setTopics(t)
      setUnresolvedEscalations(counts?.open ?? 0)

      const escTs = await getAnalyticsTimeseries(botId, { from_day: escWindow.from_day, to_day: escWindow.to_day })
      setEscSeries(escTs)

      DASHBOARD_ANALYTICS_CACHE.set(cacheKey, {
        updatedAt: Date.now(),
        summary: s,
        unresolvedEscalations: counts?.open ?? 0,
        convSeries: ts,
        escSeries: escTs,
        sources: src,
        topics: t,
      })
    } finally {
      setLoading(false)
    }
  }

  const convDays = useMemo(() => (convSeries?.points || []).map((p) => p.day), [convSeries])
  const convValues = useMemo(() => (convSeries?.points || []).map((p) => p.conversations || 0), [convSeries])
  const escDays = useMemo(() => (escSeries?.points || []).map((p) => p.day), [escSeries])
  const escValues = useMemo(() => (escSeries?.points || []).map((p) => p.escalations || 0), [escSeries])

  async function handleRefreshSummary() {
    if (!botId || refreshing || loading) return
    setRefreshing(true)
    try {
      await load({ recompute: true, forceNetwork: true })
    } finally {
      setRefreshing(false)
    }
  }

  if (!botId) {
    return <div className="empty-panel">Create or select a bot to view analytics.</div>
  }

  return (
    <>
      {loading && (
        <div className="loading-overlay" aria-hidden="true">
          <div className="loading-overlay__spinner" />
        </div>
      )}

      <section className="ui-glass-card summary-card" style={{ marginTop: 0 }}>
        <div className="summary-card-header">
          <SectionHeader
            eyebrow="Performance"
            title="Summary"
            subtitle="Live metrics and setup completion for your bot."
            titleAccessory={
              <button
                type="button"
                onClick={() => void handleRefreshSummary()}
                disabled={refreshing || loading}
                aria-label="Refresh summary stats"
                title="Recompute and refresh summary stats"
                className={`summary-refresh-icon-btn${refreshing || loading ? ' is-spinning' : ''}`}
              >
                <RotateCw size={14} strokeWidth={2.25} />
              </button>
            }
          />
        </div>
        {setupPills != null && (
          <div className="summary-pills">
            {setupPills}
          </div>
        )}
        <div className="summary-metrics">
          <MetricCard label="Total conversations" value={summary?.conversations ?? 0} icon={<MessagesSquare size={15} />} />
          <MetricCard label="Leads captured" value={summary?.escalations ?? 0} icon={<UserPlus size={15} />} />
          <MetricCard label="Messages / Conv" value={Number((summary?.messages_per_conversation ?? 0).toFixed(1))} icon={<MessageCircle size={15} />} />
          <MetricCard label="Unresolved escalations" value={unresolvedEscalations ?? 0} icon={<AlertTriangle size={15} />} />
          <div className="ui-metric-card summary-metric-card--feedback">
            <div className="ui-metric-card-head">
              <span className="ui-metric-card-label">CSAT</span>
              <span className="ui-metric-card-icon"><Smile size={15} /></span>
            </div>
            {((summary?.positive_feedback ?? 0) + (summary?.negative_feedback ?? 0)) === 0 ? (
              <span className="ui-metric-card-value">0</span>
            ) : (
              <CSATPie
                positive={summary?.positive_feedback ?? 0}
                negative={summary?.negative_feedback ?? 0}
              />
            )}
          </div>
        </div>
      </section>

      <div className="card-grid" style={{ marginTop: 12 }}>
        <section className="ui-glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div className="card-title" style={{ margin: 0 }}>Daily conversations</div>
            <RangeControls preset={convPreset} setPreset={setConvPreset} custom={convCustom} setCustom={setConvCustom} />
          </div>
          <div style={{ marginTop: 8, flex: 1, minHeight: CHART_HEIGHT, display: 'flex', flexDirection: 'column', width: '100%' }}>
            {!convValues.length && <div className="muted">No data yet.</div>}
            {!!convValues.length && (
              <LineChartWithAxes
                labels={convDays}
                values={convValues}
                stroke="#e66397"
              />
            )}
          </div>
        </section>
        <section className="ui-glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div className="card-title" style={{ margin: 0 }}>Daily escalations</div>
            <RangeControls preset={escPreset} setPreset={setEscPreset} custom={escCustom} setCustom={setEscCustom} />
          </div>
          <div style={{ marginTop: 8, flex: 1, minHeight: CHART_HEIGHT, display: 'flex', flexDirection: 'column', width: '100%' }}>
            {!escValues.length && <div className="muted">No data yet.</div>}
            {!!escValues.length && (
              <LineChartWithAxes
                labels={escDays}
                values={escValues}
                stroke="#f0806b"
              />
            )}
          </div>
        </section>
      </div>

      <div className="card-grid" style={{ marginTop: 12 }}>
        <section className="ui-glass-card analytics-list-card">
          <div className="card-title">Top sources</div>
          <div className="analytics-list-scroll">
            {!sources?.items?.length && <div className="muted">No data yet.</div>}
            {!!sources?.items?.length && (
              <div>
                {sources.items.map((s) => (
                  <div key={s.source_url} className="detail-row">
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.source_url}</span>
                    <span>{s.count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
        <section className="ui-glass-card analytics-list-card">
          <div className="card-title">Question topic report</div>
          <div className="analytics-list-scroll">
            {!topics?.items?.length && <div className="muted">No data yet.</div>}
            {!!topics?.items?.length && (
              <div>
                {topics.items.slice(0, 12).map((t) => (
                  <div key={t.topic} className="detail-row">
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.topic}</span>
                    <span>{t.count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  )
}
