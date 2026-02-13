import { useState } from 'react'
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { motion } from 'framer-motion'
import type { TopicUsageItem } from '../hooks/useDashboardData'

// ─── Color palette ───────────────────────────────────────────────────────────

const DYNAMIC_CATEGORY_PALETTE = [
  '#0ea5e9', '#22c55e', '#f97316', '#a855f7', '#ec4899',
  '#14b8a6', '#f59e0b', '#6366f1', '#ef4444', '#84cc16',
  '#06b6d4', '#8b5cf6',
]

function categoryColorHash(category: string): string {
  let hash = 0
  for (let i = 0; i < category.length; i += 1) {
    hash = (hash * 31 + category.charCodeAt(i)) >>> 0
  }
  return DYNAMIC_CATEGORY_PALETTE[hash % DYNAMIC_CATEGORY_PALETTE.length]
}

const CATEGORY_COLORS: Record<string, string> = {
  product: '#6366f1',
  pricing: '#22c55e',
  shipping: '#f59e0b',
  support: '#3b82f6',
  policy: '#8b5cf6',
  location: '#ec4899',
  hours: '#14b8a6',
  contact: '#f97316',
  faq: '#06b6d4',
  event: '#a855f7',
  other: '#64748b',
}

function getCategoryColor(category: string | null | undefined): string {
  if (!category) return CATEGORY_COLORS.other
  const key = category.toLowerCase()
  return CATEGORY_COLORS[key] ?? categoryColorHash(key)
}

// ─── Custom tooltip ───────────────────────────────────────────────────────────

interface TooltipProps {
  active?: boolean
  payload?: Array<{ payload: TopicUsageItem & { pct: string } }>
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null
  const item = payload[0].payload
  return (
    <div className="donut-tooltip">
      <div className="donut-tooltip-label">{capitalize(item.topic)}</div>
      <div className="donut-tooltip-meta">
        <span className="donut-tooltip-count">{item.question_count} questions</span>
        <span className="donut-tooltip-pct">{item.pct}%</span>
      </div>
      {item.source_url && (
        <a
          href={item.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="donut-tooltip-link"
          onClick={(e) => e.stopPropagation()}
        >
          View source ↗
        </a>
      )}
    </div>
  )
}

// ─── Custom center label ──────────────────────────────────────────────────────

interface CenterLabelProps {
  viewBox?: { cx: number; cy: number }
  totalQuestions: number
}

function CenterLabel({ viewBox, totalQuestions }: CenterLabelProps) {
  const cx = viewBox?.cx ?? 0
  const cy = viewBox?.cy ?? 0
  return (
    <>
      <text
        x={cx}
        y={cy - 8}
        textAnchor="middle"
        className="donut-center-number"
        fill="currentColor"
      >
        {totalQuestions}
      </text>
      <text
        x={cx}
        y={cy + 14}
        textAnchor="middle"
        className="donut-center-label"
        fill="currentColor"
      >
        questions
      </text>
    </>
  )
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function capitalize(s: string) {
  if (!s) return s
  return s.charAt(0).toUpperCase() + s.slice(1)
}

// ─── Legend ───────────────────────────────────────────────────────────────────

interface LegendProps {
  topics: Array<TopicUsageItem & { color: string; pct: string }>
  selectedId: string | null
  onSelect: (id: string | null) => void
}

function DonutLegend({ topics, selectedId, onSelect }: LegendProps) {
  return (
    <div className="donut-legend">
      {topics.map((t) => (
        <button
          key={t.topic_id}
          className={`donut-legend-item${selectedId === t.topic_id ? ' selected' : ''}`}
          onClick={() => onSelect(selectedId === t.topic_id ? null : t.topic_id)}
          type="button"
        >
          <span className="donut-legend-dot" style={{ background: t.color }} />
          <span className="donut-legend-name">
            {t.source_url ? (
              <a
                href={t.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="donut-legend-link"
                onClick={(e) => e.stopPropagation()}
              >
                {capitalize(t.topic)}
              </a>
            ) : (
              capitalize(t.topic)
            )}
          </span>
          <span className="donut-legend-count">{t.question_count}</span>
        </button>
      ))}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

interface TopicDonutChartProps {
  topics: TopicUsageItem[]
  totalQuestions: number
  onTopicClick?: (topicId: string) => void
}

export function TopicDonutChart({ topics, totalQuestions, onTopicClick }: TopicDonutChartProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Only show topics that have at least 1 question, or all if none do (so chart isn't blank)
  const hasActivity = topics.some((t) => t.question_count > 0)
  const displayTopics = hasActivity
    ? topics.filter((t) => t.question_count > 0)
    : topics.slice(0, 10)

  const enriched = displayTopics.map((t) => ({
    ...t,
    color: getCategoryColor(t.category),
    pct:
      totalQuestions > 0
        ? ((t.question_count / totalQuestions) * 100).toFixed(1)
        : '0',
    // recharts uses `value` for pie sizing
    value: hasActivity ? t.question_count : 1,
  }))

  const handleCellClick = (topicId: string) => {
    setSelectedId((prev) => (prev === topicId ? null : topicId))
    onTopicClick?.(topicId)
  }

  return (
    <motion.div
      className="donut-chart-wrapper"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
    >
      <div className="donut-chart-container">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie
              data={enriched}
              cx="50%"
              cy="50%"
              innerRadius={68}
              outerRadius={105}
              paddingAngle={3}
              cornerRadius={6}
              dataKey="value"
              animationBegin={0}
              animationDuration={600}
              labelLine={false}
            >
              {enriched.map((entry) => (
                <Cell
                  key={entry.topic_id}
                  fill={entry.color}
                  opacity={selectedId && selectedId !== entry.topic_id ? 0.35 : 1}
                  stroke="transparent"
                  style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
                  onClick={() => handleCellClick(entry.topic_id)}
                />
              ))}
              <CenterLabel
                // @ts-ignore recharts passes viewBox via label prop
                totalQuestions={hasActivity ? totalQuestions : topics.length}
              />
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <DonutLegend
        topics={enriched}
        selectedId={selectedId}
        onSelect={(id) => {
          setSelectedId(id)
          if (id) onTopicClick?.(id)
        }}
      />
    </motion.div>
  )
}
