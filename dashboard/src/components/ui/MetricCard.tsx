import type { ReactNode } from 'react'
import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'

type MetricCardProps = {
  className?: string
  label: string
  value: number
  icon?: ReactNode
  trend?: string
  suffix?: string
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function MetricCard({ className, label, value, icon, trend, suffix = '' }: MetricCardProps) {
  const [displayValue, setDisplayValue] = useState(0)
  const duration = 650
  const steps = useMemo(() => Math.max(12, Math.min(value, 40)), [value])

  useEffect(() => {
    const increment = value / steps
    const stepTime = Math.max(12, Math.floor(duration / steps))
    let current = 0

    const timer = window.setInterval(() => {
      current += increment
      if (current >= value) {
        setDisplayValue(value)
        window.clearInterval(timer)
        return
      }
      setDisplayValue(Math.round(current))
    }, stepTime)

    return () => window.clearInterval(timer)
  }, [steps, value])

  return (
    <motion.div
      className={cx('ui-metric-card', className)}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      whileHover={{ y: -4 }}
    >
      <div className="ui-metric-card-head">
        <span className="ui-metric-card-label">{label}</span>
        {icon ? <span className="ui-metric-card-icon">{icon}</span> : null}
      </div>
      <p className="ui-metric-card-value">
        {displayValue.toLocaleString()}
        {suffix}
      </p>
      {trend ? <span className="ui-metric-card-trend">{trend}</span> : null}
    </motion.div>
  )
}
