import type { CSSProperties, ReactNode } from 'react'
import { motion } from 'framer-motion'

type GlassCardProps = {
  className?: string
  style?: CSSProperties
  children: ReactNode
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function GlassCard({ className, style, children }: GlassCardProps) {
  return (
    <motion.div
      className={cx('ui-glass-card', className)}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      style={style}
    >
      {children}
    </motion.div>
  )
}
