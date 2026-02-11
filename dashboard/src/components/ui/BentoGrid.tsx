import type { ReactNode } from 'react'
import { motion } from 'framer-motion'

type BentoGridProps = {
  className?: string
  children: ReactNode
}

type BentoItemProps = {
  className?: string
  children: ReactNode
  colSpan?: 1 | 2 | 3
  rowSpan?: 1 | 2
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function BentoGrid({ className, children }: BentoGridProps) {
  return (
    <motion.div
      className={cx('ui-bento-grid', className)}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: {
          transition: {
            staggerChildren: 0.08,
          },
        },
      }}
    >
      {children}
    </motion.div>
  )
}

export function BentoItem({ className, children, colSpan = 1, rowSpan = 1 }: BentoItemProps) {
  return (
    <motion.div
      className={cx(
        'ui-bento-item',
        `ui-bento-item--col-${colSpan}`,
        `ui-bento-item--row-${rowSpan}`,
        className,
      )}
      variants={{
        hidden: { opacity: 0, y: 18 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.36, ease: 'easeOut' } },
      }}
    >
      {children}
    </motion.div>
  )
}
