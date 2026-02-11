import type { ReactNode } from 'react'
import { motion } from 'framer-motion'

type AnimatedPageProps = {
  className?: string
  children: ReactNode
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function AnimatedPage({ className, children }: AnimatedPageProps) {
  return (
    <motion.div
      className={cx('animated-page', className)}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: { opacity: 0, y: 16 },
        visible: {
          opacity: 1,
          y: 0,
          transition: {
            duration: 0.42,
            ease: 'easeOut',
            when: 'beforeChildren',
            staggerChildren: 0.06,
          },
        },
      }}
    >
      {children}
    </motion.div>
  )
}
