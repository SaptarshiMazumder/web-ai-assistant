const iconSize = 18
const strokeWidth = 2
// Soft line ends, not very rounded corners
const iconStroke = { strokeLinecap: 'round' as const, strokeLinejoin: 'miter' as const }

export function PlayIcon({ className }: { className?: string }) {
  return (
    <svg
      width={iconSize}
      height={iconSize}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      {...iconStroke}
      className={className}
      aria-hidden
    >
      <path d="M6 4v16l12-8L6 4z" />
    </svg>
  )
}

export function StopIcon({ className }: { className?: string }) {
  return (
    <svg
      width={iconSize}
      height={iconSize}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      {...iconStroke}
      className={className}
      aria-hidden
    >
      <rect x="6" y="6" width="12" height="12" />
    </svg>
  )
}
