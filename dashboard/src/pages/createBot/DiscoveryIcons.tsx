import { FlowIcon } from '../../components/FlowIcon'

export function PlayIcon({ className }: { className?: string }) {
  return <FlowIcon name="play_arrow" filled size="sm" className={className || ''} />
}

export function StopIcon({ className }: { className?: string }) {
  return <FlowIcon name="stop" filled size="sm" className={className || ''} />
}
