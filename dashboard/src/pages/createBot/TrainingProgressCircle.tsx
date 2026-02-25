/**
 * Circular progress bar using the same data as step 3 (training progress).
 * Change labels in trainingProgressLabels.ts and they update here too.
 */

import { getTrainingStageLabel } from './trainingProgressLabels'
import { UiCard } from '../../components/ui'
import { useTranslation } from 'react-i18next'

const SIZE = 130
const STROKE = 8
const RADIUS = (SIZE - STROKE) / 2
const CENTER = SIZE / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

type Props = {
  progress: number
  jobId: string | null
  trainingStage: string
  trainingStageName: string
  trainingPagesCrawled: number
  trainingDocsCount: number
}

export function TrainingProgressCircle({
  progress,
  jobId,
  trainingStage,
  trainingStageName,
  trainingPagesCrawled,
  trainingDocsCount,
}: Props) {
  const { t } = useTranslation()
  const label = getTrainingStageLabel(jobId, trainingStage, trainingStageName, t)
  const offset = CIRCUMFERENCE * (1 - progress / 100)

  return (
    <UiCard
      className="training-progress-circle"
      style={{
        padding: '20px',
        borderRadius: '14px',
        background: 'rgba(255, 255, 255, 0.12)',
        border: '1px solid rgba(255, 255, 255, 0.22)',
        boxShadow: 'none',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
        <div style={{ position: 'relative', width: SIZE, height: SIZE }}>
          <svg
            width={SIZE}
            height={SIZE}
            style={{ transform: 'rotate(-90deg)' }}
            aria-hidden
          >
            <defs>
              <linearGradient id="training-progress-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ffffff" />
                <stop offset="100%" stopColor="#f6b46d" />
              </linearGradient>
            </defs>
            <circle
              cx={CENTER}
              cy={CENTER}
              r={RADIUS}
              fill="none"
              stroke="rgba(255, 255, 255, 0.3)"
              strokeWidth={STROKE}
            />
            <circle
              cx={CENTER}
              cy={CENTER}
              r={RADIUS}
              fill="none"
              stroke="url(#training-progress-gradient)"
              strokeWidth={STROKE}
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={offset}
              style={{
                transition: 'stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1)',
                filter: 'drop-shadow(0 0 10px rgba(255, 255, 255, 0.35))',
              }}
            />
          </svg>
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: '1px',
            }}
          >
            <span
              style={{
                fontSize: '26px',
                fontWeight: 700,
                color: '#ffffff',
                lineHeight: 1,
              }}
            >
              {progress}%
            </span>
            <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.55)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {t('createBot.complete', 'complete')}
            </span>
          </div>
        </div>
        <div style={{ textAlign: 'center', minHeight: '40px' }}>
          <div style={{ fontWeight: 600, color: '#ffffff', marginBottom: '3px', fontSize: '0.85rem' }}>{label}</div>
          <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)' }}>
            {trainingPagesCrawled > 0 && (
              <span>{t('createBot.pagesCount', '{{count}} pages', { count: trainingPagesCrawled })}</span>
            )}
            {trainingPagesCrawled > 0 && trainingDocsCount > 0 && <span style={{ margin: '0 4px' }}>&middot;</span>}
            {trainingDocsCount > 0 && (
              <span>{t('createBot.docsCount', '{{count}} docs', { count: trainingDocsCount })}</span>
            )}
            {trainingPagesCrawled === 0 && trainingDocsCount === 0 && (
              <span>&mdash;</span>
            )}
          </div>
        </div>
      </div>
    </UiCard>
  )
}
