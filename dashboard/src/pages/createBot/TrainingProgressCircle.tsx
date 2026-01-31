/**
 * Circular progress bar using the same data as step 3 (training progress).
 * Change labels in trainingProgressLabels.ts and they update here too.
 */

import { getTrainingStageLabel } from './trainingProgressLabels'

const SIZE = 140
const STROKE = 10
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
  const label = getTrainingStageLabel(jobId, trainingStage, trainingStageName)
  const offset = CIRCUMFERENCE * (1 - progress / 100)

  return (
    <div
      className="training-progress-circle"
      style={{
        padding: '24px',
        borderRadius: '16px',
        border: '1px solid #e2e8f0',
        background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
        boxShadow: '0 4px 20px rgba(99, 102, 241, 0.08)',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
        <div style={{ position: 'relative', width: SIZE, height: SIZE }}>
          <svg
            width={SIZE}
            height={SIZE}
            style={{ transform: 'rotate(-90deg)' }}
            aria-hidden
          >
            <defs>
              <linearGradient id="training-progress-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
            <circle
              cx={CENTER}
              cy={CENTER}
              r={RADIUS}
              fill="none"
              stroke="#e2e8f0"
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
              style={{ transition: 'stroke-dashoffset 0.5s ease' }}
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
              gap: '2px',
            }}
          >
            <span
              style={{
                fontSize: '28px',
                fontWeight: 700,
                color: '#334155',
                lineHeight: 1,
              }}
            >
              {progress}%
            </span>
            <span style={{ fontSize: '11px', color: '#64748b', fontWeight: 500 }}>complete</span>
          </div>
        </div>
        <div style={{ textAlign: 'center', minHeight: '48px' }}>
          <div style={{ fontWeight: 600, color: '#334155', marginBottom: '4px' }}>{label}</div>
          <div style={{ fontSize: '14px', color: '#64748b' }}>
            {trainingPagesCrawled > 0 && <span>{trainingPagesCrawled} pages crawled</span>}
            {trainingPagesCrawled > 0 && trainingDocsCount > 0 && ' • '}
            {trainingDocsCount > 0 && <span>{trainingDocsCount} documents</span>}
            {trainingPagesCrawled === 0 && trainingDocsCount === 0 && (
              <span style={{ color: '#94a3b8' }}>—</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
