/**
 * Single source of truth for training progress labels.
 * Used by step 3 (Progress page) and step 4 (Design widget circular bar).
 * Change labels here and they update in both places.
 */

export const TRAINING_STAGE_LABELS: Record<string, string> = {
  queued: 'Starting',
  crawling: 'Reading your pages',
  uploading: 'Saving what we found',
  importing: 'Getting your agent ready',
  import_submitted: 'Almost done',
  skipped: 'Skipped (nothing added)',
  done: 'Done',
  error: 'Error',
}

export function getTrainingStageLabel(
  jobId: string | null,
  trainingStage: string,
  trainingStageName: string
): string {
  if (!jobId && trainingStage === 'training') {
    return 'Getting started…'
  }
  return TRAINING_STAGE_LABELS[trainingStageName] || trainingStageName || 'Processing'
}
