/**
 * Single source of truth for training progress labels.
 * Used by step 3 (Progress page) and step 4 (Design widget circular bar).
 * Change labels here and they update in both places.
 */

export const TRAINING_STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  crawling: 'Crawling pages',
  uploading: 'Uploading to storage',
  importing: 'Importing to knowledge base',
  import_submitted: 'Import submitted',
  done: 'Complete',
  error: 'Error',
}

export function getTrainingStageLabel(
  jobId: string | null,
  trainingStage: string,
  trainingStageName: string
): string {
  if (!jobId && trainingStage === 'training') {
    return 'Starting crawl…'
  }
  return TRAINING_STAGE_LABELS[trainingStageName] || trainingStageName || 'Processing'
}
