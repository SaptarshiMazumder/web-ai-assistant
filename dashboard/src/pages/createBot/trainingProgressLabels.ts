/**
 * Single source of truth for training progress labels.
 * Used by step 3 (Progress page) and step 4 (Design widget circular bar).
 * Change labels here and they update in both places.
 */

import type { TFunction } from 'i18next'

export const TRAINING_STAGE_LABELS: Record<string, { key: string; fallback: string }> = {
  queued: { key: 'createBot.trainingStageQueued', fallback: 'Starting' },
  crawling: { key: 'createBot.trainingStageCrawling', fallback: 'Reading your pages' },
  uploading: { key: 'createBot.trainingStageUploading', fallback: 'Saving what we found' },
  importing: { key: 'createBot.trainingStageImporting', fallback: 'Getting your agent ready' },
  prompt_queued: { key: 'createBot.trainingStagePromptQueued', fallback: 'Preparing your prompt' },
  prompt_generating: { key: 'createBot.trainingStagePromptGenerating', fallback: 'Generating your prompt' },
  import_submitted: { key: 'createBot.trainingStageImportSubmitted', fallback: 'Import submitted' },
  pipeline_queued: { key: 'createBot.trainingStagePipelineQueued', fallback: 'Automation queued' },
  pipeline_running: { key: 'createBot.trainingStagePipelineRunning', fallback: 'Finalizing training' },
  pipeline_resume_requested: { key: 'createBot.trainingStagePipelineResumeRequested', fallback: 'Resuming automation' },
  pipeline_paused: { key: 'createBot.trainingStagePipelinePaused', fallback: 'Waiting for manual input' },
  pipeline_error: { key: 'createBot.trainingStagePipelineError', fallback: 'Automation failed' },
  skipped: { key: 'createBot.trainingStageSkipped', fallback: 'Skipped (nothing added)' },
  done: { key: 'createBot.trainingStageDone', fallback: 'Done' },
  error: { key: 'createBot.trainingStageError', fallback: 'Error' },
}

export function getTrainingStageLabelByName(trainingStageName: string, t: TFunction): string {
  const entry = TRAINING_STAGE_LABELS[trainingStageName]
  if (!entry) return trainingStageName || t('createBot.trainingStageProcessing', 'Processing')
  return t(entry.key, entry.fallback)
}

export function getTrainingStageLabel(
  jobId: string | null,
  trainingStage: string,
  trainingStageName: string,
  t: TFunction,
  customMessage?: string | null
): string {
  // Use custom message if available (from pipeline)
  if (customMessage) {
    return customMessage
  }
  if (!jobId && trainingStage === 'training') {
    return t('createBot.trainingStageGettingStarted', 'Getting started...')
  }
  return getTrainingStageLabelByName(trainingStageName, t)
}
