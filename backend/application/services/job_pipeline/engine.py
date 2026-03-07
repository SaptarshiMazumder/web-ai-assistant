from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from domain.entities import JobPipelineRun, JobPipelineStep, JobPipelineStepEvent
from domain.interfaces import JobResult
from domain.platform_profiles import (
    get_job_pipeline_config,
    get_job_pipeline_default_failure_policy,
    get_job_pipeline_gates,
    get_job_pipeline_jobs,
    get_job_pipeline_workflow,
)
from domain.repositories import BotRepository, JobPipelineRepository

from .completion import CompletionCheckerRegistry
from .registry import JobRunnerRegistry

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return "jpr_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _new_event_id() -> str:
    return "jpe_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


def _clamp_pct(value: Any) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        iv = 0
    return 0 if iv < 0 else (100 if iv > 100 else iv)


def _localized_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        en = str(raw.get("en") or "").strip()
        if en:
            return en
        ja = str(raw.get("ja") or "").strip()
        if ja:
            return ja
    return ""


class JobPipelineEngine:
    """Sequential pipeline engine: queued -> running -> paused -> running -> done|error."""

    def __init__(
        self,
        *,
        pipeline_repo: JobPipelineRepository,
        bot_repo: BotRepository,
        runner_registry: JobRunnerRegistry,
        completion_registry: Optional[CompletionCheckerRegistry] = None,
    ) -> None:
        self._pipeline_repo = pipeline_repo
        self._bot_repo = bot_repo
        self._runner_registry = runner_registry
        self._completion_registry = completion_registry or CompletionCheckerRegistry()

    def start(
        self,
        *,
        bot_id: str,
        workflow_id: str = "default",
        trigger: str = "post_crawl",
        context: Optional[Dict[str, Any]] = None,
        execute_now: bool = True,
    ) -> Dict[str, Any]:
        bot = self._bot_repo.get_bot(bot_id)
        if not bot:
            raise ValueError(f"Unknown bot_id: {bot_id}")
        widget_config: Dict[str, Any] = {}
        raw_widget = getattr(bot, "widget_config", None)
        if raw_widget and str(raw_widget).strip():
            try:
                import json

                parsed = json.loads(raw_widget)
                if isinstance(parsed, dict):
                    widget_config = parsed
            except Exception:
                widget_config = {}

        jobs_catalog = get_job_pipeline_jobs()
        workflow_steps = get_job_pipeline_workflow(widget_config, workflow_id=workflow_id)
        now = _utc_now()
        run = JobPipelineRun(
            run_id=_new_run_id(),
            org_id=bot.org_id,
            bot_id=bot_id,
            workflow_id=workflow_id,
            trigger=trigger,
            status="queued",
            current_step_index=0,
            progress_pct=0,
            current_step_id=None,
            current_stage_key="queued",
            current_message=self._run_status_message("queued"),
            context=dict(context or {}),
            last_error=None,
            created_at=now,
            updated_at=now,
        )
        default_failure = get_job_pipeline_default_failure_policy()
        steps: List[JobPipelineStep] = []
        for job_id in workflow_steps:
            cfg = jobs_catalog.get(job_id) if isinstance(jobs_catalog, dict) else None
            if not isinstance(cfg, dict):
                logger.warning("Skipping unknown pipeline job id '%s' for run %s", job_id, run.run_id)
                continue
            runner_ref = str(cfg.get("runner_ref") or "").strip()
            if not runner_ref:
                logger.warning("Skipping job '%s' without runner_ref for run %s", job_id, run.run_id)
                continue
            on_failure = str(cfg.get("on_failure") or default_failure).strip().lower()
            on_failure = on_failure if on_failure in ("continue", "stop") else default_failure
            queued_msg = self._step_status_message(cfg, "queued")
            steps.append(
                JobPipelineStep(
                    run_id=run.run_id,
                    step_index=len(steps),
                    job_id=job_id,
                    runner_ref=runner_ref,
                    on_failure=on_failure,
                    status="queued",
                    progress_pct=0,
                    current_stage_key="queued",
                    current_message=queued_msg,
                    attempt=0,
                    celery_task_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        self._pipeline_repo.create_run(run, steps)
        self._append_event(
            run_id=run.run_id,
            step_index=None,
            event_type="status",
            stage_key="queued",
            message=run.current_message,
            progress_pct=0,
            details={"trigger": trigger, "workflow_id": workflow_id},
        )
        if execute_now:
            return self.execute(run.run_id, is_resume=False)
        return self._snapshot(run)

    def resume(self, run_id: str, *, execute_now: bool = True) -> Dict[str, Any]:
        if execute_now:
            return self.execute(run_id, is_resume=True)
        return self.request_resume(run_id)

    def request_resume(self, run_id: str) -> Dict[str, Any]:
        run = self._pipeline_repo.get_run(run_id)
        if not run:
            raise ValueError(f"Unknown run_id: {run_id}")
        if (run.status or "").lower() != "paused":
            return self._snapshot(run)
        steps = self._pipeline_repo.list_steps(run_id)
        run.status = "queued"
        run.current_stage_key = "resume_requested"
        run.current_message = self._run_status_message("resume_requested")
        run.progress_pct = self._calculate_run_progress(steps, get_job_pipeline_jobs())
        run.updated_at = _utc_now()
        self._pipeline_repo.update_run(run)
        self._append_event(
            run_id=run.run_id,
            step_index=None,
            event_type="status",
            stage_key="resume_requested",
            message=run.current_message,
            progress_pct=run.progress_pct,
            details={"run_id": run.run_id},
        )
        return self._snapshot(run)

    def execute(self, run_id: str, *, is_resume: bool, celery_task_id: Optional[str] = None) -> Dict[str, Any]:
        return self._execute(run_id, is_resume=is_resume, celery_task_id=celery_task_id)

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self._pipeline_repo.get_run(run_id)
        if not run:
            return None
        return self._snapshot(run)

    def get_latest_for_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        run = self._pipeline_repo.get_latest_run_for_bot(bot_id)
        if not run:
            return None
        return self._snapshot(run)

    def _execute(self, run_id: str, *, is_resume: bool, celery_task_id: Optional[str] = None) -> Dict[str, Any]:
        run = self._pipeline_repo.get_run(run_id)
        if not run:
            raise ValueError(f"Unknown run_id: {run_id}")
        steps = self._pipeline_repo.list_steps(run_id)
        jobs_catalog = get_job_pipeline_jobs()
        gates = get_job_pipeline_gates()

        if run.status in ("done", "error") and not is_resume:
            return self._snapshot(run)

        step_index = max(0, int(run.current_step_index or 0))
        if is_resume and (run.status or "").lower() != "paused":
            allowed_resume_from_queued = (
                (run.status or "").lower() == "queued"
                and (run.current_stage_key or "").strip().lower() == "resume_requested"
            )
            if not allowed_resume_from_queued:
                return self._snapshot(run)
        if is_resume and step_index < len(steps):
            paused_step = steps[step_index]
            if paused_step.status == "paused":
                paused_cfg = jobs_catalog.get(paused_step.job_id) if isinstance(jobs_catalog, dict) else {}
                paused_step.status = "done"
                paused_step.progress_pct = 100
                paused_step.current_stage_key = "done"
                paused_step.current_message = self._step_status_message(paused_cfg, "done")
                paused_step.completed_at = _utc_now()
                paused_step.output = {
                    **(paused_step.output if isinstance(paused_step.output, dict) else {}),
                    "resume_event": "manual_resume",
                }
                paused_step.updated_at = paused_step.completed_at
                if celery_task_id:
                    paused_step.celery_task_id = celery_task_id
                self._pipeline_repo.update_step(paused_step)
                self._append_event(
                    run_id=run.run_id,
                    step_index=paused_step.step_index,
                    event_type="status",
                    stage_key="done",
                    message=paused_step.current_message,
                    progress_pct=paused_step.progress_pct,
                    details={"resume_event": "manual_resume", "job_id": paused_step.job_id},
                )
                step_index += 1
                run.current_step_index = step_index
                run.last_error = None
                self._update_run_projection(
                    run=run,
                    steps=steps,
                    jobs_catalog=jobs_catalog,
                    status="running",
                    current_step_id=(steps[step_index].job_id if step_index < len(steps) else None),
                    current_stage_key="running",
                    current_message=self._run_status_message("running"),
                )

        if not steps:
            run.status = "done"
            run.current_step_index = 0
            run.progress_pct = 100
            run.current_step_id = None
            run.current_stage_key = "done"
            run.current_message = self._run_status_message("done")
            run.last_error = None
            run.updated_at = _utc_now()
            self._pipeline_repo.update_run(run)
            self._append_event(
                run_id=run.run_id,
                step_index=None,
                event_type="status",
                stage_key="done",
                message=run.current_message,
                progress_pct=100,
            )
            return self._snapshot(run)

        run.current_step_index = step_index
        run.last_error = None
        self._update_run_projection(
            run=run,
            steps=steps,
            jobs_catalog=jobs_catalog,
            status="running",
            current_step_id=(steps[step_index].job_id if step_index < len(steps) else None),
            current_stage_key="running",
            current_message=self._run_status_message("running"),
        )
        self._append_event(
            run_id=run.run_id,
            step_index=None,
            event_type="status",
            stage_key="running",
            message=run.current_message,
            progress_pct=run.progress_pct,
        )

        for idx in range(step_index, len(steps)):
            step = steps[idx]
            job_cfg = jobs_catalog.get(step.job_id) if isinstance(jobs_catalog, dict) else {}
            if step.status == "done":
                run.current_step_index = idx + 1
                continue

            gate_id, gate_cfg = self._find_pause_gate(step.job_id, gates)
            if gate_cfg and step.status == "queued":
                step.status = "paused"
                step.progress_pct = self._step_running_pct(job_cfg)
                step.current_stage_key = "paused"
                step.current_message = self._step_status_message(job_cfg, "paused")
                step.output = {
                    **(step.output if isinstance(step.output, dict) else {}),
                    "status": "waiting_for_manual_selection",
                    "gate_id": gate_id,
                    "resume_event": str(gate_cfg.get("resume_event") or "manual_resume"),
                }
                step.updated_at = _utc_now()
                if celery_task_id:
                    step.celery_task_id = celery_task_id
                self._pipeline_repo.update_step(step)
                self._append_event(
                    run_id=run.run_id,
                    step_index=step.step_index,
                    event_type="status",
                    stage_key="paused",
                    message=step.current_message,
                    progress_pct=step.progress_pct,
                    details={"gate_id": gate_id, "job_id": step.job_id},
                )

                run.status = "paused"
                run.current_step_index = idx
                run.last_error = None
                self._update_run_projection(
                    run=run,
                    steps=steps,
                    jobs_catalog=jobs_catalog,
                    status="paused",
                    current_step_id=step.job_id,
                    current_stage_key="paused",
                    current_message=step.current_message or self._run_status_message("paused"),
                )
                return self._snapshot(run)

            step.status = "running"
            step.attempt = int(step.attempt or 0) + 1
            step.started_at = step.started_at or _utc_now()
            step.progress_pct = self._step_running_pct(job_cfg)
            step.current_stage_key = "running"
            step.current_message = self._step_status_message(job_cfg, "running")
            step.updated_at = _utc_now()
            if celery_task_id:
                step.celery_task_id = celery_task_id
            self._pipeline_repo.update_step(step)
            self._append_event(
                run_id=run.run_id,
                step_index=step.step_index,
                event_type="progress",
                stage_key="running",
                message=step.current_message,
                progress_pct=step.progress_pct,
                details={"attempt": step.attempt, "job_id": step.job_id},
            )

            run.current_step_index = idx
            self._update_run_projection(
                run=run,
                steps=steps,
                jobs_catalog=jobs_catalog,
                status="running",
                current_step_id=step.job_id,
                current_stage_key="running",
                current_message=step.current_message,
            )

            runner_context = {
                **(run.context if isinstance(run.context, dict) else {}),
                "run_id": run.run_id,
                "bot_id": run.bot_id,
                "org_id": run.org_id,
                "workflow_id": run.workflow_id,
                "trigger": run.trigger,
                "step_index": step.step_index,
                "job_id": step.job_id,
                "job_config": dict(job_cfg) if isinstance(job_cfg, dict) else {},
                "celery_task_id": celery_task_id,
            }
            result = self._run_step(step, runner_context)
            if result.status == "done" and self._completion_enabled(job_cfg, result):
                result = self._await_linked_job_completion(
                    run=run,
                    steps=steps,
                    step=step,
                    job_cfg=job_cfg,
                    result=result,
                    celery_task_id=celery_task_id,
                    jobs_catalog=jobs_catalog,
                )

            if result.status == "paused":
                step.status = "paused"
                step.progress_pct = self._step_running_pct(job_cfg)
                step.current_stage_key = "paused"
                step.current_message = self._step_status_message(job_cfg, "paused")
                step.output = dict(result.output or {})
                step.linked_job_type = result.linked_job_type
                step.linked_job_id = result.linked_job_id
                step.last_error = result.error
                step.updated_at = _utc_now()
                self._pipeline_repo.update_step(step)
                self._append_event(
                    run_id=run.run_id,
                    step_index=step.step_index,
                    event_type="status",
                    stage_key="paused",
                    message=step.current_message,
                    progress_pct=step.progress_pct,
                    details={"job_id": step.job_id, "linked_job_id": step.linked_job_id},
                )

                run.status = "paused"
                run.current_step_index = idx
                run.last_error = result.error
                self._update_run_projection(
                    run=run,
                    steps=steps,
                    jobs_catalog=jobs_catalog,
                    status="paused",
                    current_step_id=step.job_id,
                    current_stage_key="paused",
                    current_message=step.current_message,
                    last_error=run.last_error,
                )
                return self._snapshot(run)

            if result.status == "error":
                step.status = "error"
                step.progress_pct = 100 if step.on_failure == "continue" else self._step_running_pct(job_cfg)
                step.current_stage_key = "error"
                error_message = result.error or self._step_status_message(job_cfg, "error") or "step failed"
                step.current_message = error_message
                step.last_error = error_message
                step.output = dict(result.output or {})
                step.linked_job_type = result.linked_job_type
                step.linked_job_id = result.linked_job_id
                step.completed_at = _utc_now()
                step.updated_at = step.completed_at
                self._pipeline_repo.update_step(step)
                self._append_event(
                    run_id=run.run_id,
                    step_index=step.step_index,
                    event_type="error",
                    stage_key="error",
                    message=step.current_message,
                    progress_pct=step.progress_pct,
                    details={"job_id": step.job_id, "on_failure": step.on_failure},
                )

                run.last_error = step.last_error
                if step.on_failure == "continue":
                    run.current_step_index = idx + 1
                    self._update_run_projection(
                        run=run,
                        steps=steps,
                        jobs_catalog=jobs_catalog,
                        status="running",
                        current_step_id=(steps[idx + 1].job_id if idx + 1 < len(steps) else None),
                        current_stage_key="running",
                        current_message=self._run_status_message("running"),
                        last_error=run.last_error,
                    )
                    continue

                run.status = "error"
                run.current_step_index = idx
                self._update_run_projection(
                    run=run,
                    steps=steps,
                    jobs_catalog=jobs_catalog,
                    status="error",
                    current_step_id=step.job_id,
                    current_stage_key="error",
                    current_message=step.current_message or self._run_status_message("error"),
                    last_error=run.last_error,
                )
                self._append_event(
                    run_id=run.run_id,
                    step_index=None,
                    event_type="status",
                    stage_key="error",
                    message=run.current_message,
                    progress_pct=run.progress_pct,
                )
                return self._snapshot(run)

            step.status = "done"
            step.progress_pct = 100
            step.current_stage_key = "done"
            step.current_message = self._step_status_message(job_cfg, "done")
            step.output = dict(result.output or {})
            step.linked_job_type = result.linked_job_type
            step.linked_job_id = result.linked_job_id
            step.last_error = None
            step.completed_at = _utc_now()
            step.updated_at = step.completed_at
            self._pipeline_repo.update_step(step)
            self._append_event(
                run_id=run.run_id,
                step_index=step.step_index,
                event_type="status",
                stage_key="done",
                message=step.current_message,
                progress_pct=step.progress_pct,
                details={"job_id": step.job_id, "linked_job_id": step.linked_job_id},
            )

            run.current_step_index = idx + 1
            self._update_run_projection(
                run=run,
                steps=steps,
                jobs_catalog=jobs_catalog,
                status="running",
                current_step_id=(steps[idx + 1].job_id if idx + 1 < len(steps) else None),
                current_stage_key="running",
                current_message=self._run_status_message("running"),
            )

        run.status = "done"
        run.last_error = None
        run.current_step_index = len(steps)
        run.current_step_id = None
        run.current_stage_key = "done"
        run.current_message = self._run_status_message("done")
        run.progress_pct = 100
        run.updated_at = _utc_now()
        self._pipeline_repo.update_run(run)
        self._append_event(
            run_id=run.run_id,
            step_index=None,
            event_type="status",
            stage_key="done",
            message=run.current_message,
            progress_pct=100,
        )
        return self._snapshot(run)

    def _run_step(self, step: JobPipelineStep, context: Dict[str, Any]) -> JobResult:
        try:
            runner = self._runner_registry.resolve(step.runner_ref)
            return runner.run(context)
        except Exception as exc:
            logger.exception("Pipeline step failed (run=%s, step=%s): %s", step.run_id, step.job_id, exc)
            return JobResult(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")

    def _completion_cfg(self, job_cfg: Any) -> Dict[str, Any]:
        if not isinstance(job_cfg, dict):
            return {}
        cfg = job_cfg.get("completion")
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _completion_enabled(self, job_cfg: Any, result: JobResult) -> bool:
        if not (result.linked_job_id and result.linked_job_type):
            return False
        cfg = self._completion_cfg(job_cfg)
        return bool(cfg.get("enabled"))

    def _completion_checker_ref(self, job_cfg: Any) -> str:
        cfg = self._completion_cfg(job_cfg)
        return str(cfg.get("checker_ref") or "").strip()

    def _completion_poll_interval_sec(self, job_cfg: Any) -> int:
        cfg = self._completion_cfg(job_cfg)
        try:
            value = int(cfg.get("poll_interval_sec"))
        except (TypeError, ValueError):
            value = 5
        return value if value > 0 else 5

    def _completion_max_wait_sec(self, job_cfg: Any) -> int:
        cfg = self._completion_cfg(job_cfg)
        try:
            value = int(cfg.get("max_wait_sec"))
        except (TypeError, ValueError):
            value = 1200
        return value if value > 0 else 1200

    def _await_linked_job_completion(
        self,
        *,
        run: JobPipelineRun,
        steps: List[JobPipelineStep],
        step: JobPipelineStep,
        job_cfg: Dict[str, Any],
        result: JobResult,
        celery_task_id: Optional[str],
        jobs_catalog: Dict[str, Dict[str, Any]],
    ) -> JobResult:
        checker_ref = self._completion_checker_ref(job_cfg)
        if not checker_ref:
            return JobResult(
                status="error",
                error=f"Missing completion.checker_ref for step '{step.job_id}'",
                output=dict(result.output or {}),
                linked_job_type=result.linked_job_type,
                linked_job_id=result.linked_job_id,
            )

        try:
            checker = self._completion_registry.resolve(checker_ref)
        except Exception as exc:
            return JobResult(
                status="error",
                error=f"Completion checker resolution failed: {type(exc).__name__}: {str(exc)[:200]}",
                output=dict(result.output or {}),
                linked_job_type=result.linked_job_type,
                linked_job_id=result.linked_job_id,
            )

        poll_interval = self._completion_poll_interval_sec(job_cfg)
        max_wait = self._completion_max_wait_sec(job_cfg)
        deadline = time.monotonic() + max_wait
        running_msg = self._step_status_message(job_cfg, "running")
        last_signature: Optional[Tuple[str, Optional[str], int]] = None

        while True:
            check_context = {
                **(run.context if isinstance(run.context, dict) else {}),
                "run_id": run.run_id,
                "bot_id": run.bot_id,
                "org_id": run.org_id,
                "workflow_id": run.workflow_id,
                "trigger": run.trigger,
                "step_index": step.step_index,
                "job_id": step.job_id,
                "job_config": dict(job_cfg),
                "linked_job_type": result.linked_job_type,
                "linked_job_id": result.linked_job_id,
                "celery_task_id": celery_task_id,
            }
            try:
                completion = checker.check(check_context)
            except Exception as exc:
                return JobResult(
                    status="error",
                    error=f"Completion check failed: {type(exc).__name__}: {str(exc)[:200]}",
                    output=dict(result.output or {}),
                    linked_job_type=result.linked_job_type,
                    linked_job_id=result.linked_job_id,
                )

            child_status = str(getattr(completion, "status", "") or "").strip().lower()
            child_details = dict(getattr(completion, "details", {}) or {})
            child_message = str(getattr(completion, "message", "") or "").strip() or running_msg
            child_progress_raw = getattr(completion, "progress_pct", None)
            child_progress = _clamp_pct(child_progress_raw) if child_progress_raw is not None else self._step_running_pct(job_cfg)
            child_error = str(getattr(completion, "error", "") or "").strip()
            if child_status == "done":
                # Avoid confusing "child_done at running %" in dashboard activity.
                child_progress = 100
                done_msg = self._step_status_message(job_cfg, "done")
                if done_msg:
                    child_message = done_msg

            step.status = "running"
            step.progress_pct = child_progress
            step.current_stage_key = f"child_{child_status or 'running'}"
            step.current_message = child_message
            step.linked_job_type = result.linked_job_type
            step.linked_job_id = result.linked_job_id
            step.output = {
                **(dict(result.output or {})),
                "completion": {
                    "status": child_status,
                    "details": child_details,
                },
            }
            step.updated_at = _utc_now()
            if celery_task_id:
                step.celery_task_id = celery_task_id
            self._pipeline_repo.update_step(step)

            run.current_step_index = step.step_index
            self._update_run_projection(
                run=run,
                steps=steps,
                jobs_catalog=jobs_catalog,
                status="running",
                current_step_id=step.job_id,
                current_stage_key=step.current_stage_key,
                current_message=step.current_message,
            )

            signature = (child_status, child_message, child_progress)
            if signature != last_signature:
                self._append_event(
                    run_id=run.run_id,
                    step_index=step.step_index,
                    event_type="progress",
                    stage_key=step.current_stage_key,
                    message=step.current_message,
                    progress_pct=child_progress,
                    details={"job_id": step.job_id, "linked_job_id": step.linked_job_id, **child_details},
                )
                last_signature = signature

            if child_status == "done":
                return JobResult(
                    status="done",
                    output=dict(step.output or {}),
                    linked_job_type=result.linked_job_type,
                    linked_job_id=result.linked_job_id,
                )
            if child_status == "error":
                return JobResult(
                    status="error",
                    error=child_error or f"Linked job failed for step '{step.job_id}'",
                    output=dict(step.output or {}),
                    linked_job_type=result.linked_job_type,
                    linked_job_id=result.linked_job_id,
                )
            if time.monotonic() >= deadline:
                return JobResult(
                    status="error",
                    error=f"Linked job timeout after {max_wait}s for step '{step.job_id}'",
                    output=dict(step.output or {}),
                    linked_job_type=result.linked_job_type,
                    linked_job_id=result.linked_job_id,
                )
            time.sleep(poll_interval)

    def _find_pause_gate(
        self,
        job_id: str,
        gates: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not isinstance(gates, dict):
            return None, None
        target = str(job_id or "").strip()
        if not target:
            return None, None
        for gate_id, gate in gates.items():
            if not isinstance(gate, dict):
                continue
            step_id = str(gate.get("step_id") or "").strip()
            pause_on_step = bool(gate.get("pause_on_step"))
            if pause_on_step and step_id == target:
                return str(gate_id or "").strip() or None, gate
        return None, None

    def _step_running_pct(self, job_cfg: Any) -> int:
        if isinstance(job_cfg, dict):
            return _clamp_pct(job_cfg.get("running_progress_pct"))
        return 0

    def _step_status_message(self, job_cfg: Any, status: str) -> Optional[str]:
        if not isinstance(job_cfg, dict):
            return None
        raw_messages = job_cfg.get("progress_messages")
        if not isinstance(raw_messages, dict):
            return None
        text = _localized_text(raw_messages.get(status))
        return text if text else None

    def _run_status_message(self, status: str) -> Optional[str]:
        pipeline_cfg = get_job_pipeline_config()
        defaults = pipeline_cfg.get("defaults") if isinstance(pipeline_cfg.get("defaults"), dict) else {}
        run_messages = defaults.get("run_messages") if isinstance(defaults.get("run_messages"), dict) else {}
        text = _localized_text(run_messages.get(status))
        return text if text else None

    def _step_weight(self, job_cfg: Any) -> int:
        if not isinstance(job_cfg, dict):
            return 1
        try:
            weight = int(job_cfg.get("progress_weight"))
        except (TypeError, ValueError):
            weight = 1
        return 1 if weight <= 0 else weight

    def _calculate_run_progress(self, steps: List[JobPipelineStep], jobs_catalog: Dict[str, Dict[str, Any]]) -> int:
        if not steps:
            return 0
        total_weight = 0
        weighted_progress = 0.0
        for step in steps:
            job_cfg = jobs_catalog.get(step.job_id) if isinstance(jobs_catalog, dict) else {}
            weight = self._step_weight(job_cfg)
            total_weight += weight
            weighted_progress += float(_clamp_pct(step.progress_pct)) * float(weight)
        if total_weight <= 0:
            return 0
        return _clamp_pct(round(weighted_progress / float(total_weight)))

    def _append_event(
        self,
        *,
        run_id: str,
        step_index: Optional[int],
        event_type: str,
        stage_key: Optional[str],
        message: Optional[str],
        progress_pct: Optional[int],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._pipeline_repo.append_event(
            JobPipelineStepEvent(
                event_id=_new_event_id(),
                run_id=run_id,
                step_index=step_index,
                event_type=str(event_type or "info").strip() or "info",
                stage_key=str(stage_key or "").strip() or None,
                message=str(message or "").strip() or None,
                progress_pct=_clamp_pct(progress_pct) if progress_pct is not None else None,
                details=dict(details or {}),
                created_at=_utc_now(),
            )
        )

    def _update_run_projection(
        self,
        *,
        run: JobPipelineRun,
        steps: List[JobPipelineStep],
        jobs_catalog: Dict[str, Dict[str, Any]],
        status: str,
        current_step_id: Optional[str],
        current_stage_key: Optional[str],
        current_message: Optional[str],
        last_error: Optional[str] = None,
    ) -> None:
        run.status = status
        run.current_step_id = current_step_id
        run.current_stage_key = current_stage_key
        run.current_message = current_message
        run.progress_pct = self._calculate_run_progress(steps, jobs_catalog)
        run.last_error = last_error
        run.updated_at = _utc_now()
        self._pipeline_repo.update_run(run)

    def _snapshot(self, run: JobPipelineRun) -> Dict[str, Any]:
        steps = self._pipeline_repo.list_steps(run.run_id)
        events = self._pipeline_repo.list_events(run.run_id)
        return {
            "run": {
                "run_id": run.run_id,
                "org_id": run.org_id,
                "bot_id": run.bot_id,
                "workflow_id": run.workflow_id,
                "trigger": run.trigger,
                "status": run.status,
                "current_step_index": run.current_step_index,
                "progress_pct": run.progress_pct,
                "current_step_id": run.current_step_id,
                "current_stage_key": run.current_stage_key,
                "current_message": run.current_message,
                "context": dict(run.context or {}),
                "last_error": run.last_error,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            },
            "steps": [
                {
                    "run_id": step.run_id,
                    "step_index": step.step_index,
                    "job_id": step.job_id,
                    "runner_ref": step.runner_ref,
                    "on_failure": step.on_failure,
                    "status": step.status,
                    "progress_pct": step.progress_pct,
                    "current_stage_key": step.current_stage_key,
                    "current_message": step.current_message,
                    "attempt": step.attempt,
                    "celery_task_id": step.celery_task_id,
                    "linked_job_type": step.linked_job_type,
                    "linked_job_id": step.linked_job_id,
                    "output": dict(step.output or {}),
                    "last_error": step.last_error,
                    "started_at": step.started_at,
                    "completed_at": step.completed_at,
                    "created_at": step.created_at,
                    "updated_at": step.updated_at,
                }
                for step in steps
            ],
            "events": [
                {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "step_index": event.step_index,
                    "event_type": event.event_type,
                    "stage_key": event.stage_key,
                    "message": event.message,
                    "progress_pct": event.progress_pct,
                    "details": dict(event.details or {}),
                    "created_at": event.created_at,
                }
                for event in events
            ],
        }
