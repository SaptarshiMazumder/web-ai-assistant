from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from domain.entities import JobPipelineRun, JobPipelineStep
from domain.interfaces import JobResult
from domain.repositories import BotRepository, JobPipelineRepository
from domain.platform_profiles import (
    get_job_pipeline_default_failure_policy,
    get_job_pipeline_gates,
    get_job_pipeline_jobs,
    get_job_pipeline_workflow,
)

from .registry import JobRunnerRegistry

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return "jpr_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")


class JobPipelineEngine:
    """Sequential pipeline engine: queued -> running -> paused -> running -> done|error."""

    def __init__(
        self,
        *,
        pipeline_repo: JobPipelineRepository,
        bot_repo: BotRepository,
        runner_registry: JobRunnerRegistry,
    ) -> None:
        self._pipeline_repo = pipeline_repo
        self._bot_repo = bot_repo
        self._runner_registry = runner_registry

    def start(
        self,
        *,
        bot_id: str,
        workflow_id: str = "default",
        trigger: str = "post_crawl",
        context: Optional[Dict[str, Any]] = None,
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
            steps.append(
                JobPipelineStep(
                    run_id=run.run_id,
                    step_index=len(steps),
                    job_id=job_id,
                    runner_ref=runner_ref,
                    on_failure=on_failure,
                    status="queued",
                    created_at=now,
                    updated_at=now,
                )
            )
        self._pipeline_repo.create_run(run, steps)
        return self._execute(run.run_id, is_resume=False)

    def resume(self, run_id: str) -> Dict[str, Any]:
        return self._execute(run_id, is_resume=True)

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

    def _execute(self, run_id: str, *, is_resume: bool) -> Dict[str, Any]:
        run = self._pipeline_repo.get_run(run_id)
        if not run:
            raise ValueError(f"Unknown run_id: {run_id}")
        steps = self._pipeline_repo.list_steps(run_id)
        if run.status in ("done", "error") and not is_resume:
            return self._snapshot(run)

        step_index = max(0, int(run.current_step_index or 0))
        if is_resume and run.status != "paused":
            return self._snapshot(run)
        if is_resume and step_index < len(steps):
            paused_step = steps[step_index]
            if paused_step.status == "paused":
                paused_step.status = "done"
                paused_step.completed_at = _utc_now()
                paused_step.output = {
                    **(paused_step.output if isinstance(paused_step.output, dict) else {}),
                    "resume_event": "manual_resume",
                }
                paused_step.updated_at = paused_step.completed_at
                self._pipeline_repo.update_step(paused_step)
                step_index += 1
                run.current_step_index = step_index
                run.updated_at = _utc_now()
                self._pipeline_repo.update_run(run)

        run.status = "running"
        run.updated_at = _utc_now()
        self._pipeline_repo.update_run(run)
        jobs_catalog = get_job_pipeline_jobs()
        gates = get_job_pipeline_gates()

        for idx in range(step_index, len(steps)):
            step = steps[idx]
            if step.status == "done":
                run.current_step_index = idx + 1
                run.updated_at = _utc_now()
                self._pipeline_repo.update_run(run)
                continue

            gate_id, gate_cfg = self._find_pause_gate(step.job_id, gates)
            if gate_cfg and step.status == "queued":
                step.status = "paused"
                step.output = {
                    **(step.output if isinstance(step.output, dict) else {}),
                    "status": "waiting_for_manual_selection",
                    "gate_id": gate_id,
                    "resume_event": str(gate_cfg.get("resume_event") or "manual_resume"),
                }
                step.updated_at = _utc_now()
                self._pipeline_repo.update_step(step)

                run.status = "paused"
                run.current_step_index = idx
                run.last_error = None
                run.updated_at = _utc_now()
                self._pipeline_repo.update_run(run)
                return self._snapshot(run)

            step.status = "running"
            step.started_at = step.started_at or _utc_now()
            step.updated_at = _utc_now()
            self._pipeline_repo.update_step(step)

            job_cfg = jobs_catalog.get(step.job_id) if isinstance(jobs_catalog, dict) else {}
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
            }
            result = self._run_step(step, runner_context)

            if result.status == "paused":
                step.status = "paused"
                step.output = dict(result.output or {})
                step.linked_job_type = result.linked_job_type
                step.linked_job_id = result.linked_job_id
                step.last_error = result.error
                step.updated_at = _utc_now()
                self._pipeline_repo.update_step(step)

                run.status = "paused"
                run.current_step_index = idx
                run.last_error = result.error
                run.updated_at = _utc_now()
                self._pipeline_repo.update_run(run)
                return self._snapshot(run)

            if result.status == "error":
                step.status = "error"
                step.last_error = result.error or "step failed"
                step.output = dict(result.output or {})
                step.linked_job_type = result.linked_job_type
                step.linked_job_id = result.linked_job_id
                step.completed_at = _utc_now()
                step.updated_at = step.completed_at
                self._pipeline_repo.update_step(step)

                run.last_error = step.last_error
                if step.on_failure == "continue":
                    run.current_step_index = idx + 1
                    run.updated_at = _utc_now()
                    self._pipeline_repo.update_run(run)
                    continue

                run.status = "error"
                run.current_step_index = idx
                run.updated_at = _utc_now()
                self._pipeline_repo.update_run(run)
                return self._snapshot(run)

            step.status = "done"
            step.output = dict(result.output or {})
            step.linked_job_type = result.linked_job_type
            step.linked_job_id = result.linked_job_id
            step.last_error = None
            step.completed_at = _utc_now()
            step.updated_at = step.completed_at
            self._pipeline_repo.update_step(step)

            run.current_step_index = idx + 1
            run.updated_at = _utc_now()
            self._pipeline_repo.update_run(run)

        run.status = "done"
        run.last_error = None
        run.updated_at = _utc_now()
        self._pipeline_repo.update_run(run)
        return self._snapshot(run)

    def _run_step(self, step: JobPipelineStep, context: Dict[str, Any]) -> JobResult:
        try:
            runner = self._runner_registry.resolve(step.runner_ref)
            return runner.run(context)
        except Exception as exc:
            logger.exception("Pipeline step failed (run=%s, step=%s): %s", step.run_id, step.job_id, exc)
            return JobResult(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")

    def _find_pause_gate(
        self,
        job_id: str,
        gates: Dict[str, Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
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

    def _snapshot(self, run: JobPipelineRun) -> Dict[str, Any]:
        steps = self._pipeline_repo.list_steps(run.run_id)
        return {
            "run": {
                "run_id": run.run_id,
                "org_id": run.org_id,
                "bot_id": run.bot_id,
                "workflow_id": run.workflow_id,
                "trigger": run.trigger,
                "status": run.status,
                "current_step_index": run.current_step_index,
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
        }
