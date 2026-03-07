import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from application.services.job_pipeline.engine import JobPipelineEngine
from domain.entities import Bot, JobPipelineRun, JobPipelineStep
from domain.interfaces import JobResult


class _InMemoryPipelineRepo:
    def __init__(self) -> None:
        self.runs: Dict[str, JobPipelineRun] = {}
        self.steps: Dict[str, List[JobPipelineStep]] = {}

    def create_run(self, run: JobPipelineRun, steps: List[JobPipelineStep]) -> None:
        self.runs[run.run_id] = run
        self.steps[run.run_id] = list(steps)

    def get_run(self, run_id: str) -> Optional[JobPipelineRun]:
        return self.runs.get(run_id)

    def get_latest_run_for_bot(self, bot_id: str) -> Optional[JobPipelineRun]:
        runs = [r for r in self.runs.values() if r.bot_id == bot_id]
        if not runs:
            return None
        return sorted(runs, key=lambda r: r.updated_at or "", reverse=True)[0]

    def list_steps(self, run_id: str) -> List[JobPipelineStep]:
        return list(self.steps.get(run_id, []))

    def update_run(self, run: JobPipelineRun) -> None:
        self.runs[run.run_id] = run

    def update_step(self, step: JobPipelineStep) -> None:
        items = self.steps.get(step.run_id, [])
        for idx, existing in enumerate(items):
            if existing.step_index == step.step_index:
                items[idx] = step
                break


class _BotRepo:
    def __init__(self) -> None:
        self._bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config="{}",
        )

    def get_bot(self, bot_id: str) -> Optional[Bot]:
        return self._bot if bot_id == self._bot.bot_id else None


class _Runner:
    def __init__(self, result: JobResult, trace: List[str]) -> None:
        self._result = result
        self._trace = trace

    def run(self, context: Dict[str, Any]) -> JobResult:
        self._trace.append(str(context.get("job_id") or ""))
        return self._result


class _Registry:
    def __init__(self, runners: Dict[str, _Runner]) -> None:
        self._runners = runners

    def resolve(self, runner_ref: str) -> _Runner:
        return self._runners[runner_ref]


class JobPipelineEngineTests(unittest.TestCase):
    def test_sequential_order_and_continue_on_failure(self) -> None:
        trace: List[str] = []
        engine = JobPipelineEngine(
            pipeline_repo=_InMemoryPipelineRepo(),
            bot_repo=_BotRepo(),
            runner_registry=_Registry(
                {
                    "r.a": _Runner(JobResult(status="done"), trace),
                    "r.b": _Runner(JobResult(status="error", error="b failed"), trace),
                    "r.c": _Runner(JobResult(status="done"), trace),
                }
            ),
        )

        with (
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_jobs",
                return_value={
                    "a": {"runner_ref": "r.a"},
                    "b": {"runner_ref": "r.b"},
                    "c": {"runner_ref": "r.c"},
                },
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_workflow",
                return_value=["a", "b", "c"],
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_default_failure_policy",
                return_value="continue",
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_gates",
                return_value={},
            ),
        ):
            snapshot = engine.start(bot_id="bot_1")

        self.assertEqual(["a", "b", "c"], trace)
        self.assertEqual("done", snapshot["run"]["status"])
        self.assertEqual([0, 1, 2], [s["step_index"] for s in snapshot["steps"]])
        self.assertEqual(["done", "error", "done"], [s["status"] for s in snapshot["steps"]])

    def test_stop_on_blocking_failure(self) -> None:
        trace: List[str] = []
        engine = JobPipelineEngine(
            pipeline_repo=_InMemoryPipelineRepo(),
            bot_repo=_BotRepo(),
            runner_registry=_Registry(
                {
                    "r.a": _Runner(JobResult(status="done"), trace),
                    "r.b": _Runner(JobResult(status="error", error="b failed"), trace),
                    "r.c": _Runner(JobResult(status="done"), trace),
                }
            ),
        )

        with (
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_jobs",
                return_value={
                    "a": {"runner_ref": "r.a"},
                    "b": {"runner_ref": "r.b", "on_failure": "stop"},
                    "c": {"runner_ref": "r.c"},
                },
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_workflow",
                return_value=["a", "b", "c"],
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_default_failure_policy",
                return_value="continue",
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_gates",
                return_value={},
            ),
        ):
            snapshot = engine.start(bot_id="bot_1")

        self.assertEqual(["a", "b"], trace)
        self.assertEqual("error", snapshot["run"]["status"])
        self.assertEqual(["done", "error", "queued"], [s["status"] for s in snapshot["steps"]])

    def test_pause_and_resume_with_gate(self) -> None:
        trace: List[str] = []
        pipeline_repo = _InMemoryPipelineRepo()
        engine = JobPipelineEngine(
            pipeline_repo=pipeline_repo,
            bot_repo=_BotRepo(),
            runner_registry=_Registry(
                {
                    "r.discovery": _Runner(JobResult(status="done"), trace),
                    "r.booking": _Runner(JobResult(status="done"), trace),
                }
            ),
        )

        with (
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_jobs",
                return_value={
                    "discovery": {"runner_ref": "r.discovery"},
                    "booking_link": {"runner_ref": "r.booking"},
                },
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_workflow",
                return_value=["discovery", "booking_link"],
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_default_failure_policy",
                return_value="continue",
            ),
            patch(
                "application.services.job_pipeline.engine.get_job_pipeline_gates",
                return_value={
                    "discovery_manual_selection": {
                        "step_id": "discovery",
                        "pause_on_step": True,
                        "resume_event": "urls_selected",
                    }
                },
            ),
        ):
            started = engine.start(bot_id="bot_1")
            self.assertEqual("paused", started["run"]["status"])
            self.assertEqual([], trace)
            run_id = started["run"]["run_id"]
            resumed = engine.resume(run_id)

        self.assertEqual("done", resumed["run"]["status"])
        self.assertEqual(["booking_link"], trace)
        self.assertEqual(["done", "done"], [s["status"] for s in resumed["steps"]])


if __name__ == "__main__":
    unittest.main()
