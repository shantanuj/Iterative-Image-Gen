from __future__ import annotations

import copy
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .config import load_questions_from_path, normalize_questions
from .io_utils import (
    parse_action_prompt,
    score_from_scores,
    strip_json_fence,
    write_json,
)
from .prompts import (
    QUESTION_MAPPING_SYSTEM_PROMPT,
    STEPBYSTEP_CRITIC_CORE_SYSTEM_PROMPT,
    STEPBYSTEP_CRITIC_EDIT_SYSTEM_PROMPT,
    VIDEO_EDIT_ACTION_SYSTEM_PROMPT,
    get_plan_system_prompt,
)
from .schemas import Candidate, RunResult, StepRecord, TrajectoryResult
from .video_providers import VideoEditor, VideoGenerator, build_video_editor, build_video_generator
from .vlm_providers import VLMClient, build_vlm_client


def _sanitize_config(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if "api_key" in key.lower() and key.lower() != "api_key_env":
                out[key] = "<redacted>"
            else:
                out[key] = _sanitize_config(nested)
        return out
    if isinstance(value, list):
        return [_sanitize_config(item) for item in value]
    return value


def _format_scores(scores: list[tuple[str, float]]) -> str:
    rows = []
    for question, score in scores:
        if question == "Cumulative mean binary score":
            rows.append(f"  Overall score: {float(score):.2f}")
        else:
            rows.append(f"  {question}: {'yes' if score == 1 else 'no'}")
    return "\n".join(rows)


def _retry(label: str, fn: Callable[[], Any], attempts: int = 2, sleep_seconds: int = 15) -> Any:
    last_error = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if idx + 1 < attempts:
                print(f"  [{label}] failed; retrying in {sleep_seconds}s:\n{traceback.format_exc()}")
                time.sleep(sleep_seconds)
    raise last_error  # type: ignore[misc]


class IterativeVideoGenRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.run_cfg = self.config["run"]
        self.method_cfg = self.config["method"]
        self.generation_cfg = self.config["generation"]
        self.editing_cfg = self.config["editing"]
        self.iterations = int(self.run_cfg["iterations"])
        self.parallel = int(self.run_cfg["parallel"])
        self.seed = self.run_cfg.get("seed")
        configured_max_workers = self.run_cfg.get("max_workers")
        self.max_workers = int(configured_max_workers) if configured_max_workers else None

        self.generator: VideoGenerator = build_video_generator(
            self.config["models"]["generator"], self.generation_cfg
        )
        self.editor: VideoEditor = build_video_editor(
            self.config["models"]["editor"], self.editing_cfg
        )
        self.critic: VLMClient = build_vlm_client(self.config["models"]["critic"])
        self.verifier: VLMClient = build_vlm_client(self.config["models"]["verifier"])
        eval_verifier_cfg = self.config["models"].get("eval_verifier")
        self.eval_verifier = build_vlm_client(eval_verifier_cfg) if eval_verifier_cfg else None

    def _seed_for(self, offset: int) -> int | None:
        if self.seed is None:
            return None
        return int(self.seed) + offset

    def _worker_count(self, task_count: int) -> int:
        if task_count <= 1:
            return 1
        configured = self.max_workers if self.max_workers is not None else self.parallel
        return max(1, min(int(configured), task_count))

    def _verify(
        self,
        verifier: VLMClient,
        video_path: str,
        questions: list[str],
    ) -> tuple[str, list[tuple[str, float]]]:
        return verifier.verify(video_path, questions)

    def _candidate(
        self,
        video_path: str,
        scores: list[tuple[str, float]],
        trajectory_id: str,
        step_index: int,
        prompt: str,
        action: str,
    ) -> Candidate:
        return Candidate(
            video_path=video_path,
            score=score_from_scores(scores),
            verifier_scores=scores,
            trajectory_id=trajectory_id,
            step_index=step_index,
            prompt=prompt,
            action=action,
        )

    def _iterative_critic(
        self,
        video_path: str,
        prompt: str,
        scores: list[tuple[str, float]],
        previous_edit_prompts: list[str],
    ) -> tuple[str, str, str, str]:
        edit_history = ""
        if previous_edit_prompts:
            edit_history = "\nPrevious editing steps:\n"
            for idx, edit_prompt in enumerate(previous_edit_prompts):
                edit_history += f"  Step {idx + 1}: {edit_prompt}\n"
        user_prompt = (
            f"Target video prompt: {prompt}\n\n"
            f"Current verifier scores:\n{_format_scores(scores)}\n"
            f"{edit_history}\n"
            "Based on the attached video and the scores above, decide the next action "
            "and provide a brief, simple editing prompt."
        )
        raw = self.critic.complete(VIDEO_EDIT_ACTION_SYSTEM_PROMPT, user_prompt, video_path=video_path)
        action, edit_prompt = parse_action_prompt(raw)
        return raw, action, edit_prompt, user_prompt

    def _step_critic(
        self,
        video_path: str,
        full_prompt: str,
        core_prompt: str,
        completed_steps: list[str],
        current_step_desc: str,
        scores: list[tuple[str, float]],
        previous_edit_prompts: list[str],
        attempt_number: int,
        max_attempts: int,
        is_core_step: bool,
    ) -> tuple[str, str, str, str, str]:
        expected_so_far = f"Core generation: {core_prompt}"
        for idx, step in enumerate(completed_steps):
            expected_so_far += f"\n  Completed add step {idx + 1}: {step}"
        expected_so_far += f"\n  Current step just attempted: {current_step_desc}"

        remaining = max_attempts - attempt_number
        attempt_info = (
            f"This is refinement attempt {attempt_number + 1} of {max_attempts}. "
            f"You have {remaining} attempt(s) remaining (including this one)."
        )
        if remaining <= 1:
            attempt_info += " This is your LAST attempt — make it count."

        edit_history = ""
        if previous_edit_prompts:
            edit_history = "\nEdit/refinement history so far:\n"
            for idx, edit_prompt in enumerate(previous_edit_prompts):
                edit_history += f"  {idx + 1}. {edit_prompt}\n"

        user_prompt = (
            f"Full target prompt: {full_prompt}\n\n"
            f"What SHOULD be present in the video by now:\n{expected_so_far}\n\n"
            f"Current verifier scores:\n{_format_scores(scores)}\n"
            f"\n{attempt_info}\n"
            f"{edit_history}\n"
            "Evaluate: are the elements that should be present by now actually in the video? "
            "What should we do next?"
        )
        system_prompt = (
            STEPBYSTEP_CRITIC_CORE_SYSTEM_PROMPT
            if is_core_step
            else STEPBYSTEP_CRITIC_EDIT_SYSTEM_PROMPT
        )
        raw = self.critic.complete(system_prompt, user_prompt, video_path=video_path)
        action, edit_prompt = parse_action_prompt(raw)
        return raw, action, edit_prompt, system_prompt, user_prompt

    def run_single_iterative_trajectory(
        self,
        prompt: str,
        questions: list[str],
        output_dir: str | Path,
        trajectory_id: str,
        seed_offset: int,
    ) -> TrajectoryResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = TrajectoryResult(trajectory_id=trajectory_id, output_dir=str(output_dir))
        previous_edit_prompts: list[str] = []
        previous_videos: list[str] = []
        best_candidate: Candidate | None = None

        def log(message: str) -> None:
            print(message)
            result.text_log += message + "\n"

        step0_path = output_dir / "step_0.mp4"
        log(f"[{trajectory_id}] Step 0: generating initial video")
        current_video = _retry(
            "video_gen",
            lambda: self.generator.generate(
                prompt,
                step0_path,
                seed=self._seed_for(seed_offset + 1000),
            ),
        )
        previous_videos.append(current_video)

        step_index = 0
        while step_index < self.iterations:
            log(f"[{trajectory_id}] Step {step_index}: verify + critic")
            raw_verifier, scores = _retry(
                "verifier", lambda: self._verify(self.verifier, current_video, questions)
            )
            candidate = self._candidate(
                current_video, scores, trajectory_id, step_index, prompt, "VERIFY"
            )
            result.candidates.append(candidate)
            if best_candidate is None or candidate.score >= best_candidate.score:
                best_candidate = candidate

            raw_critic, action, edit_prompt, critic_user_prompt = _retry(
                "critic",
                lambda: self._iterative_critic(
                    current_video, prompt, scores, previous_edit_prompts
                ),
            )
            result.steps.append(
                StepRecord(
                    step_index=step_index,
                    action=action,
                    prompt=edit_prompt,
                    video_path=current_video,
                    source_video_path=None,
                    verifier_scores=scores,
                    raw_verifier_response=raw_verifier,
                    raw_critic_response=raw_critic,
                    critic_system_prompt=VIDEO_EDIT_ACTION_SYSTEM_PROMPT,
                    critic_user_prompt=critic_user_prompt,
                )
            )
            step_record = result.steps[-1]
            log(f"[{trajectory_id}] score={candidate.score:.3f} action={action} prompt={edit_prompt}")

            if action in {"STOP", "LOOKS_GOOD"}:
                break
            if step_index + 1 >= self.iterations:
                step_record.notes["budget_exhausted_before_applying_action"] = True
                break

            next_path = output_dir / f"step_{step_index + 1}.mp4"
            if action == "FRESH_START":
                step_record.notes["next_output_video_path"] = str(next_path)
                current_video = _retry(
                    "fresh_start",
                    lambda: self.generator.generate(
                        prompt,
                        next_path,
                        seed=self._seed_for(seed_offset + 2000 + step_index),
                    ),
                )
                previous_edit_prompts.append("FRESH_START")
            elif action == "EASY_FRESH_START":
                step_record.notes["next_output_video_path"] = str(next_path)
                current_video = _retry(
                    "easy_fresh_start",
                    lambda: self.generator.generate(
                        edit_prompt,
                        next_path,
                        seed=self._seed_for(seed_offset + 2000 + step_index),
                    ),
                )
                previous_edit_prompts.append(f"EASY_FRESH_START: {edit_prompt}")
            else:
                source_video = current_video
                if action == "BACKTRACK" and len(previous_videos) > 1:
                    source_video = previous_videos[-2]
                    previous_videos = previous_videos[:-1]
                    previous_edit_prompts = previous_edit_prompts[:-1]
                elif action not in {"CONTINUE", "BACKTRACK"}:
                    action = "CONTINUE"
                    step_record.action = action
                elif best_candidate is not None:
                    source_video = best_candidate.video_path
                    if source_video != current_video:
                        log(
                            f"[{trajectory_id}] using best-scoring video "
                            f"(score={best_candidate.score:.3f}) as edit source"
                        )
                step_record.source_video_path = source_video
                step_record.notes["next_output_video_path"] = str(next_path)
                current_video = _retry(
                    "video_edit",
                    lambda: self.editor.edit(
                        edit_prompt,
                        source_video,
                        next_path,
                        seed=self._seed_for(seed_offset + 3000 + step_index),
                    ),
                )
                previous_edit_prompts.append(edit_prompt)
            previous_videos.append(current_video)
            step_index += 1

        result.best_candidate = best_candidate
        write_json(output_dir / "trajectory.json", result.to_json())
        return result

    def _run_refine_loop(
        self,
        current_video: str,
        full_prompt: str,
        core_prompt: str,
        completed_steps: list[str],
        current_step_desc: str,
        questions: list[str],
        all_edit_prompts: list[str],
        output_dir: Path,
        trajectory_id: str,
        step_counter: int,
        max_refine: int,
        best_candidate: Candidate | None,
        result: TrajectoryResult,
        label: str,
        gen_prompt: str | None = None,
        retry_edit_source: str | None = None,
        retry_edit_default_prompt: str | None = None,
        seed_offset: int = 0,
        is_core_step: bool = False,
    ) -> tuple[str, int, Candidate | None, str | None]:
        best_in_step: Candidate | None = None
        final_action: str | None = None
        attempt = 0
        while attempt < max_refine:
            raw_verifier, scores = _retry(
                "verifier", lambda: self._verify(self.verifier, current_video, questions)
            )
            candidate = self._candidate(
                current_video,
                scores,
                trajectory_id,
                step_counter,
                current_step_desc,
                f"{label}_VERIFY",
            )
            result.candidates.append(candidate)
            if best_candidate is None or candidate.score >= best_candidate.score:
                best_candidate = candidate
            if best_in_step is None or candidate.score >= best_in_step.score:
                best_in_step = candidate

            raw_critic, action, edit_prompt, system_prompt, user_prompt = _retry(
                "critic",
                lambda: self._step_critic(
                    current_video,
                    full_prompt,
                    core_prompt,
                    completed_steps,
                    current_step_desc,
                    scores,
                    all_edit_prompts,
                    attempt,
                    max_refine,
                    is_core_step,
                ),
            )
            final_action = action
            result.steps.append(
                StepRecord(
                    step_index=step_counter,
                    action=action,
                    prompt=edit_prompt,
                    video_path=current_video,
                    source_video_path=None,
                    verifier_scores=scores,
                    raw_verifier_response=raw_verifier,
                    raw_critic_response=raw_critic,
                    critic_system_prompt=system_prompt,
                    critic_user_prompt=user_prompt,
                    notes={"label": label, "attempt": attempt},
                )
            )
            step_counter += 1
            if action == "LOOKS_GOOD":
                break
            if attempt + 1 >= max_refine:
                result.steps[-1].notes["budget_exhausted_before_applying_action"] = True
                break

            next_path = output_dir / f"step_{step_counter}_{label}_{attempt}.mp4"
            if action == "RESAMPLE" and gen_prompt:
                current_video = _retry(
                    "resample",
                    lambda: self.generator.generate(
                        gen_prompt,
                        next_path,
                        seed=self._seed_for(seed_offset + 4000 + step_counter),
                    ),
                )
                all_edit_prompts.append(f"RESAMPLE: {gen_prompt}")
            elif action == "REPHRASE_AND_RETRY" and retry_edit_source:
                retry_prompt = (
                    edit_prompt
                    if edit_prompt and edit_prompt.lower() not in {"none", "n/a"}
                    else retry_edit_default_prompt
                )
                current_video = _retry(
                    "retry_edit",
                    lambda: self.editor.edit(
                        retry_prompt or current_step_desc,
                        retry_edit_source,
                        next_path,
                        seed=self._seed_for(seed_offset + 5000 + step_counter),
                    ),
                )
                all_edit_prompts.append(f"REPHRASE_AND_RETRY: {retry_prompt}")
            elif action == "REFINE":
                refine_prompt = (
                    edit_prompt
                    if edit_prompt and edit_prompt.lower() not in {"none", "n/a"}
                    else current_step_desc
                )
                current_video = _retry(
                    "refine_edit",
                    lambda: self.editor.edit(
                        refine_prompt,
                        current_video,
                        next_path,
                        seed=self._seed_for(seed_offset + 6000 + step_counter),
                    ),
                )
                all_edit_prompts.append(refine_prompt)
            else:
                break
            attempt += 1

        video_for_next = best_in_step.video_path if best_in_step else current_video
        return video_for_next, step_counter, best_candidate, final_action

    def run_single_step_by_step_trajectory(
        self,
        prompt: str,
        plan: dict[str, Any],
        output_dir: str | Path,
        trajectory_id: str,
        seed_offset: int,
    ) -> TrajectoryResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = TrajectoryResult(trajectory_id=trajectory_id, output_dir=str(output_dir))
        best_candidate: Candidate | None = None
        all_edit_prompts: list[str] = []
        completed_steps: list[str] = []
        step_counter = 0
        max_refine = int(self.method_cfg.get("max_refine_per_step", 2))

        core_prompt = plan["core_prompt"]
        core_questions = list(plan.get("core_questions", []))
        add_steps = list(plan.get("add_steps", []))

        current_video = _retry(
            "core_gen",
            lambda: self.generator.generate(
                core_prompt,
                output_dir / "step_0_core.mp4",
                seed=self._seed_for(seed_offset + 1000),
            ),
        )
        result.steps.append(
            StepRecord(
                step_index=step_counter,
                action="CORE_GENERATION",
                prompt=core_prompt,
                video_path=current_video,
                source_video_path=None,
            )
        )
        step_counter += 1

        if core_questions:
            current_video, step_counter, best_candidate, _ = self._run_refine_loop(
                current_video,
                prompt,
                core_prompt,
                completed_steps,
                f"Core generation: {core_prompt}",
                core_questions,
                all_edit_prompts,
                output_dir,
                trajectory_id,
                step_counter,
                max_refine,
                best_candidate,
                result,
                "core",
                gen_prompt=core_prompt,
                seed_offset=seed_offset,
                is_core_step=True,
            )

        cumulative_questions = list(core_questions)
        for add_idx, add_step in enumerate(add_steps):
            edit_prompt = add_step["edit_prompt"]
            step_questions = list(add_step.get("questions", []))
            cumulative_questions.extend(step_questions)
            pre_add_video = current_video
            current_video = _retry(
                "add_edit",
                lambda: self.editor.edit(
                    edit_prompt,
                    pre_add_video,
                    output_dir / f"step_{step_counter}_add{add_idx + 1}.mp4",
                    seed=self._seed_for(seed_offset + 2000 + step_counter),
                ),
            )
            all_edit_prompts.append(edit_prompt)
            result.steps.append(
                StepRecord(
                    step_index=step_counter,
                    action="ADD_STEP",
                    prompt=edit_prompt,
                    video_path=current_video,
                    source_video_path=pre_add_video,
                    notes={"add_step_index": add_idx},
                )
            )
            step_counter += 1

            if cumulative_questions:
                current_video, step_counter, best_candidate, _ = self._run_refine_loop(
                    current_video,
                    prompt,
                    core_prompt,
                    completed_steps,
                    edit_prompt,
                    cumulative_questions,
                    all_edit_prompts,
                    output_dir,
                    trajectory_id,
                    step_counter,
                    max_refine,
                    best_candidate,
                    result,
                    f"add{add_idx + 1}",
                    retry_edit_source=pre_add_video,
                    retry_edit_default_prompt=edit_prompt,
                    seed_offset=seed_offset,
                    is_core_step=False,
                )
            completed_steps.append(edit_prompt)

        all_questions = list(core_questions)
        for add_step in add_steps:
            all_questions.extend(add_step.get("questions", []))
        if all_questions:
            raw_verifier, scores = _retry(
                "final_verifier", lambda: self._verify(self.verifier, current_video, all_questions)
            )
            final_candidate = self._candidate(
                current_video,
                scores,
                trajectory_id,
                step_counter,
                prompt,
                "FINAL_VERIFY",
            )
            result.candidates.append(final_candidate)
            result.steps.append(
                StepRecord(
                    step_index=step_counter,
                    action="FINAL_VERIFY",
                    prompt=prompt,
                    video_path=current_video,
                    source_video_path=None,
                    verifier_scores=scores,
                    raw_verifier_response=raw_verifier,
                )
            )
            if best_candidate is None or final_candidate.score >= best_candidate.score:
                best_candidate = final_candidate

        result.best_candidate = best_candidate
        write_json(output_dir / "trajectory.json", result.to_json())
        return result

    def run_parallel_baseline(
        self,
        prompt: str,
        questions: list[str],
        output_dir: str | Path,
        num_samples: int,
    ) -> list[Candidate]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def run_sample(idx: int) -> Candidate:
            runner = IterativeVideoGenRunner(self.config)
            video_path = runner.generator.generate(
                prompt,
                output_dir / f"sample_{idx}.mp4",
                seed=runner._seed_for(10000 + idx),
            )
            _, scores = runner._verify(runner.verifier, video_path, questions)
            return runner._candidate(
                video_path,
                scores,
                trajectory_id="parallel_baseline",
                step_index=idx,
                prompt=prompt,
                action="PARALLEL_SAMPLE",
            )

        if self._worker_count(num_samples) == 1:
            candidates = [run_sample(idx) for idx in range(num_samples)]
        else:
            by_idx: dict[int, Candidate] = {}
            with ThreadPoolExecutor(max_workers=self._worker_count(num_samples)) as executor:
                futures = {executor.submit(run_sample, idx): idx for idx in range(num_samples)}
                for future in as_completed(futures):
                    idx = futures[future]
                    by_idx[idx] = future.result()
            candidates = [by_idx[idx] for idx in sorted(by_idx)]
        write_json(output_dir / "parallel_candidates.json", [c.to_json() for c in candidates])
        return candidates

    def _apply_eval_verifier(
        self,
        candidates: list[Candidate],
        questions: list[str],
    ) -> Candidate | None:
        if not self.eval_verifier or not candidates:
            return None
        best: Candidate | None = None
        for candidate in candidates:
            _, scores = self._verify(self.eval_verifier, candidate.video_path, questions)
            candidate.eval_verifier_scores = scores
            candidate.eval_score = score_from_scores(scores)
            if best is None or candidate.eval_score >= (best.eval_score or -1):
                best = candidate
        return best

    def run(self, prompt: str, questions: list[str], plan: dict[str, Any] | None = None) -> RunResult:
        mode = str(self.run_cfg.get("mode", "step_by_step"))
        output_dir = Path(self.run_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        write_json(output_dir / "config.resolved.json", _sanitize_config(self.config))
        write_json(output_dir / "questions.json", questions)
        if plan:
            write_json(output_dir / "plan.json", plan)

        result = RunResult(
            output_dir=str(output_dir),
            prompt=prompt,
            questions=questions,
            mode=mode,
            plan=plan,
        )
        all_candidates: list[Candidate] = []

        if mode in {"parallel", "all"}:
            num_samples = self.iterations * self.parallel
            parallel_candidates = self.run_parallel_baseline(
                prompt, questions, output_dir / "parallel", num_samples
            )
            result.parallel_candidates = parallel_candidates
            all_candidates.extend(parallel_candidates)

        if mode in {"iterative", "iterative_parallel", "iter_parallel", "all"}:
            num_streams = self.parallel if mode in {"iterative_parallel", "iter_parallel", "all"} else 1
            trajectories = self._run_streams(
                num_streams,
                lambda runner, stream_idx: runner.run_single_iterative_trajectory(
                    prompt,
                    questions,
                    output_dir / f"trajectory_{stream_idx}",
                    f"trajectory_{stream_idx}",
                    stream_idx * 100,
                ),
            )
            result.trajectories.extend(trajectories)
            for trajectory in trajectories:
                all_candidates.extend(trajectory.candidates)

        if mode in {"step_by_step", "step_by_step_parallel"}:
            if plan is None:
                raise ValueError("step_by_step modes require a plan.")
            num_streams = self.parallel if mode == "step_by_step_parallel" else 1
            trajectories = self._run_streams(
                num_streams,
                lambda runner, stream_idx: runner.run_single_step_by_step_trajectory(
                    prompt,
                    plan,
                    output_dir / f"trajectory_{stream_idx}",
                    f"trajectory_{stream_idx}",
                    stream_idx * 100,
                ),
            )
            result.trajectories.extend(trajectories)
            for trajectory in trajectories:
                all_candidates.extend(trajectory.candidates)

        if all_candidates:
            result.best_candidate = max(all_candidates, key=lambda candidate: candidate.score)
            result.best_eval_candidate = self._apply_eval_verifier(all_candidates, questions)

        write_json(output_dir / "summary.json", result.to_json())
        return result

    def _run_streams(
        self,
        num_streams: int,
        stream_fn: Callable[["IterativeVideoGenRunner", int], TrajectoryResult],
    ) -> list[TrajectoryResult]:
        if self._worker_count(num_streams) == 1:
            return [stream_fn(self, stream_idx) for stream_idx in range(num_streams)]

        by_idx: dict[int, TrajectoryResult] = {}
        with ThreadPoolExecutor(max_workers=self._worker_count(num_streams)) as executor:
            futures = {}
            for stream_idx in range(num_streams):
                runner = IterativeVideoGenRunner(self.config)
                futures[executor.submit(stream_fn, runner, stream_idx)] = stream_idx
            for future in as_completed(futures):
                stream_idx = futures[future]
                by_idx[stream_idx] = future.result()
        return [by_idx[idx] for idx in sorted(by_idx)]


def prepare_questions(config: dict[str, Any], prompt: str, critic: VLMClient | None = None) -> list[str]:
    question_cfg = config.get("questions", {})
    questions = normalize_questions(question_cfg.get("items"))
    if question_cfg.get("path"):
        questions.extend(load_questions_from_path(question_cfg["path"]))
    if not questions and question_cfg.get("auto", True):
        question_client = critic or build_vlm_client(config["models"]["critic"])
        questions = question_client.generate_questions(prompt)
    if not questions:
        questions = [f"Does the video satisfy this prompt: {prompt}"]
    return questions


def make_step_by_step_plan(
    prompt: str,
    questions: list[str],
    planner: VLMClient,
    prompt_style: str = "embellished",
    max_add_steps: int = 2,
) -> dict[str, Any]:
    raw_plan = planner.complete(get_plan_system_prompt(prompt_style), prompt)
    plan = json.loads(strip_json_fence(raw_plan))
    if "core_prompt" not in plan or "add_steps" not in plan:
        raise ValueError(f"Unexpected plan format: {plan}")
    add_step_prompts = []
    for step in plan["add_steps"]:
        if isinstance(step, str):
            add_step_prompts.append(step)
        elif isinstance(step, dict):
            add_step_prompts.append(step.get("edit_prompt", str(step)))
        else:
            add_step_prompts.append(str(step))
    if max_add_steps >= 0:
        add_step_prompts = add_step_prompts[:max_add_steps]

    questions_str = "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(questions))
    plan_str = json.dumps(
        {"core_prompt": plan["core_prompt"], "add_steps": add_step_prompts},
        indent=2,
    )
    mapping_raw = planner.complete(
        QUESTION_MAPPING_SYSTEM_PROMPT,
        f"Questions:\n{questions_str}\n\nPlan:\n{plan_str}",
    )
    mapping = json.loads(strip_json_fence(mapping_raw))

    core_indices = [int(idx) - 1 for idx in mapping.get("core", [])]
    core_questions = [questions[idx] for idx in core_indices if 0 <= idx < len(questions)]
    assigned = set(core_indices)
    step_questions: list[list[str]] = []
    for step_idx in range(len(add_step_prompts)):
        key = f"add_step_{step_idx + 1}"
        indices = [int(idx) - 1 for idx in mapping.get(key, [])]
        assigned.update(indices)
        step_questions.append([questions[idx] for idx in indices if 0 <= idx < len(questions)])
    for idx, question in enumerate(questions):
        if idx not in assigned:
            core_questions.append(question)

    enriched = {
        "core_prompt": plan["core_prompt"],
        "core_questions": core_questions,
        "add_steps": [],
        "raw_plan_response": raw_plan,
        "raw_question_mapping_response": mapping_raw,
        "max_add_steps": max_add_steps,
    }
    for idx, edit_prompt in enumerate(add_step_prompts):
        enriched["add_steps"].append(
            {
                "edit_prompt": edit_prompt,
                "questions": step_questions[idx] if idx < len(step_questions) else [],
            }
        )
    return enriched
