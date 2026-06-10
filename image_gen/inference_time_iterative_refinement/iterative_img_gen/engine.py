from __future__ import annotations

import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .image_providers import ImageProvider, build_image_provider
from .io_utils import parse_first_prompt, parse_next_step, write_json
from .prompts import FIRST_STEP_TEMPLATES, NEXT_STEP_TEMPLATES
from .schemas import Candidate, RunResult, StepRecord, TrajectoryResult
from .vlm_providers import VLMClient, build_vlm_client


def _scores_to_text(verifier_questions_and_scores: list[tuple[str, float]]) -> str:
    verifier_scores_prompt = ""
    for question, score in verifier_questions_and_scores:
        verifier_scores_prompt += f"\t {question}: {score}\n"
    return verifier_scores_prompt


def _history_to_text(step_prompts: list[str]) -> str:
    entire_edit_history_prompt = ""
    for step_num, step_prompt in enumerate(step_prompts):
        entire_edit_history_prompt += f"\tStep {step_num+1}: {step_prompt}\n"
    return entire_edit_history_prompt


def _initial_prompts(complex_prompt: str, method_cfg: dict[str, Any], iterations: int) -> tuple[str, str]:
    specify_num_steps_in_prompt = method_cfg["specify_num_steps_in_prompt"]
    if specify_num_steps_in_prompt:
        edit_steps_prompt = "The maximum number of editing steps is {num_max_edit_steps}. This will be the first step of image generation. Decide the first step prompt accordingly.".format(
            num_max_edit_steps=iterations
        )
    else:
        edit_steps_prompt = ""
    first_template = FIRST_STEP_TEMPLATES[method_cfg["first_step_prompt_style"]]
    system_prompt_filled = first_template.format(edit_steps_prompt=edit_steps_prompt)
    user_prompt_inputs = """Full complex prompt: {complex_prompt}""".format(
        complex_prompt=complex_prompt
    )
    return system_prompt_filled, user_prompt_inputs


def _next_step_prompts(
    complex_prompt: str,
    previous_step_prompts: list[str],
    previous_step_verifier_scores: list[list[tuple[str, float]]],
    method_cfg: dict[str, Any],
    iterations: int,
) -> tuple[str, str]:
    if method_cfg["use_decision_making"]:
        raise ValueError("Decision making is not supported by this release runner.")

    provide_image_in_subsequent_steps = method_cfg["provide_image_in_subsequent_steps"]
    provide_entire_edit_history = method_cfg["provide_entire_edit_history"]
    specify_num_steps_in_prompt = method_cfg["specify_num_steps_in_prompt"]
    num_steps_completed = len(previous_step_prompts)
    remaining_num_steps = iterations - num_steps_completed

    if specify_num_steps_in_prompt:
        if remaining_num_steps == 1:
            edit_steps_prompt = "This is the last step of image editing. Decide the next step prompt accordingly to complete the entire task."
        else:
            edit_steps_prompt = "The maximum number of editing steps is {num_max_edit_steps}. This is step {step_number} of image editing and you will have {step_number_left} steps left to complete the task. Decide the next step prompt accordingly.".format(
                num_max_edit_steps=iterations,
                step_number=num_steps_completed,
                step_number_left=remaining_num_steps,
            )
    else:
        edit_steps_prompt = ""

    if provide_image_in_subsequent_steps:
        with_image_prompt = "previously generated image along with verifier scores (sometimes verifier can be wrong for attribute counts questions)"
        with_image_prompt_attached = " (which is attached for your reference)"
    else:
        with_image_prompt = "verifier scores for previously generated image"
        with_image_prompt_attached = ""

    verifier_scores_string = _scores_to_text(previous_step_verifier_scores[-1])
    step_prompts_history_string = _history_to_text(previous_step_prompts)

    if provide_entire_edit_history:
        entire_edit_history_prompt = """Your previous step prompts were:\n{step_prompts_history_string}\nThe most recently generated image{with_image_prompt_attached} had the following verifier scores:\n{verifier_scores_string}""".format(
            step_prompts_history_string=step_prompts_history_string,
            with_image_prompt_attached=with_image_prompt_attached,
            verifier_scores_string=verifier_scores_string,
        )
    else:
        entire_edit_history_prompt = """Your previous step prompt was: {previous_step_prompt}
            The most recently generated image {with_image_prompt_attached} had the following verifier scores:\n{verifier_scores_string}""".format(
            previous_step_prompt=previous_step_prompts[-1],
            with_image_prompt_attached=with_image_prompt_attached,
            verifier_scores_string=verifier_scores_string,
        )

    if provide_entire_edit_history and provide_image_in_subsequent_steps:
        following_inputs_str = """- Your previously proposed step prompts\n- The most recently generated image {with_image_prompt_attached} along with verifier scores (sometimes verifier can be wrong for attribute counts questions)""".format(
            with_image_prompt_attached=with_image_prompt_attached
        )
    elif provide_entire_edit_history and not provide_image_in_subsequent_steps:
        following_inputs_str = """- Your previously proposed step prompts\n- The verifier scores for the most recently generated image (sometimes verifier can be wrong for attribute counts questions)"""
    elif not provide_entire_edit_history and provide_image_in_subsequent_steps:
        following_inputs_str = """- Your most recently proposed step prompt
            - The most recently generated image {with_image_prompt_attached} along with verifier scores (sometimes verifier can be wrong for attribute counts questions)""".format(
            with_image_prompt_attached=with_image_prompt_attached
        )
    else:
        following_inputs_str = """- Your most recently proposed step prompt
            - The verifier scores for the most recently generated image (sometimes verifier can be wrong for attribute counts questions)"""

    user_prompt_inputs = """Full complex prompt: {complex_prompt}\n{entire_edit_history_prompt}\n{edit_steps_prompt}""".format(
        complex_prompt=complex_prompt,
        entire_edit_history_prompt=entire_edit_history_prompt,
        edit_steps_prompt=edit_steps_prompt,
    )

    next_template = NEXT_STEP_TEMPLATES[method_cfg["next_step_prompt_style"]]
    system_prompt = next_template.format(
        edit_steps_prompt=edit_steps_prompt,
        with_image_prompt=with_image_prompt,
        following_inputs_str=following_inputs_str,
    )
    return system_prompt, user_prompt_inputs


def _score_from_scores(scores: list[tuple[str, float]]) -> float:
    return float(scores[-1][1])


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


class IterativeImageGenRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.run_cfg = self.config["run"]
        self.method_cfg = self.config["method"]
        self.generation_cfg = self.config["generation"]
        self.iterations = int(self.run_cfg["iterations"])
        self.parallel = int(self.run_cfg["parallel"])
        self.seed = self.run_cfg.get("seed")
        configured_max_workers = self.run_cfg.get("max_workers")
        self.max_workers = int(configured_max_workers) if configured_max_workers else None

        self.generator = build_image_provider(
            self.config["models"]["generator"], self.generation_cfg
        )
        self.editor = build_image_provider(self.config["models"]["editor"], self.generation_cfg)
        self.critic = build_vlm_client(self.config["models"]["critic"])
        self.verifier = build_vlm_client(self.config["models"]["verifier"])
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

    def _verify(self, verifier: VLMClient, image_path: str, questions: list[str]) -> list[tuple[str, float]]:
        return verifier.verify(image_path, questions)

    def _candidate(
        self,
        image_path: str,
        scores: list[tuple[str, float]],
        trajectory_id: str,
        step_index: int,
        prompt: str,
        action: str,
    ) -> Candidate:
        return Candidate(
            image_path=image_path,
            score=_score_from_scores(scores),
            verifier_scores=scores,
            trajectory_id=trajectory_id,
            step_index=step_index,
            prompt=prompt,
            action=action,
        )

    def run_single_trajectory(
        self,
        prompt: str,
        questions: list[str],
        output_dir: str | Path,
        trajectory_id: str,
        seed_offset: int,
    ) -> TrajectoryResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        previous_step_prompts: list[str] = []
        previous_step_images: list[str] = []
        previous_step_verifier_scores: list[list[tuple[str, float]]] = []

        result = TrajectoryResult(trajectory_id=trajectory_id, output_dir=str(output_dir))

        rephrase_first_step = bool(self.method_cfg.get("rephrase_first_step", False))
        if not rephrase_first_step:
            first_prompt = prompt
            raw_first_response = None
            first_system_prompt = None
            first_user_prompt = None
        else:
            first_system_prompt, first_user_prompt = _initial_prompts(
                prompt, self.method_cfg, self.iterations
            )
            raw_first_response = self.critic.complete(
                first_system_prompt,
                first_user_prompt,
                seed=self._seed_for(seed_offset),
            )
            first_prompt = parse_first_prompt(raw_first_response)

        first_image_path = str(output_dir / "step_0.png")
        first_image_path = self.generator.generate(
            first_prompt,
            first_image_path,
            seed=self._seed_for(seed_offset + 1000),
        )
        first_scores = self._verify(self.verifier, first_image_path, questions)

        previous_step_prompts.append(first_prompt)
        previous_step_images.append(first_image_path)
        previous_step_verifier_scores.append(first_scores)

        first_step = StepRecord(
            step_index=0,
            action="START",
            prompt=first_prompt,
            image_path=first_image_path,
            source_image_path=None,
            verifier_scores=first_scores,
            raw_critic_response=raw_first_response,
            first_step_system_prompt=first_system_prompt,
            first_step_user_prompt=first_user_prompt,
        )
        result.steps.append(first_step)
        first_candidate = self._candidate(
            first_image_path, first_scores, trajectory_id, 0, first_prompt, "START"
        )
        result.candidates.append(first_candidate)
        best_candidate = first_candidate

        num_steps_completed = 1
        while num_steps_completed < self.iterations:
            next_system_prompt, next_user_prompt = _next_step_prompts(
                prompt,
                previous_step_prompts,
                previous_step_verifier_scores,
                self.method_cfg,
                self.iterations,
            )
            critic_image = (
                previous_step_images[-1]
                if self.method_cfg["provide_image_in_subsequent_steps"]
                else None
            )
            raw_next_response = self.critic.complete(
                next_system_prompt,
                next_user_prompt,
                image_path=critic_image,
                seed=self._seed_for(seed_offset + num_steps_completed),
            )
            action, next_prompt = parse_next_step(raw_next_response)

            if action == "STOP":
                stop_scores = self._verify(self.verifier, previous_step_images[-1], questions)
                stop_candidate = self._candidate(
                    previous_step_images[-1],
                    stop_scores,
                    trajectory_id,
                    num_steps_completed,
                    previous_step_prompts[-1],
                    "STOP",
                )
                result.candidates.append(stop_candidate)
                if stop_candidate.score >= best_candidate.score:
                    best_candidate = stop_candidate
                break

            source_image_path: str | None
            if action == "BACKTRACK" and len(previous_step_images) > 1:
                source_image_path = previous_step_images[-2]
                previous_step_images = previous_step_images[:-1]
                previous_step_prompts = previous_step_prompts[:-1]
                previous_step_verifier_scores = previous_step_verifier_scores[:-1]
            else:
                source_image_path = previous_step_images[-1]

            output_image_path = str(output_dir / f"step_{num_steps_completed}.png")
            image_seed = self._seed_for(seed_offset + 1000 + num_steps_completed)

            if action == "FRESH_START":
                output_image_path = self.generator.generate(
                    next_prompt,
                    output_image_path,
                    seed=image_seed,
                )
                stored_prompt = "START AGAIN with prompt: " + next_prompt
                source_image_path = None
            else:
                output_image_path = self.editor.edit(
                    next_prompt,
                    source_image_path,
                    output_image_path,
                    seed=image_seed,
                )
                stored_prompt = next_prompt

            verifier_scores = self._verify(self.verifier, output_image_path, questions)
            previous_step_prompts.append(stored_prompt)
            previous_step_images.append(output_image_path)
            previous_step_verifier_scores.append(verifier_scores)

            step_record = StepRecord(
                step_index=num_steps_completed,
                action=action,
                prompt=next_prompt,
                image_path=output_image_path,
                source_image_path=source_image_path,
                verifier_scores=verifier_scores,
                raw_critic_response=raw_next_response,
                next_step_system_prompt=next_system_prompt,
                next_step_user_prompt=next_user_prompt,
            )
            result.steps.append(step_record)

            candidate = self._candidate(
                output_image_path,
                verifier_scores,
                trajectory_id,
                num_steps_completed,
                next_prompt,
                action,
            )
            result.candidates.append(candidate)
            if candidate.score >= best_candidate.score:
                best_candidate = candidate

            num_steps_completed += 1

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
            runner = IterativeImageGenRunner(self.config)
            image_path = str(output_dir / f"sample_{idx}.png")
            image_path = runner.generator.generate(
                prompt,
                image_path,
                seed=runner._seed_for(10000 + idx),
            )
            scores = runner._verify(runner.verifier, image_path, questions)
            return runner._candidate(
                image_path,
                scores,
                trajectory_id="parallel_baseline",
                step_index=idx,
                prompt=prompt,
                action="PARALLEL_SAMPLE",
            )

        max_workers = self._worker_count(num_samples)
        if max_workers == 1:
            candidates = []
            for idx in range(num_samples):
                image_path = str(output_dir / f"sample_{idx}.png")
                image_path = self.generator.generate(
                    prompt,
                    image_path,
                    seed=self._seed_for(10000 + idx),
                )
                scores = self._verify(self.verifier, image_path, questions)
                candidates.append(
                    self._candidate(
                        image_path,
                        scores,
                        trajectory_id="parallel_baseline",
                        step_index=idx,
                        prompt=prompt,
                        action="PARALLEL_SAMPLE",
                    )
                )
        else:
            by_idx: dict[int, Candidate] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            scores = self._verify(self.eval_verifier, candidate.image_path, questions)
            candidate.eval_verifier_scores = scores
            candidate.eval_score = _score_from_scores(scores)
            if best is None or candidate.eval_score >= (best.eval_score or -1):
                best = candidate
        return best

    def run(self, prompt: str, questions: list[str]) -> RunResult:
        mode = str(self.run_cfg.get("mode", "iterative_parallel"))
        output_dir = Path(self.run_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        write_json(output_dir / "config.resolved.json", _sanitize_config(self.config))
        write_json(output_dir / "questions.json", questions)

        result = RunResult(
            output_dir=str(output_dir),
            prompt=prompt,
            questions=questions,
            mode=mode,
        )

        all_candidates: list[Candidate] = []

        if mode in {"parallel", "all"}:
            num_samples = self.iterations * self.parallel
            parallel_candidates = self.run_parallel_baseline(
                prompt,
                questions,
                output_dir / "parallel",
                num_samples=num_samples,
            )
            result.parallel_candidates = parallel_candidates
            all_candidates.extend(parallel_candidates)

        if mode in {"iterative", "iterative_parallel", "iter_parallel", "all"}:
            num_streams = self.parallel if mode in {"iterative_parallel", "iter_parallel", "all"} else 1

            def run_stream(stream_idx: int) -> TrajectoryResult:
                runner = IterativeImageGenRunner(self.config)
                return runner.run_single_trajectory(
                    prompt,
                    questions,
                    output_dir / f"trajectory_{stream_idx}",
                    trajectory_id=f"trajectory_{stream_idx}",
                    seed_offset=stream_idx * 100,
                )

            max_workers = self._worker_count(num_streams)
            if max_workers == 1:
                trajectories = [
                    self.run_single_trajectory(
                        prompt,
                        questions,
                        output_dir / f"trajectory_{stream_idx}",
                        trajectory_id=f"trajectory_{stream_idx}",
                        seed_offset=stream_idx * 100,
                    )
                    for stream_idx in range(num_streams)
                ]
            else:
                by_idx: dict[int, TrajectoryResult] = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(run_stream, stream_idx): stream_idx
                        for stream_idx in range(num_streams)
                    }
                    for future in as_completed(futures):
                        stream_idx = futures[future]
                        by_idx[stream_idx] = future.result()
                trajectories = [by_idx[idx] for idx in sorted(by_idx)]

            for trajectory in trajectories:
                result.trajectories.append(trajectory)
                all_candidates.extend(trajectory.candidates)

        if all_candidates:
            result.best_candidate = max(all_candidates, key=lambda candidate: candidate.score)
            result.best_eval_candidate = self._apply_eval_verifier(all_candidates, questions)

        write_json(output_dir / "summary.json", result.to_json())
        return result


def prepare_questions(config: dict[str, Any], prompt: str, critic: VLMClient | None = None) -> list[str]:
    from .config import load_questions_from_path, normalize_questions

    question_cfg = config.get("questions", {})
    questions = normalize_questions(question_cfg.get("items"))
    if question_cfg.get("path"):
        questions.extend(load_questions_from_path(question_cfg["path"]))
    if not questions and question_cfg.get("auto", True):
        question_client = critic or build_vlm_client(config["models"]["critic"])
        questions = question_client.generate_questions(
            prompt,
            tiif_prompt=bool(question_cfg.get("tiif_prompt", False)),
        )
    if not questions:
        questions = [f"Does the image satisfy this prompt: {prompt}"]
    return questions
