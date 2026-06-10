from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ScoreList = list[tuple[str, float]]


@dataclass
class Candidate:
    image_path: str
    score: float
    verifier_scores: ScoreList
    trajectory_id: str
    step_index: int
    prompt: str
    action: str
    eval_score: float | None = None
    eval_verifier_scores: ScoreList | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    step_index: int
    action: str
    prompt: str
    image_path: str
    source_image_path: str | None
    verifier_scores: ScoreList
    raw_critic_response: str | None = None
    first_step_system_prompt: str | None = None
    first_step_user_prompt: str | None = None
    next_step_system_prompt: str | None = None
    next_step_user_prompt: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryResult:
    trajectory_id: str
    output_dir: str
    steps: list[StepRecord] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    best_candidate: Candidate | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "output_dir": self.output_dir,
            "steps": [step.to_json() for step in self.steps],
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "best_candidate": self.best_candidate.to_json()
            if self.best_candidate
            else None,
        }


@dataclass
class RunResult:
    output_dir: str
    prompt: str
    questions: list[str]
    mode: str
    trajectories: list[TrajectoryResult] = field(default_factory=list)
    parallel_candidates: list[Candidate] = field(default_factory=list)
    best_candidate: Candidate | None = None
    best_eval_candidate: Candidate | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "prompt": self.prompt,
            "questions": self.questions,
            "mode": self.mode,
            "trajectories": [trajectory.to_json() for trajectory in self.trajectories],
            "parallel_candidates": [
                candidate.to_json() for candidate in self.parallel_candidates
            ],
            "best_candidate": self.best_candidate.to_json()
            if self.best_candidate
            else None,
            "best_eval_candidate": self.best_eval_candidate.to_json()
            if self.best_eval_candidate
            else None,
        }

