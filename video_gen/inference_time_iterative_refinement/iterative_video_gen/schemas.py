from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ScoreList = list[tuple[str, float]]


@dataclass
class Candidate:
    video_path: str
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
    video_path: str
    source_video_path: str | None
    verifier_scores: ScoreList | None = None
    raw_verifier_response: str | None = None
    raw_critic_response: str | None = None
    critic_system_prompt: str | None = None
    critic_user_prompt: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryResult:
    trajectory_id: str
    output_dir: str
    steps: list[StepRecord] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    best_candidate: Candidate | None = None
    text_log: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "output_dir": self.output_dir,
            "steps": [step.to_json() for step in self.steps],
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "best_candidate": self.best_candidate.to_json()
            if self.best_candidate
            else None,
            "text_log": self.text_log,
        }


@dataclass
class RunResult:
    output_dir: str
    prompt: str
    questions: list[str]
    mode: str
    plan: dict[str, Any] | None = None
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
            "plan": self.plan,
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
