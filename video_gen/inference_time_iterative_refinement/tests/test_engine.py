from pathlib import Path

from iterative_video_gen.engine import IterativeVideoGenRunner
from iterative_video_gen.schemas import TrajectoryResult


class _FailingEditor:
    def edit(self, *args, **kwargs):
        raise AssertionError("editor should not run after refinement budget is exhausted")


class _FailingGenerator:
    def generate(self, *args, **kwargs):
        raise AssertionError("generator should not run after refinement budget is exhausted")


def test_refine_loop_does_not_apply_unverified_final_budget_action(tmp_path: Path):
    runner = IterativeVideoGenRunner.__new__(IterativeVideoGenRunner)
    runner.editor = _FailingEditor()
    runner.generator = _FailingGenerator()
    runner.verifier = object()
    runner._seed_for = lambda offset: None
    runner._verify = lambda verifier, video_path, questions: (
        "Is there a bird?: no",
        [("Is there a bird?", 0.0), ("Cumulative mean binary score", 0.0)],
    )
    runner._step_critic = lambda *args, **kwargs: (
        "Action: REFINE\nPrompt: Add a bird.",
        "REFINE",
        "Add a bird.",
        "system",
        "user",
    )

    result = TrajectoryResult(trajectory_id="trajectory_0", output_dir=str(tmp_path))

    video_for_next, step_counter, best_candidate, final_action = runner._run_refine_loop(
        current_video="current.mp4",
        full_prompt="A bird flies over a road.",
        core_prompt="A road.",
        completed_steps=[],
        current_step_desc="Add a bird.",
        questions=["Is there a bird?"],
        all_edit_prompts=[],
        output_dir=tmp_path,
        trajectory_id="trajectory_0",
        step_counter=0,
        max_refine=1,
        best_candidate=None,
        result=result,
        label="add1",
    )

    assert video_for_next == "current.mp4"
    assert step_counter == 1
    assert best_candidate is not None
    assert final_action == "REFINE"
    assert result.steps[0].notes["budget_exhausted_before_applying_action"] is True
