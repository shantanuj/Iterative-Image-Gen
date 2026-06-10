from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import expand_env_vars, load_config, normalize_questions, set_nested
from .engine import IterativeVideoGenRunner, make_step_by_step_plan, prepare_questions


def _set_if_present(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    if value is not None:
        set_nested(cfg, dotted_key, value)


def _apply_run_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _set_if_present(cfg, "run.mode", args.mode)
    _set_if_present(cfg, "run.iterations", args.iterations)
    _set_if_present(cfg, "run.parallel", args.parallel)
    _set_if_present(cfg, "run.max_workers", args.max_workers)
    _set_if_present(cfg, "run.output_dir", args.output_dir)
    _set_if_present(cfg, "run.seed", args.seed)

    _set_if_present(cfg, "method.plan_prompt_style", args.plan_prompt_style)
    _set_if_present(cfg, "method.max_add_steps", args.max_add_steps)
    _set_if_present(cfg, "method.max_refine_per_step", args.max_refine_per_step)
    if (
        args.iterations is not None
        and args.max_refine_per_step is None
        and str(cfg["run"].get("mode", "")).startswith("step_by_step")
    ):
        set_nested(cfg, "method.max_refine_per_step", args.iterations)

    _set_if_present(cfg, "generation.resolution", args.resolution)
    _set_if_present(cfg, "generation.num_frames", args.num_frames)
    _set_if_present(cfg, "generation.num_steps", args.num_steps)
    _set_if_present(cfg, "generation.guidance_scale", args.guidance_scale)
    _set_if_present(cfg, "generation.flow_shift", args.flow_shift)
    _set_if_present(cfg, "generation.negative_prompt", args.negative_prompt)

    _set_if_present(cfg, "editing.num_frames", args.edit_num_frames)
    _set_if_present(cfg, "editing.num_steps", args.edit_num_steps)
    _set_if_present(cfg, "editing.height", args.height)
    _set_if_present(cfg, "editing.width", args.width)
    _set_if_present(cfg, "editing.guidance_scale", args.edit_guidance_scale)
    _set_if_present(cfg, "editing.image_guidance_scale", args.image_guidance_scale)
    _set_if_present(cfg, "editing.timestep_shift", args.timestep_shift)
    _set_if_present(cfg, "editing.teacache_thresh", args.teacache_thresh)
    if args.cfg_scale is not None:
        set_nested(cfg, "editing.cfg_scale", args.cfg_scale)
        set_nested(cfg, "editing.guidance_scale", args.cfg_scale)
    if args.edit_strength is not None:
        set_nested(cfg, "editing.edit_strength", args.edit_strength)
        set_nested(cfg, "editing.image_guidance_scale", args.edit_strength)
    if args.cfg_txt is not None:
        set_nested(cfg, "editing.cfg_txt", args.cfg_txt)
        set_nested(cfg, "editing.timestep_shift", args.cfg_txt)
    _set_if_present(cfg, "editing.negative_prompt", args.edit_negative_prompt)
    _set_if_present(cfg, "editing.start_frame", args.start_frame)

    _set_if_present(cfg, "models.generator.provider", args.base_generator)
    _set_if_present(cfg, "models.generator.model", args.generator_model)
    _set_if_present(cfg, "models.generator.base_url", args.generator_base_url)

    _set_if_present(cfg, "models.editor.provider", args.editor)
    _set_if_present(cfg, "models.editor.model", args.editor_model)
    _set_if_present(cfg, "models.editor.base_url", args.editor_base_url)

    _set_if_present(cfg, "models.critic.provider", args.critic_provider)
    _set_if_present(cfg, "models.critic.model", args.critic_model)
    _set_if_present(cfg, "models.critic.base_url", args.critic_base_url)
    _set_if_present(cfg, "models.critic.api_key_env", args.critic_api_key_env)
    _set_if_present(cfg, "models.critic.num_frames", args.critic_num_frames)
    if args.critic_frame_input:
        set_nested(cfg, "models.critic.use_native_video", False)
    if args.critic_native_video:
        set_nested(cfg, "models.critic.use_native_video", True)

    _set_if_present(cfg, "models.verifier.provider", args.verifier_provider)
    _set_if_present(cfg, "models.verifier.model", args.verifier_model)
    _set_if_present(cfg, "models.verifier.base_url", args.verifier_base_url)
    _set_if_present(cfg, "models.verifier.api_key_env", args.verifier_api_key_env)
    _set_if_present(cfg, "models.verifier.num_frames", args.verifier_num_frames)
    if args.verifier_frame_input:
        set_nested(cfg, "models.verifier.use_native_video", False)
    if args.verifier_native_video:
        set_nested(cfg, "models.verifier.use_native_video", True)

    if args.eval_verifier_provider or args.eval_verifier_model:
        eval_cfg = cfg["models"].get("eval_verifier") or {}
        if args.eval_verifier_provider:
            eval_cfg["provider"] = args.eval_verifier_provider
        if args.eval_verifier_model:
            eval_cfg["model"] = args.eval_verifier_model
        if args.eval_verifier_base_url:
            eval_cfg["base_url"] = args.eval_verifier_base_url
        if args.eval_verifier_api_key_env:
            eval_cfg["api_key_env"] = args.eval_verifier_api_key_env
        if args.eval_verifier_num_frames:
            eval_cfg["num_frames"] = args.eval_verifier_num_frames
        cfg["models"]["eval_verifier"] = eval_cfg
    if args.no_eval_verifier:
        cfg["models"]["eval_verifier"] = None

    if args.questions:
        question_arg = args.questions
        path = Path(question_arg)
        if path.exists():
            set_nested(cfg, "questions.path", str(path))
        else:
            cfg["questions"]["items"] = normalize_questions(question_arg)
    if args.questions_file:
        set_nested(cfg, "questions.path", args.questions_file)
    if args.auto_questions:
        set_nested(cfg, "questions.auto", True)
    if args.no_auto_questions:
        set_nested(cfg, "questions.auto", False)

    return expand_env_vars(cfg)


def run_cmd(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg = _apply_run_overrides(cfg, args)
    runner = IterativeVideoGenRunner(cfg)
    questions = prepare_questions(cfg, args.prompt, critic=runner.critic)
    plan = None
    if str(cfg["run"].get("mode", "")).startswith("step_by_step"):
        plan = make_step_by_step_plan(
            args.prompt,
            questions,
            runner.critic,
            prompt_style=cfg["method"].get("plan_prompt_style", "embellished"),
            max_add_steps=int(cfg["method"].get("max_add_steps", 2)),
        )
    result = runner.run(args.prompt, questions, plan=plan)

    print(f"Output directory: {result.output_dir}")
    if result.best_candidate:
        print(f"Best video: {result.best_candidate.video_path}")
        print(f"Best loop-verifier score: {result.best_candidate.score:.4f}")
    if result.best_eval_candidate:
        print(f"Best eval-verifier video: {result.best_eval_candidate.video_path}")
        print(f"Best eval-verifier score: {result.best_eval_candidate.eval_score:.4f}")
    print(f"Summary JSON: {Path(result.output_dir) / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iterative-video-gen",
        description="Run iterative refinement for compositional video generation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run video generation from a prompt.")
    run.add_argument("--prompt", required=True, help="Full compositional video prompt.")
    run.add_argument("--config", default=None, help="YAML config path.")
    run.add_argument(
        "--mode",
        choices=[
            "step_by_step",
            "step_by_step_parallel",
            "iterative",
            "iterative_parallel",
            "parallel",
            "all",
        ],
        default=None,
    )
    run.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Iterative-mode video-call budget. In step_by_step modes, also sets "
            "--max-refine-per-step unless that flag is provided."
        ),
    )
    run.add_argument("--parallel", type=int, default=None, help="Number of independent streams.")
    run.add_argument("--max-workers", type=int, default=None, help="Maximum concurrent streams/samples.")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--seed", type=int, default=None)

    run.add_argument("--plan-prompt-style", choices=["embellished", "simple"], default=None)
    run.add_argument("--max-add-steps", type=int, default=None)
    run.add_argument("--max-refine-per-step", type=int, default=None)

    run.add_argument("--resolution", default=None, help="Wan resolution string, e.g. 832*480.")
    run.add_argument("--num-frames", type=int, default=None)
    run.add_argument("--num-steps", type=int, default=None)
    run.add_argument("--guidance-scale", type=float, default=None)
    run.add_argument("--flow-shift", type=float, default=None)
    run.add_argument("--negative-prompt", default=None)

    run.add_argument("--edit-num-frames", type=int, default=None)
    run.add_argument("--edit-num-steps", type=int, default=None)
    run.add_argument("--height", type=int, default=None)
    run.add_argument("--width", type=int, default=None)
    run.add_argument("--edit-guidance-scale", type=float, default=None)
    run.add_argument("--image-guidance-scale", type=float, default=None)
    run.add_argument("--timestep-shift", type=float, default=None)
    run.add_argument("--teacache-thresh", type=float, default=None)
    run.add_argument("--cfg-scale", type=float, default=None)
    run.add_argument("--edit-strength", type=float, default=None)
    run.add_argument("--cfg-txt", type=float, default=None)
    run.add_argument("--edit-negative-prompt", default=None)
    run.add_argument("--start-frame", type=int, default=None)

    run.add_argument(
        "--base-generator",
        default=None,
        help="wan-t2v-gradio, wan2.1-t2v-gradio, wan2.2-t2v-gradio.",
    )
    run.add_argument("--generator-model", default=None)
    run.add_argument("--generator-base-url", default=None)

    run.add_argument("--editor", default=None, help="univideo-gradio or wan-edit-gradio.")
    run.add_argument("--editor-model", default=None)
    run.add_argument("--editor-base-url", default=None)

    run.add_argument("--critic-provider", default=None, help="gemini, gpt, openrouter, qwen3vl.")
    run.add_argument("--critic-model", default=None)
    run.add_argument("--critic-base-url", default=None)
    run.add_argument("--critic-api-key-env", default=None)
    run.add_argument("--critic-num-frames", type=int, default=None)
    run.add_argument("--critic-frame-input", action="store_true")
    run.add_argument("--critic-native-video", action="store_true")

    run.add_argument("--verifier-provider", default=None, help="gemini, gpt, openrouter, qwen3vl.")
    run.add_argument("--verifier-model", default=None)
    run.add_argument("--verifier-base-url", default=None)
    run.add_argument("--verifier-api-key-env", default=None)
    run.add_argument("--verifier-num-frames", type=int, default=None)
    run.add_argument("--verifier-frame-input", action="store_true")
    run.add_argument("--verifier-native-video", action="store_true")

    run.add_argument("--eval-verifier-provider", default=None)
    run.add_argument("--eval-verifier-model", default=None)
    run.add_argument("--eval-verifier-base-url", default=None)
    run.add_argument("--eval-verifier-api-key-env", default=None)
    run.add_argument("--eval-verifier-num-frames", type=int, default=None)
    run.add_argument("--no-eval-verifier", action="store_true")

    run.add_argument("--questions", default=None, help="Question JSON/list string or path.")
    run.add_argument("--questions-file", default=None)
    run.add_argument("--auto-questions", action="store_true")
    run.add_argument("--no-auto-questions", action="store_true")

    run.set_defaults(func=run_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
