from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import expand_env_vars, load_config, normalize_questions, set_nested
from .engine import IterativeImageGenRunner, prepare_questions


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

    _set_if_present(cfg, "generation.width", args.width)
    _set_if_present(cfg, "generation.height", args.height)
    if args.width is not None and args.height is not None:
        set_nested(cfg, "generation.size", f"{args.width}x{args.height}")
    _set_if_present(cfg, "generation.size", args.size)
    _set_if_present(cfg, "generation.num_inference_steps", args.num_inference_steps)
    _set_if_present(cfg, "generation.guidance_scale", args.guidance_scale)
    _set_if_present(cfg, "generation.true_cfg_scale", args.true_cfg_scale)
    _set_if_present(cfg, "generation.quality", args.quality)

    _set_if_present(cfg, "models.generator.provider", args.base_generator)
    _set_if_present(cfg, "models.generator.backend", args.generator_backend)
    _set_if_present(cfg, "models.generator.model", args.generator_model)
    _set_if_present(cfg, "models.generator.base_url", args.generator_base_url)
    _set_if_present(cfg, "models.generator.generation_endpoint", args.generator_endpoint)

    _set_if_present(cfg, "models.editor.provider", args.editor)
    _set_if_present(cfg, "models.editor.backend", args.editor_backend)
    _set_if_present(cfg, "models.editor.model", args.editor_model)
    _set_if_present(cfg, "models.editor.base_url", args.editor_base_url)
    _set_if_present(cfg, "models.editor.edit_endpoint", args.editor_endpoint)

    _set_if_present(cfg, "models.critic.provider", args.critic_provider)
    _set_if_present(cfg, "models.critic.model", args.critic_model)
    _set_if_present(cfg, "models.critic.base_url", args.critic_base_url)
    _set_if_present(cfg, "models.critic.api_key_env", args.critic_api_key_env)

    _set_if_present(cfg, "models.verifier.provider", args.verifier_provider)
    _set_if_present(cfg, "models.verifier.model", args.verifier_model)
    _set_if_present(cfg, "models.verifier.base_url", args.verifier_base_url)
    _set_if_present(cfg, "models.verifier.api_key_env", args.verifier_api_key_env)

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
    if args.tiif_questions:
        set_nested(cfg, "questions.tiif_prompt", True)

    if args.rephrase_first_step:
        set_nested(cfg, "method.rephrase_first_step", True)
        set_nested(cfg, "method.do_first_step_as_original_prompt", False)
    if args.original_first_step:
        set_nested(cfg, "method.rephrase_first_step", False)
        set_nested(cfg, "method.do_first_step_as_original_prompt", True)

    return expand_env_vars(cfg)


def run_cmd(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg = _apply_run_overrides(cfg, args)
    runner = IterativeImageGenRunner(cfg)
    questions = prepare_questions(cfg, args.prompt, critic=runner.critic)
    result = runner.run(args.prompt, questions)

    print(f"Output directory: {result.output_dir}")
    if result.best_candidate:
        print(f"Best image: {result.best_candidate.image_path}")
        print(f"Best loop-verifier score: {result.best_candidate.score:.4f}")
    if result.best_eval_candidate:
        print(f"Best eval-verifier image: {result.best_eval_candidate.image_path}")
        print(f"Best eval-verifier score: {result.best_eval_candidate.eval_score:.4f}")
    print(f"Summary JSON: {Path(result.output_dir) / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iterative-image-gen",
        description="Run iterative refinement for compositional image generation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run image generation from a prompt.")
    run.add_argument("--prompt", required=True, help="Full complex image prompt.")
    run.add_argument("--config", default=None, help="YAML config path.")
    run.add_argument(
        "--mode",
        choices=["iterative", "iterative_parallel", "parallel", "all"],
        default=None,
        help="Run mode. parallel/all use iterations*parallel one-step samples.",
    )
    run.add_argument("--iterations", type=int, default=None, help="Image calls per trajectory.")
    run.add_argument("--parallel", type=int, default=None, help="Number of iterative streams.")
    run.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent streams/samples. Defaults to --parallel.",
    )
    run.add_argument("--output-dir", default=None, help="Directory for images and logs.")
    run.add_argument("--seed", type=int, default=None)

    run.add_argument("--width", type=int, default=None)
    run.add_argument("--height", type=int, default=None)
    run.add_argument("--size", default=None, help="OpenAI-style size string, e.g. 1024x1024.")
    run.add_argument("--num-inference-steps", type=int, default=None)
    run.add_argument("--guidance-scale", type=float, default=None)
    run.add_argument("--true-cfg-scale", type=float, default=None)
    run.add_argument("--quality", default=None)

    run.add_argument("--base-generator", default=None, help="qwen-image, nanobanana, gpt-image, flux-dev.")
    run.add_argument("--generator-backend", default=None, help="vllm_omni, legacy_http, diffusers.")
    run.add_argument("--generator-model", default=None)
    run.add_argument("--generator-base-url", default=None)
    run.add_argument("--generator-endpoint", choices=["chat", "images"], default=None)

    run.add_argument("--editor", default=None, help="qwen-image-edit, nanobanana, gpt-image, flux-kontext.")
    run.add_argument("--editor-backend", default=None, help="vllm_omni, legacy_http, diffusers.")
    run.add_argument("--editor-model", default=None)
    run.add_argument("--editor-base-url", default=None)
    run.add_argument("--editor-endpoint", choices=["chat"], default=None)

    run.add_argument("--critic-provider", default=None, help="gemini, gpt, openrouter, qwen3vl.")
    run.add_argument("--critic-model", default=None)
    run.add_argument("--critic-base-url", default=None)
    run.add_argument("--critic-api-key-env", default=None)

    run.add_argument("--verifier-provider", default=None, help="gemini, gpt, openrouter, qwen3vl.")
    run.add_argument("--verifier-model", default=None)
    run.add_argument("--verifier-base-url", default=None)
    run.add_argument("--verifier-api-key-env", default=None)

    run.add_argument("--eval-verifier-provider", default=None)
    run.add_argument("--eval-verifier-model", default=None)
    run.add_argument("--eval-verifier-base-url", default=None)
    run.add_argument("--eval-verifier-api-key-env", default=None)
    run.add_argument("--no-eval-verifier", action="store_true")

    run.add_argument("--questions", default=None, help="Question JSON/list string or path.")
    run.add_argument("--questions-file", default=None)
    run.add_argument("--auto-questions", action="store_true")
    run.add_argument("--no-auto-questions", action="store_true")
    run.add_argument("--tiif-questions", action="store_true")
    run.add_argument(
        "--rephrase-first-step",
        action="store_true",
        help="Ask the critic to rewrite the initial prompt before step 0 generation.",
    )
    run.add_argument(
        "--original-first-step",
        action="store_true",
        help="Use the full input prompt exactly for step 0 generation.",
    )
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
