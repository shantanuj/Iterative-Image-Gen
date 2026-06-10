from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "mode": "iterative_parallel",
        "iterations": 4,
        "parallel": 1,
        "max_workers": None,
        "output_dir": "outputs/run",
        "seed": None,
    },
    "method": {
        "first_step_prompt_style": "step_by_step",
        "next_step_prompt_style": "step_by_step",
        "rephrase_first_step": False,
        "do_first_step_as_original_prompt": True,
        "specify_num_steps_in_prompt": True,
        "provide_image_in_subsequent_steps": True,
        "provide_entire_edit_history": True,
        "use_decision_making": False,
    },
    "questions": {
        "auto": True,
        "tiif_prompt": False,
        "path": None,
        "items": [],
    },
    "generation": {
        "width": 1024,
        "height": 1024,
        "size": "1024x1024",
        "num_inference_steps": 50,
        "guidance_scale": None,
        "true_cfg_scale": None,
        "negative_prompt": "",
        "quality": "low",
    },
    "models": {
        "generator": {
            "provider": "qwen-image",
            "backend": "vllm_omni",
            "model": "Qwen/Qwen-Image",
            "base_url": "${QWEN_IMAGE_BASE_URL:-http://localhost:8091/v1}",
            "generation_endpoint": "chat",
        },
        "editor": {
            "provider": "qwen-image-edit",
            "backend": "vllm_omni",
            "model": "Qwen/Qwen-Image-Edit",
            "base_url": "${QWEN_IMAGE_EDIT_BASE_URL:-http://localhost:8092/v1}",
            "edit_endpoint": "chat",
        },
        "critic": {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "api_key_env": "GEMINI_API_KEY",
        },
        "verifier": {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "api_key_env": "GEMINI_API_KEY",
        },
        "eval_verifier": None,
    },
}


def _expand_shell_default(value: str) -> str:
    """Expand ${VAR:-default} and ordinary environment variables."""
    if "${" not in value:
        return os.path.expandvars(value)

    out = value
    start = out.find("${")
    while start != -1:
        end = out.find("}", start)
        if end == -1:
            break
        expr = out[start + 2 : end]
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            replacement = os.environ.get(name, default)
        else:
            replacement = os.environ.get(expr, "")
        out = out[:start] + replacement + out[end + 1 :]
        start = out.find("${", start + len(replacement))
    return os.path.expandvars(out)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return _expand_shell_default(value)
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    loaded: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, loaded)
    method_cfg = cfg.setdefault("method", {})
    loaded_method = loaded.get("method", {}) if isinstance(loaded.get("method"), dict) else {}
    if "do_first_step_as_original_prompt" in loaded_method and "rephrase_first_step" not in loaded_method:
        method_cfg["rephrase_first_step"] = not bool(
            method_cfg.get("do_first_step_as_original_prompt", True)
        )
    else:
        method_cfg["do_first_step_as_original_prompt"] = not bool(
            method_cfg["rephrase_first_step"]
        )
    return expand_env_vars(cfg)


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_questions_from_path(path: str | Path) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return normalize_questions(data)
    if path.suffix.lower() == ".jsonl":
        questions: list[str] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            questions.extend(normalize_questions(json.loads(line)))
        return questions
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_questions(data: Any) -> list[str]:
    if data is None:
        return []
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith("["):
            return normalize_questions(json.loads(stripped))
        return [stripped]
    if isinstance(data, dict):
        if "questions" in data:
            return normalize_questions(data["questions"])
        if "yn_question_list" in data:
            return normalize_questions(data["yn_question_list"])
        if "question" in data:
            return [str(data["question"])]
        raise ValueError(f"Could not find questions in dictionary keys: {sorted(data)}")
    if isinstance(data, list):
        out: list[str] = []
        for item in data:
            if isinstance(item, dict):
                if "question" in item:
                    out.append(str(item["question"]))
                elif "text" in item:
                    out.append(str(item["text"]))
                else:
                    raise ValueError(f"Could not find question text in item: {item}")
            else:
                out.append(str(item))
        return out
    raise TypeError(f"Unsupported questions type: {type(data)}")
