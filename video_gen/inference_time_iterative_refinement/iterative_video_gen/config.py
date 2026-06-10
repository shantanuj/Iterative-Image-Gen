from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "mode": "step_by_step",
        "iterations": 3,
        "parallel": 1,
        "max_workers": None,
        "output_dir": "outputs/run",
        "seed": None,
    },
    "method": {
        "planner_provider": "critic",
        "plan_prompt_style": "embellished",
        "max_add_steps": 2,
        "max_refine_per_step": 2,
    },
    "questions": {
        "auto": True,
        "path": None,
        "items": [],
    },
    "generation": {
        "resolution": "832*480",
        "num_frames": 37,
        "num_steps": 50,
        "guidance_scale": 6.0,
        "flow_shift": 8.0,
        "seed": -1,
        "negative_prompt": "",
        "poll_interval": 10,
        "max_wait": 900,
    },
    "editing": {
        "num_frames": 37,
        "num_steps": 50,
        "height": 480,
        "width": 832,
        "guidance_scale": 7.0,
        "image_guidance_scale": 2.0,
        "timestep_shift": 7.0,
        "teacache_thresh": 0,
        "seed": -1,
        "negative_prompt": "",
        "image_input": None,
        "poll_interval": 10,
        "max_wait": 900,
    },
    "models": {
        "generator": {
            "provider": "wan-t2v-gradio",
            "model": "Wan2.1-T2V",
            "base_url": "${WAN_T2V_BASE_URL:-http://localhost:8861}",
        },
        "editor": {
            "provider": "univideo-gradio",
            "model": "UniVideo",
            "base_url": "${UNIVIDEO_EDIT_BASE_URL:-http://localhost:9861}",
        },
        "critic": {
            "provider": "gpt",
            "model": "gpt-5.2",
            "api_key_env": "OPENAI_API_KEY",
            "num_frames": 6,
        },
        "verifier": {
            "provider": "gpt",
            "model": "gpt-5.2",
            "api_key_env": "OPENAI_API_KEY",
            "num_frames": 6,
        },
        "eval_verifier": None,
    },
}


def _expand_shell_default(value: str) -> str:
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
        return {key: expand_env_vars(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
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
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, loaded)
        loaded_editing = loaded.get("editing", {}) or {}
        alias_map = {
            "cfg_scale": "guidance_scale",
            "edit_strength": "image_guidance_scale",
            "cfg_txt": "timestep_shift",
            "start_frame": "teacache_thresh",
        }
        for legacy_key, canonical_key in alias_map.items():
            if legacy_key in loaded_editing and canonical_key not in loaded_editing:
                cfg["editing"][canonical_key] = loaded_editing[legacy_key]
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
        return normalize_questions(json.loads(text))
    if path.suffix.lower() == ".jsonl":
        questions: list[str] = []
        for line in text.splitlines():
            if line.strip():
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
