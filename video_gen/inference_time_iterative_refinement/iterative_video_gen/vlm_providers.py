from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from .io_utils import (
    extract_question_wise_scores,
    extract_text_from_chat_content,
    file_to_data_url,
    parse_json_questions,
)
from .media_utils import frame_b64_to_data_url, sample_video_frames_base64
from .prompts import (
    QUESTION_GENERATION_SYSTEM_PROMPT,
    VIDEO_RATING_NATIVE_SYSTEM_PROMPT,
    VIDEO_RATING_SYSTEM_PROMPT,
)


class VLMClient:
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError

    def verify(self, video_path: str, questions: list[str]) -> tuple[str, list[tuple[str, float]]]:
        questions_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        user_prompt = f"Answer each question about this video:\n{questions_block}"
        raw = self.complete(VIDEO_RATING_SYSTEM_PROMPT, user_prompt, video_path=video_path)
        return raw, extract_question_wise_scores(raw, questions)

    def generate_questions(self, prompt: str) -> list[str]:
        response = self.complete(QUESTION_GENERATION_SYSTEM_PROMPT, prompt)
        questions = parse_json_questions(response)
        if not questions:
            raise ValueError(f"Could not parse questions from response:\n{response}")
        return questions


def _api_key_from_cfg(cfg: dict[str, Any], default_env: str | None = None) -> str | None:
    if cfg.get("api_key"):
        return str(cfg["api_key"])
    env_name = cfg.get("api_key_env") or default_env
    if env_name:
        return os.environ.get(str(env_name))
    return None


class GeminiVideoVLMClient(VLMClient):
    def __init__(self, cfg: dict[str, Any]):
        from google import genai
        from google.genai import types

        self.cfg = cfg
        self.types = types
        self.model = cfg.get("model", "gemini-2.5-flash")
        self.use_native_video = bool(cfg.get("use_native_video", True))
        self.num_frames = int(cfg.get("num_frames", 6))
        self.client = genai.Client(api_key=_api_key_from_cfg(cfg, "GEMINI_API_KEY"))

    def _config(self, system_prompt: str, seed: int | None) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": [self.types.Part.from_text(text=system_prompt)],
        }
        thinking_budget = self.cfg.get("thinking_budget")
        if thinking_budget is not None:
            kwargs["thinking_config"] = self.types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        if seed is not None:
            kwargs["seed"] = seed
        try:
            return self.types.GenerateContentConfig(**kwargs)
        except TypeError:
            kwargs.pop("thinking_config", None)
            return self.types.GenerateContentConfig(**kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        parts = []
        actual_system_prompt = system_prompt
        if video_path is not None and self.use_native_video:
            parts.append(
                self.types.Part(
                    inline_data=self.types.Blob(
                        data=Path(video_path).read_bytes(),
                        mime_type="video/mp4",
                    )
                )
            )
            if system_prompt == VIDEO_RATING_SYSTEM_PROMPT:
                actual_system_prompt = VIDEO_RATING_NATIVE_SYSTEM_PROMPT
        elif video_path is not None:
            frame_b64 = sample_video_frames_base64(video_path, self.num_frames)
            if not frame_b64:
                raise RuntimeError(f"Could not extract frames from {video_path}")
            for frame in frame_b64:
                parts.append(
                    self.types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=base64.b64decode(frame),
                    )
                )
        parts.append(self.types.Part.from_text(text=user_prompt))
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.types.Content(role="user", parts=parts),
            config=self._config(actual_system_prompt, seed),
        )
        return response.text or ""


class OpenAICompatibleVideoVLMClient(VLMClient):
    def __init__(
        self,
        cfg: dict[str, Any],
        default_api_key_env: str | None = None,
        default_base_url: str | None = None,
    ):
        from openai import OpenAI

        self.cfg = cfg
        self.model = cfg.get("model")
        if not self.model:
            raise ValueError("OpenAI-compatible VLM config requires model.")
        self.num_frames = int(cfg.get("num_frames", 6))
        self.use_video_data_url = bool(cfg.get("use_video_data_url", False))
        self.client = OpenAI(
            api_key=_api_key_from_cfg(cfg, default_api_key_env),
            base_url=cfg.get("base_url") or default_base_url,
            default_headers=cfg.get("default_headers") or None,
        )

    def _video_content(self, video_path: str) -> list[dict[str, Any]]:
        if self.use_video_data_url:
            return [
                {
                    "type": "video_url",
                    "video_url": {"url": file_to_data_url(video_path, "video/mp4")},
                }
            ]
        frame_b64 = sample_video_frames_base64(video_path, self.num_frames)
        if not frame_b64:
            raise RuntimeError(f"Could not extract frames from {video_path}")
        return [
            {"type": "image_url", "image_url": {"url": frame_b64_to_data_url(frame)}}
            for frame in frame_b64
        ]

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        if video_path is None:
            user_content: Any = user_prompt
        else:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(self._video_content(video_path))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.cfg.get("temperature", 0),
        }
        if seed is not None:
            kwargs["seed"] = seed
        kwargs.update(self.cfg.get("extra_body", {}) or {})
        response = self.client.chat.completions.create(**kwargs)
        return extract_text_from_chat_content(response.choices[0].message.content)


class TransformersQwen3VLVideoVLMClient(VLMClient):
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from transformers import pipeline

        self.cfg = cfg
        self.model = cfg.get("model", "Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.num_frames = int(cfg.get("num_frames", 6))
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model,
            torch_dtype=torch.bfloat16,
            device_map=cfg.get("device_map", "auto"),
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        content = []
        if video_path is not None:
            frame_b64 = sample_video_frames_base64(video_path, self.num_frames)
            if not frame_b64:
                raise RuntimeError(f"Could not extract frames from {video_path}")
            for frame in frame_b64:
                content.append({"type": "image", "url": frame_b64_to_data_url(frame)})
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        generated = self.pipe(text=messages, max_new_tokens=self.cfg.get("max_new_tokens", 2048))
        output = generated[-1]["generated_text"]
        if isinstance(output, list):
            return output[-1].get("content", "")
        return str(output)


class MockVideoVLMClient(VLMClient):
    """Deterministic VLM for offline smoke tests."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        lowered = system_prompt.lower()
        if "video generation planner" in lowered:
            return json_escape_plan(user_prompt)
        if "assign each question" in lowered:
            count = sum(
                1 for line in user_prompt.splitlines() if line.strip()[:1].isdigit()
            )
            indices = ", ".join(str(i) for i in range(1, count + 1))
            return f'{{"core": [{indices}]}}'
        if "action:" in system_prompt and "looks_good" in lowered:
            return "Action: LOOKS_GOOD\nPrompt: none"
        if "action:" in system_prompt:
            return "Action: STOP\nPrompt: none"
        if "json list" in lowered or "generates simple yes/no questions" in lowered:
            return '["Does the video satisfy the prompt?"]'

        questions = []
        for line in user_prompt.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit() and "." in stripped:
                questions.append(stripped.split(".", 1)[1].strip())
            elif stripped.endswith("?"):
                questions.append(stripped)
        if not questions:
            questions = ["Does the video satisfy the prompt?"]
        return "\n".join(f"{question}: yes" for question in questions)


def json_escape_plan(prompt: str) -> str:
    import json

    return json.dumps({"core_prompt": prompt, "add_steps": []})


def build_vlm_client(cfg: dict[str, Any]) -> VLMClient:
    provider = str(cfg.get("provider", "")).lower().replace("_", "-")

    if provider == "mock":
        return MockVideoVLMClient(cfg)
    if provider == "gemini":
        return GeminiVideoVLMClient(cfg)
    if provider in {"gpt", "openai", "openai-compatible"}:
        return OpenAICompatibleVideoVLMClient(cfg, default_api_key_env="OPENAI_API_KEY")
    if provider == "openrouter":
        if not cfg.get("api_key_env") and not cfg.get("api_key"):
            cfg = {**cfg, "api_key_env": "OPENROUTER_API_KEY"}
        return OpenAICompatibleVideoVLMClient(
            cfg,
            default_api_key_env="OPENROUTER_API_KEY",
            default_base_url="https://openrouter.ai/api/v1",
        )
    if provider in {"qwen3vl", "qwen3-vl"}:
        backend = str(cfg.get("backend", "openai_compatible")).lower().replace("_", "-")
        if backend == "transformers":
            return TransformersQwen3VLVideoVLMClient(cfg)
        return OpenAICompatibleVideoVLMClient(
            cfg,
            default_api_key_env=cfg.get("api_key_env") or "QWEN3VL_API_KEY",
            default_base_url=cfg.get("base_url"),
        )

    raise ValueError(f"Unsupported VLM provider: {provider}")
