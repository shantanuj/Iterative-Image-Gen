from __future__ import annotations

import os
from typing import Any

from .io_utils import (
    extract_question_wise_scores,
    extract_text_from_chat_content,
    image_to_data_url,
    parse_json_questions,
)
from .prompts import (
    question_generation_system_prompt,
    question_generation_system_prompt_tiif,
    rating_system_prompt,
)


class VLMClient:
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError

    def verify(self, image_path: str, questions: list[str]) -> list[tuple[str, float]]:
        user_prompt = "\n".join(questions)
        response = self.complete(rating_system_prompt, user_prompt, image_path=image_path)
        return extract_question_wise_scores(response, questions)

    def generate_questions(self, prompt: str, tiif_prompt: bool = False) -> list[str]:
        system_prompt = (
            question_generation_system_prompt_tiif
            if tiif_prompt
            else question_generation_system_prompt
        )
        response = self.complete(system_prompt, prompt)
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


class GeminiVLMClient(VLMClient):
    def __init__(self, cfg: dict[str, Any]):
        from google import genai
        from google.genai import types

        self.cfg = cfg
        self.types = types
        self.model = cfg.get("model", "gemini-2.5-flash")
        self.client = genai.Client(api_key=_api_key_from_cfg(cfg, "GEMINI_API_KEY"))

    def _config(self, system_prompt: str, seed: int | None) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": [self.types.Part.from_text(text=system_prompt)],
        }
        thinking_budget = self.cfg.get("thinking_budget", -1)
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
        image_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        parts = []
        if image_path is not None:
            parts.append(
                self.types.Part.from_bytes(
                    mime_type="image/png",
                    data=open(image_path, "rb").read(),
                )
            )
        parts.append(self.types.Part.from_text(text=user_prompt))
        contents = [self.types.Content(role="user", parts=parts)]
        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=self._config(system_prompt, seed),
        ):
            response_text += chunk.text or ""
        return response_text


class OpenAICompatibleVLMClient(VLMClient):
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
        headers = cfg.get("default_headers") or {}
        self.client = OpenAI(
            api_key=_api_key_from_cfg(cfg, default_api_key_env),
            base_url=cfg.get("base_url") or default_base_url,
            default_headers=headers or None,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        if image_path is None:
            user_content: Any = user_prompt
        else:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ]
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


class TransformersQwen3VLVLMClient(VLMClient):
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from transformers import pipeline

        self.cfg = cfg
        self.model = cfg.get("model", "Qwen/Qwen3-VL-30B-A3B-Instruct")
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
        image_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        content = []
        if image_path is not None:
            content.append({"type": "image", "url": image_path})
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        generated = self.pipe(
            text=messages,
            max_new_tokens=self.cfg.get("max_new_tokens", 2048),
        )
        output = generated[-1]["generated_text"]
        if isinstance(output, list):
            return output[-1].get("content", "")
        return str(output)


def build_vlm_client(cfg: dict[str, Any]) -> VLMClient:
    provider = str(cfg.get("provider", "")).lower().replace("_", "-")

    if provider == "gemini":
        return GeminiVLMClient(cfg)
    if provider in {"gpt", "openai", "openai-compatible"}:
        return OpenAICompatibleVLMClient(cfg, default_api_key_env="OPENAI_API_KEY")
    if provider == "openrouter":
        if not cfg.get("api_key_env") and not cfg.get("api_key"):
            cfg = {**cfg, "api_key_env": "OPENROUTER_API_KEY"}
        return OpenAICompatibleVLMClient(
            cfg,
            default_api_key_env="OPENROUTER_API_KEY",
            default_base_url="https://openrouter.ai/api/v1",
        )
    if provider in {"qwen3vl", "qwen3-vl"}:
        backend = str(cfg.get("backend", "openai_compatible")).lower().replace("_", "-")
        if backend == "transformers":
            return TransformersQwen3VLVLMClient(cfg)
        return OpenAICompatibleVLMClient(
            cfg,
            default_api_key_env=cfg.get("api_key_env") or "QWEN3VL_API_KEY",
            default_base_url=cfg.get("base_url"),
        )

    raise ValueError(f"Unsupported VLM provider: {provider}")

