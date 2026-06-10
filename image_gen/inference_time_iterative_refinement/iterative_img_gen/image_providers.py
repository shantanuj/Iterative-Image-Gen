from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .io_utils import (
    ensure_parent,
    extract_image_from_openai_chat_response,
    image_to_data_url,
    normalize_image_file,
    save_base64_image,
)


class ImageProvider:
    def generate(self, prompt: str, output_path: str, seed: int | None = None) -> str:
        raise NotImplementedError

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError


def _api_key_from_cfg(cfg: dict[str, Any], default_env: str | None = None) -> str | None:
    if cfg.get("api_key"):
        return str(cfg["api_key"])
    env_name = cfg.get("api_key_env") or default_env
    if env_name:
        return os.environ.get(str(env_name))
    return None


def _endpoint_url(base_url: str, endpoint: str) -> str:
    base_url = base_url.rstrip("/")
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if base_url.endswith("/v1") and endpoint.startswith("/v1/"):
        endpoint = endpoint[3:]
    return base_url + endpoint


class VLLMOmniImageProvider(ImageProvider):
    """OpenAI-compatible vLLM-Omni image generation/editing client."""

    def __init__(self, cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.model = cfg.get("model")
        self.base_url = str(cfg.get("base_url", "http://localhost:8000/v1"))
        self.generation_endpoint = cfg.get("generation_endpoint", "chat")
        self.edit_endpoint = cfg.get("edit_endpoint", "chat")
        self.timeout = float(cfg.get("timeout", 600))
        self.api_key = _api_key_from_cfg(cfg) or "none"

    def _extra_body(self, seed: int | None) -> dict[str, Any]:
        width = self.generation_cfg.get("width")
        height = self.generation_cfg.get("height")
        body = {
            "height": height,
            "width": width,
            "size": self.generation_cfg.get("size") or f"{width}x{height}",
            "num_inference_steps": self.generation_cfg.get("num_inference_steps"),
            "guidance_scale": self.generation_cfg.get("guidance_scale"),
            "true_cfg_scale": self.generation_cfg.get("true_cfg_scale"),
            "negative_prompt": self.generation_cfg.get("negative_prompt"),
            "seed": seed,
        }
        body.update(self.cfg.get("extra_body", {}) or {})
        return {k: v for k, v in body.items() if v is not None}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_chat(self, messages: list[dict[str, Any]], output_path: str, seed: int | None) -> str:
        ensure_parent(output_path)
        payload = {
            "model": self.model,
            "messages": messages,
            "extra_body": self._extra_body(seed),
        }
        response = requests.post(
            _endpoint_url(self.base_url, "/v1/chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return extract_image_from_openai_chat_response(response.json(), output_path)

    def _post_images(self, prompt: str, output_path: str, seed: int | None) -> str:
        ensure_parent(output_path)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": self.generation_cfg.get("size"),
            "response_format": "b64_json",
            **self._extra_body(seed),
        }
        response = requests.post(
            _endpoint_url(self.base_url, "/v1/images/generations"),
            headers=self._headers(),
            json={k: v for k, v in payload.items() if v is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"][0]["b64_json"]
        return save_base64_image(data, output_path)

    def generate(self, prompt: str, output_path: str, seed: int | None = None) -> str:
        if self.generation_endpoint == "images":
            return self._post_images(prompt, output_path, seed)
        messages = [{"role": "user", "content": prompt}]
        return self._post_chat(messages, output_path, seed)

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        seed: int | None = None,
    ) -> str:
        if self.edit_endpoint != "chat":
            raise ValueError("vLLM-Omni editing currently expects edit_endpoint=chat.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ]
        return self._post_chat(messages, output_path, seed)


class LegacyHTTPImageProvider(ImageProvider):
    """Compatibility client for the original /generate Flask-style servers."""

    def __init__(self, cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.generate_url = cfg.get("generate_url") or cfg.get("base_url")
        self.edit_url = cfg.get("edit_url") or cfg.get("base_url")
        if not self.generate_url:
            raise ValueError("Legacy HTTP provider requires base_url or generate_url.")
        self.timeout = float(cfg.get("timeout", 600))

    def _post(self, url: str, payload: dict[str, Any], output_path: str) -> str:
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("b64_json"):
            return save_base64_image(data["b64_json"], output_path)
        if data.get("image"):
            return save_base64_image(data["image"], output_path)
        if data.get("success") is False:
            raise RuntimeError(data.get("error", "Image server returned success=false"))
        returned_path = data.get("output") or data.get("path") or output_path
        if Path(returned_path).exists() and Path(returned_path) != Path(output_path):
            return normalize_image_file(returned_path, output_path)
        return str(returned_path)

    def _base_config(self, output_path: str, seed: int | None) -> dict[str, Any]:
        width = self.generation_cfg.get("width")
        height = self.generation_cfg.get("height")
        return {
            "width": width,
            "height": height,
            "seed": seed if seed is not None else random.randint(0, 1000000),
            "image_save_path": str(Path(output_path).absolute()),
            "num_inference_steps": self.generation_cfg.get("num_inference_steps"),
            "num_edit_inference_steps": self.generation_cfg.get("num_inference_steps"),
            "guidance_scale": self.generation_cfg.get("guidance_scale"),
            "true_cfg_scale": self.generation_cfg.get("true_cfg_scale"),
            "edit_guidance_scale": self.generation_cfg.get("guidance_scale"),
            "negative_prompt": self.generation_cfg.get("negative_prompt", ""),
        }

    def generate(self, prompt: str, output_path: str, seed: int | None = None) -> str:
        config = self._base_config(output_path, seed)
        config["prompt"] = prompt
        config.update(self.cfg.get("extra_body", {}) or {})
        return self._post(str(self.generate_url), {"config": config}, output_path)

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        seed: int | None = None,
    ) -> str:
        config = self._base_config(output_path, seed)
        config["edit_prompt"] = prompt
        config["prev_image_path"] = str(Path(image_path).absolute())
        config.update(self.cfg.get("extra_body", {}) or {})
        return self._post(str(self.edit_url), {"config": config}, output_path)


class OpenAIImageProvider(ImageProvider):
    def __init__(self, cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        from openai import OpenAI

        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.model = cfg.get("model", "gpt-image-1")
        self.client = OpenAI(
            api_key=_api_key_from_cfg(cfg, "OPENAI_API_KEY"),
            base_url=cfg.get("base_url"),
        )

    def _save_response(self, response: Any, output_path: str) -> str:
        data = response.data[0]
        if getattr(data, "b64_json", None):
            return save_base64_image(data.b64_json, output_path)
        if getattr(data, "url", None):
            request = requests.get(data.url, timeout=300)
            request.raise_for_status()
            ensure_parent(output_path)
            Path(output_path).write_bytes(request.content)
            return output_path
        raise ValueError("OpenAI image response did not include b64_json or url.")

    def generate(self, prompt: str, output_path: str, seed: int | None = None) -> str:
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "size": self.generation_cfg.get("size", "1024x1024"),
            "quality": self.generation_cfg.get("quality"),
        }
        kwargs.update(self.cfg.get("extra_body", {}) or {})
        response = self.client.images.generate(**{k: v for k, v in kwargs.items() if v is not None})
        return self._save_response(response, output_path)

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        seed: int | None = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "size": self.generation_cfg.get("size", "1024x1024"),
            "quality": self.generation_cfg.get("quality"),
        }
        kwargs.update(self.cfg.get("extra_body", {}) or {})
        with open(image_path, "rb") as image_file:
            response = self.client.images.edit(
                image=[image_file],
                **{k: v for k, v in kwargs.items() if v is not None},
            )
        return self._save_response(response, output_path)


class GeminiImageProvider(ImageProvider):
    def __init__(self, cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        from google import genai

        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.model = cfg.get("model", "gemini-2.5-flash-image-preview")
        self.client = genai.Client(api_key=_api_key_from_cfg(cfg, "GEMINI_API_KEY"))

    def _save_inline_image(self, response: Any, output_path: str) -> str:
        ensure_parent(output_path)
        for part in response.candidates[0].content.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None:
                Path(output_path).write_bytes(inline_data.data)
                return output_path
        raise ValueError("Gemini image response did not include inline image data.")

    def generate(self, prompt: str, output_path: str, seed: int | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
        )
        return self._save_inline_image(response, output_path)

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        seed: int | None = None,
    ) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image],
        )
        return self._save_inline_image(response, output_path)


class DiffusersImageProvider(ImageProvider):
    """Optional single-process fallback for local experiments."""

    def __init__(self, cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        import torch
        from diffusers import DiffusionPipeline

        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.torch = torch
        self.model = cfg.get("model")
        self.device_map = cfg.get("device_map", "balanced")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        )

    def _generator(self, seed: int | None) -> Any:
        if seed is None:
            return None
        return self.torch.Generator(device="cuda").manual_seed(seed)

    def generate(self, prompt: str, output_path: str, seed: int | None = None) -> str:
        ensure_parent(output_path)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.generation_cfg.get("negative_prompt", ""),
            width=self.generation_cfg.get("width"),
            height=self.generation_cfg.get("height"),
            num_inference_steps=self.generation_cfg.get("num_inference_steps"),
            guidance_scale=self.generation_cfg.get("guidance_scale"),
            true_cfg_scale=self.generation_cfg.get("true_cfg_scale"),
            generator=self._generator(seed),
        ).images[0]
        image.save(output_path)
        return output_path

    def edit(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        seed: int | None = None,
    ) -> str:
        ensure_parent(output_path)
        image = Image.open(image_path).convert("RGB")
        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=self.generation_cfg.get("negative_prompt", ""),
            width=self.generation_cfg.get("width"),
            height=self.generation_cfg.get("height"),
            num_inference_steps=self.generation_cfg.get("num_inference_steps"),
            guidance_scale=self.generation_cfg.get("guidance_scale"),
            true_cfg_scale=self.generation_cfg.get("true_cfg_scale"),
            generator=self._generator(seed),
        ).images[0]
        result.save(output_path)
        return output_path


def build_image_provider(cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> ImageProvider:
    provider = str(cfg.get("provider", "")).lower().replace("_", "-")
    backend = str(cfg.get("backend", "")).lower().replace("_", "-")

    if provider in {"gpt-image", "gpt", "openai-image"}:
        return OpenAIImageProvider(cfg, generation_cfg)
    if provider in {"nanobanana", "nanobana", "gemini-image", "gemini"}:
        return GeminiImageProvider(cfg, generation_cfg)

    if backend in {"vllm-omni", "vllm"}:
        return VLLMOmniImageProvider(cfg, generation_cfg)
    if backend in {"legacy-http", "http-legacy"}:
        return LegacyHTTPImageProvider(cfg, generation_cfg)
    if backend == "diffusers":
        return DiffusersImageProvider(cfg, generation_cfg)

    if provider in {
        "qwen-image",
        "qwen-image-edit",
        "flux-dev",
        "flux-kontext",
        "flux-context",
    }:
        return VLLMOmniImageProvider(cfg, generation_cfg)

    raise ValueError(f"Unsupported image provider/backend: provider={provider}, backend={backend}")
