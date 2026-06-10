from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

import requests


def image_to_data_url(path: str | Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def save_base64_image(data: str, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    output_path.write_bytes(base64.b64decode(data))
    return str(output_path)


def extract_image_from_chat_response(response_json: dict[str, Any], output_path: str | Path) -> str:
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, str):
        match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
        if match:
            return save_base64_image(match.group(0), output_path)
        raise ValueError("Qwen edit response was text and did not contain an image data URL.")

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if url and url.startswith("data:"):
                return save_base64_image(url, output_path)
        if isinstance(item.get("image"), str):
            return save_base64_image(item["image"], output_path)

    raise ValueError("Could not find an edited image in the Qwen edit response.")


class QwenImageEditClient:
    """OpenAI-compatible vLLM-Omni client for Qwen-Image-Edit."""

    def __init__(
        self,
        base_url: str = "http://localhost:8092/v1",
        model: str = "Qwen/Qwen-Image-Edit",
        api_key: str = "none",
        timeout: float = 900.0,
        num_inference_steps: int | None = 30,
        guidance_scale: float | None = None,
        true_cfg_scale: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.true_cfg_scale = true_cfg_scale

    def edit(self, image_path: str | Path, instruction: str, output_path: str | Path, seed: int | None = None) -> str:
        extra_body = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "seed": seed,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
            "extra_body": {k: v for k, v in extra_body.items() if v is not None},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return extract_image_from_chat_response(response.json(), output_path)
