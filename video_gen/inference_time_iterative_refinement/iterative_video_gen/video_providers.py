from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import requests


def _poll_gradio_result(
    base_url: str,
    event_id: str,
    poll_interval: int = 10,
    max_wait: int = 900,
) -> Any:
    result_url = f"{base_url}/{event_id}"
    start = time.time()

    while (time.time() - start) < max_wait:
        try:
            resp = requests.get(result_url, stream=True, timeout=max_wait)
            current_event = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    current_event = None
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if current_event == "complete":
                        try:
                            return json.loads(data_str)
                        except json.JSONDecodeError:
                            return data_str
                    if current_event == "error":
                        raise RuntimeError(f"Gradio server error: {data_str}")
        except requests.exceptions.Timeout:
            pass
        except Exception as exc:
            print(f"  [gradio] Polling error: {exc}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Gradio API did not return within {max_wait}s")


def _extract_path_from_gradio_result(result: Any) -> str | None:
    item = None
    if isinstance(result, list) and result:
        item = result[0]
    elif isinstance(result, dict) and "data" in result:
        data = result["data"]
        item = data[0] if isinstance(data, list) and data else None
    elif isinstance(result, str):
        return result

    if isinstance(item, dict):
        return item.get("path") or item.get("url")
    if isinstance(item, str):
        return item
    return None


def _resolve_gradio_output(
    result: Any,
    output_path: str | Path,
    tag: str,
    base_url: str | None = None,
) -> str:
    output_path = Path(output_path).resolve()
    remote_path = _extract_path_from_gradio_result(result)

    if not remote_path:
        if output_path.exists():
            return str(output_path)
        raise RuntimeError(f"[{tag}] Could not parse output path from result: {result}")

    if not remote_path.startswith("http") and os.path.exists(remote_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(remote_path).resolve() != output_path:
            shutil.copy2(remote_path, output_path)
        return str(output_path)

    url = remote_path if remote_path.startswith("http") else None
    if not url and base_url:
        item = result[0] if isinstance(result, list) and result else result
        if isinstance(item, dict) and item.get("url"):
            url = item["url"]

    if url:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(output_path)

    raise RuntimeError(f"[{tag}] Could not locate output video: {remote_path}")


class VideoGenerator:
    def generate(self, prompt: str, output_path: str | Path, seed: int | None = None) -> str:
        raise NotImplementedError


class VideoEditor:
    def edit(
        self,
        prompt: str,
        source_video_path: str | Path,
        output_path: str | Path,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError


class GradioWanT2VGenerator(VideoGenerator):
    """Wan T2V-style Gradio API.

    Expected payload:
    [prompt, resolution, num_frames, num_steps, guidance_scale, flow_shift,
     seed, negative_prompt, output_path]
    """

    def __init__(self, cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        self.cfg = cfg
        self.generation_cfg = generation_cfg
        self.base_url = cfg.get("base_url", "http://localhost:8861").rstrip("/")
        self.poll_interval = int(cfg.get("poll_interval", generation_cfg.get("poll_interval", 10)))
        self.max_wait = int(cfg.get("max_wait", generation_cfg.get("max_wait", 900)))

    def generate(self, prompt: str, output_path: str | Path, seed: int | None = None) -> str:
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        api_url = f"{self.base_url}/gradio_api/call/predict"
        chosen_seed = seed if seed is not None else self.generation_cfg.get("seed", -1)
        payload = {
            "data": [
                prompt,
                self.generation_cfg.get("resolution", "832*480"),
                int(self.generation_cfg.get("num_frames", 37)),
                int(self.generation_cfg.get("num_steps", 50)),
                float(self.generation_cfg.get("guidance_scale", 6.0)),
                float(self.generation_cfg.get("flow_shift", 8.0)),
                int(chosen_seed),
                self.generation_cfg.get("negative_prompt", ""),
                output_path,
            ]
        }
        resp = requests.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        event_id = resp.json().get("event_id")
        if not event_id:
            raise RuntimeError(f"No event_id returned: {resp.text}")
        result = _poll_gradio_result(
            f"{self.base_url}/gradio_api/call/predict",
            event_id,
            self.poll_interval,
            self.max_wait,
        )
        return _resolve_gradio_output(result, output_path, "video_gen", self.base_url)


class GradioUniVideoEditor(VideoEditor):
    """UniVideo/HunyuanVideo editing Gradio API used by the research scripts.

    Expected payload:
    [video_upload, source_video_path_override, edit_prompt, num_steps, num_frames,
     height, width, guidance_scale, image_guidance_scale, seed, timestep_shift,
     negative_prompt, teacache_thresh, output_path]
    """

    def __init__(self, cfg: dict[str, Any], edit_cfg: dict[str, Any]):
        self.cfg = cfg
        self.edit_cfg = edit_cfg
        self.base_url = cfg.get("base_url", "http://localhost:9861").rstrip("/")
        self.poll_interval = int(cfg.get("poll_interval", edit_cfg.get("poll_interval", 10)))
        self.max_wait = int(cfg.get("max_wait", edit_cfg.get("max_wait", 900)))

    def edit(
        self,
        prompt: str,
        source_video_path: str | Path,
        output_path: str | Path,
        seed: int | None = None,
    ) -> str:
        source_video_path = str(Path(source_video_path).resolve())
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        api_url = f"{self.base_url}/gradio_api/call/predict"
        chosen_seed = seed if seed is not None else self.edit_cfg.get("seed", -1)
        guidance_scale = self.edit_cfg.get("guidance_scale", self.edit_cfg.get("cfg_scale", 7.0))
        image_guidance_scale = self.edit_cfg.get(
            "image_guidance_scale", self.edit_cfg.get("edit_strength", 2.0)
        )
        timestep_shift = self.edit_cfg.get("timestep_shift", self.edit_cfg.get("cfg_txt", 7.0))
        teacache_thresh = self.edit_cfg.get("teacache_thresh", self.edit_cfg.get("start_frame", 0))
        payload = {
            "data": [
                self.edit_cfg.get("image_input"),
                source_video_path,
                prompt,
                int(self.edit_cfg.get("num_steps", 50)),
                int(self.edit_cfg.get("num_frames", 37)),
                int(self.edit_cfg.get("height", 480)),
                int(self.edit_cfg.get("width", 832)),
                float(guidance_scale),
                float(image_guidance_scale),
                int(chosen_seed),
                float(timestep_shift),
                self.edit_cfg.get("negative_prompt", ""),
                float(teacache_thresh),
                output_path,
            ]
        }
        resp = requests.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        event_id = resp.json().get("event_id")
        if not event_id:
            raise RuntimeError(f"No event_id returned: {resp.text}")
        result = _poll_gradio_result(
            f"{self.base_url}/gradio_api/call/predict",
            event_id,
            self.poll_interval,
            self.max_wait,
        )
        return _resolve_gradio_output(result, output_path, "video_edit", self.base_url)


def build_video_generator(cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> VideoGenerator:
    provider = str(cfg.get("provider", "wan-t2v-gradio")).lower().replace("_", "-")
    if provider in {
        "wan",
        "wan-t2v",
        "wan-t2v-gradio",
        "wan2.1-t2v-gradio",
        "wan2.2-t2v-gradio",
        "wan22-t2v-gradio",
        "gradio-wan-t2v",
    }:
        return GradioWanT2VGenerator(cfg, generation_cfg)
    raise ValueError(f"Unsupported video generator provider: {provider}")


def build_video_editor(cfg: dict[str, Any], edit_cfg: dict[str, Any]) -> VideoEditor:
    provider = str(cfg.get("provider", "univideo-gradio")).lower().replace("_", "-")
    if provider in {
        "univideo",
        "univideo-gradio",
        "wan-edit",
        "wan-edit-gradio",
        "gradio-univideo",
    }:
        return GradioUniVideoEditor(cfg, edit_cfg)
    raise ValueError(f"Unsupported video editor provider: {provider}")
