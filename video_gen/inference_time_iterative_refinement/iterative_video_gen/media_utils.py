from __future__ import annotations

import base64
import io
from pathlib import Path


def sample_video_frames_base64(video_path: str | Path, num_frames: int = 6) -> list[str]:
    """Extract uniformly spaced video frames and return base64 JPEG strings."""
    frames = []
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = [0] if num_frames <= 1 else [
            int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
        ]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame[:, :, ::-1])
        cap.release()
    except ImportError:
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(str(video_path), "ffmpeg")
            total = reader.count_frames()
            indices = [0] if num_frames <= 1 else [
                int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
            ]
            for idx in indices:
                frames.append(reader.get_data(idx))
            reader.close()
        except Exception:
            return []

    if not frames:
        return []

    try:
        from PIL import Image
    except ImportError:
        return []

    encoded = []
    for frame in frames:
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return encoded


def frame_b64_to_data_url(frame_b64: str) -> str:
    return f"data:image/jpeg;base64,{frame_b64}"
