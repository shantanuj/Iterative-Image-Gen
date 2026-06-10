from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from PIL import Image


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def image_to_data_url(path: str | Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def save_base64_image(data: str, output_path: str | Path) -> str:
    ensure_parent(output_path)
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    Path(output_path).write_bytes(image_bytes)
    return str(output_path)


def normalize_image_file(input_path: str | Path, output_path: str | Path) -> str:
    ensure_parent(output_path)
    image = Image.open(input_path).convert("RGB")
    image.save(output_path)
    return str(output_path)


def extract_text_from_chat_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("text"):
                    parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def extract_image_from_openai_chat_response(response_json: dict[str, Any], output_path: str | Path) -> str:
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, str):
        match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
        if match:
            return save_base64_image(match.group(0), output_path)
        raise ValueError("Chat response content was text and did not include a data URL image.")

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if url and url.startswith("data:"):
                return save_base64_image(url, output_path)
        if "image" in item and isinstance(item["image"], str):
            return save_base64_image(item["image"], output_path)

    raise ValueError("Could not find generated image in OpenAI-compatible chat response.")


def parse_first_prompt(llm_response: str) -> str:
    output_pattern = r"Output:\s*(.+?)(?:\n|$)"
    match = re.search(output_pattern, llm_response, re.IGNORECASE | re.DOTALL)
    if "no further edits needed" in llm_response.lower():
        return "complete"
    if match:
        extracted_output = match.group(1).strip()
        if "No further edits needed" in extracted_output:
            return "complete"
        return extracted_output
    raise ValueError(f"Could not parse first prompt from response:\n{llm_response}")


def parse_next_step(llm_response: str) -> tuple[str, str]:
    action_pattern = r"Action:\s*(.+?)(?:\n|$)"
    prompt_pattern = r"Prompt:\s*(.+?)(?:\n|$)"
    action_match = re.search(action_pattern, llm_response, re.IGNORECASE | re.DOTALL)
    prompt_match = re.search(prompt_pattern, llm_response, re.IGNORECASE | re.DOTALL)
    if not action_match or not prompt_match:
        raise ValueError(f"Could not parse action/prompt from response:\n{llm_response}")
    return normalize_action(action_match.group(1).strip()), prompt_match.group(1).strip()


def normalize_action(action: str) -> str:
    normalized = action.strip().upper().replace(" ", "_").replace("-", "_")
    if normalized in {"FRESHSTART", "START_AGAIN", "RESTART"}:
        return "FRESH_START"
    if normalized in {"COMPLETE", "DONE"}:
        return "STOP"
    if normalized not in {"CONTINUE", "BACKTRACK", "FRESH_START", "STOP"}:
        return normalized
    return normalized


def parse_json_questions(text: str) -> list[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass
    questions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            questions.append(stripped[1:].strip())
        elif stripped.endswith("?"):
            questions.append(stripped)
    return questions


def extract_question_wise_scores(
    response_text: str,
    questions: list[str] | None = None,
) -> list[tuple[str, float]]:
    question_wise_scores: list[tuple[str, float]] = []
    question_lookup = {q.strip().lower(): q for q in questions or []}

    for line in response_text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        question, score_text = stripped.split(":", 1)
        question = question.strip()
        score_text = score_text.strip().lower()
        score = 1.0 if score_text in {"yes", "true", "y", "1"} else 0.0
        question = question_lookup.get(question.lower(), question)
        question_wise_scores.append((question, score))

    if not question_wise_scores and questions:
        lowered = response_text.lower()
        for question in questions:
            if re.search(r"\byes\b|\btrue\b", lowered):
                score = 1.0
            else:
                score = 0.0
            question_wise_scores.append((question, score))

    if questions and len(question_wise_scores) < len(questions):
        seen = {q.strip().lower() for q, _ in question_wise_scores}
        for question in questions:
            if question.strip().lower() not in seen:
                question_wise_scores.append((question, 0.0))

    if not question_wise_scores:
        raise ValueError(f"Could not parse verifier scores from response:\n{response_text}")

    mean_score = sum(score for _, score in question_wise_scores) / len(question_wise_scores)
    question_wise_scores.append(("Cumulative mean binary score:", mean_score))
    return question_wise_scores

