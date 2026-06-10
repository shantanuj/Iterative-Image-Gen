from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def file_to_data_url(path: str | Path, mime_type: str | None = None) -> str:
    resolved_mime = mime_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{resolved_mime};base64,{encoded}"


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


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def parse_json_questions(text: str) -> list[str]:
    try:
        data = json.loads(strip_json_fence(text))
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


def parse_action_prompt(text: str) -> tuple[str, str]:
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    prompt_match = re.search(r"Prompt:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not action_match or not prompt_match:
        raise ValueError(f"Could not parse action/prompt from response:\n{text}")
    return normalize_action(action_match.group(1)), prompt_match.group(1).strip()


def normalize_action(action: str) -> str:
    normalized = action.strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "FRESHSTART": "FRESH_START",
        "START_AGAIN": "FRESH_START",
        "RESTART": "FRESH_START",
        "COMPLETE": "STOP",
        "DONE": "STOP",
    }
    return aliases.get(normalized, normalized)


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
        question, score_text = stripped.rsplit(":", 1)
        question = question.strip()
        score_text = score_text.strip().lower()
        score = 1.0 if score_text in {"yes", "true", "y", "1"} else 0.0
        question = question_lookup.get(question.lower(), question)
        question_wise_scores.append((question, score))

    if questions and len(question_wise_scores) < len(questions):
        seen = {q.strip().lower() for q, _ in question_wise_scores}
        for question in questions:
            if question.strip().lower() not in seen:
                question_wise_scores.append((question, 0.0))

    if not question_wise_scores:
        raise ValueError(f"Could not parse verifier scores from response:\n{response_text}")

    mean_score = sum(score for _, score in question_wise_scores) / len(question_wise_scores)
    question_wise_scores.append(("Cumulative mean binary score", mean_score))
    return question_wise_scores


def score_from_scores(scores: list[tuple[str, float]] | None) -> float:
    if not scores:
        return 0.0
    for question, score in scores:
        if question.strip().lower().startswith("cumulative mean"):
            return float(score)
    return float(sum(score for _, score in scores) / len(scores))
