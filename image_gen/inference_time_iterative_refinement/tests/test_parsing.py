from iterative_img_gen.config import load_config
from iterative_img_gen.io_utils import (
    extract_question_wise_scores,
    parse_first_prompt,
    parse_next_step,
)


def test_prompt_parsers():
    assert parse_first_prompt("Output: a wide shot of a room") == "a wide shot of a room"
    assert parse_next_step("Action: BACKTRACK\nPrompt: fix the red cube") == (
        "BACKTRACK",
        "fix the red cube",
    )


def test_verifier_score_parser():
    scores = extract_question_wise_scores(
        "Is there a cube?: yes\nIs it blue?: no",
        ["Is there a cube?", "Is it blue?"],
    )
    assert scores[-1] == ("Cumulative mean binary score:", 0.5)


def test_example_config_loads():
    cfg = load_config("configs/qwen_vllm_omni.yaml")
    assert cfg["models"]["generator"]["provider"] == "qwen-image"
    assert cfg["models"]["editor"]["provider"] == "qwen-image-edit"
    assert cfg["method"]["rephrase_first_step"] is False
    assert cfg["method"]["do_first_step_as_original_prompt"] is True


def test_rephrase_first_step_sets_legacy_inverse(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("method:\n  rephrase_first_step: true\n", encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg["method"]["rephrase_first_step"] is True
    assert cfg["method"]["do_first_step_as_original_prompt"] is False
