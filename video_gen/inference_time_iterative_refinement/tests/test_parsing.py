from iterative_video_gen.io_utils import (
    extract_question_wise_scores,
    parse_action_prompt,
    parse_json_questions,
    score_from_scores,
)


def test_parse_action_prompt():
    action, prompt = parse_action_prompt("Action: CONTINUE\nPrompt: Add a red bird.")
    assert action == "CONTINUE"
    assert prompt == "Add a red bird."


def test_parse_json_questions():
    assert parse_json_questions('["Is there a dog?", "Is it walking?"]') == [
        "Is there a dog?",
        "Is it walking?",
    ]


def test_extract_question_wise_scores_fills_missing():
    questions = ["Is there a dog?", "Is there a cat?"]
    scores = extract_question_wise_scores("Is there a dog?: yes", questions)
    assert scores[0] == ("Is there a dog?", 1.0)
    assert scores[1] == ("Is there a cat?", 0.0)
    assert score_from_scores(scores) == 0.5
