from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from generate_flux_images_and_detect_objects import gen_prompt_str


def sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def count_items(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def load_state(path: Path) -> tuple[set[str], dict[str, int]]:
    if not path.exists():
        return set(), {}
    with path.open("rb") as f:
        files_processed, rel_transition_counts = pickle.load(f)
    return set(files_processed), dict(rel_transition_counts)


def save_state(path: Path, files_processed: set[str], rel_transition_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump((sorted(files_processed), rel_transition_counts), f)


def transition_allowed(transition: str, counts: dict[str, int], max_samples: int) -> bool:
    return counts.get(transition, 0) < max_samples


def mark_transition(transition: str, counts: dict[str, int]) -> None:
    counts[transition] = counts.get(transition, 0) + 1


def next_index(output_dir: Path) -> int:
    return len(list(output_dir.glob("*_condition.png")))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_black_condition(reference_image_path: str | Path, output_path: Path) -> None:
    size = Image.open(reference_image_path).size
    Image.new("RGB", size, (0, 0, 0)).save(output_path)


def build_records(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, int, int]:
    original_items = data["original_items"]
    original_rel_tuples = data["original_rel_tuples"]
    removed_relations = [tuple(rel) for rel in data["removed_relations"]]
    removed_objects = set(data["removed_objects"])

    full_context_prompt = gen_prompt_str(
        original_rel_tuples,
        data.get("style"),
        data.get("background"),
        count_items(original_items),
    )

    zeroth_step_rel_tuples = [tuple(rel) for rel in original_rel_tuples if tuple(rel) not in removed_relations]
    zeroth_step_objects = [obj for obj in original_items if obj not in removed_objects]
    num_rels_pre = len(zeroth_step_rel_tuples)
    num_rels_post = len(original_rel_tuples)

    if data["is_direct_0th_step"]:
        zeroth_step_prompt = full_context_prompt
    else:
        zeroth_step_prompt = gen_prompt_str(
            zeroth_step_rel_tuples,
            data.get("style"),
            data.get("background"),
            count_items(zeroth_step_objects),
        )

    first_step_data = {
        "is_first_step": True,
        "original_items": original_items,
        "original_rel_tuples": original_rel_tuples,
        "background": data.get("background"),
        "style": data.get("style"),
        "removed_relations": data["removed_relations"],
        "removed_objects": data["removed_objects"],
        "step_relation_tuples": zeroth_step_rel_tuples,
        "step_objects": zeroth_step_objects,
        "addition_prompt_v1": data["addition_prompt"],
        "addition_prompt_v2": data["addition_promptv2"],
        "actual_step_by_step_prompt": zeroth_step_prompt,
        "full_context_prompt": full_context_prompt,
    }

    if data["is_direct_0th_step"]:
        return first_step_data, None, num_rels_post, num_rels_post

    second_step_data = {
        "is_first_step": False,
        "original_items": original_items,
        "original_rel_tuples": original_rel_tuples,
        "background": data.get("background"),
        "style": data.get("style"),
        "removed_relations": data["removed_relations"],
        "removed_objects": data["removed_objects"],
        "step_relation_tuples": original_rel_tuples,
        "step_objects": original_items,
        "addition_prompt_v1": data["addition_prompt"],
        "addition_prompt_v2": data["addition_promptv2"],
        "actual_step_by_step_prompt": data["addition_promptv2"],
        "full_context_prompt": full_context_prompt,
    }
    return first_step_data, second_step_data, num_rels_pre, num_rels_post


def collate_record(data: dict[str, Any], output_dir: Path, start_idx: int) -> int:
    first_step_data, second_step_data, _, _ = build_records(data)

    write_json(output_dir / f"{start_idx}.json", first_step_data)
    make_black_condition(data["post_image_path"], output_dir / f"{start_idx}_condition.png")
    first_target = data["pre_image_path"] if data["pre_image_path"] is not None else data["post_image_path"]
    shutil.copy(first_target, output_dir / f"{start_idx}.png")

    if second_step_data is None:
        return start_idx + 1

    next_idx = start_idx + 1
    write_json(output_dir / f"{next_idx}.json", second_step_data)
    shutil.copy(data["pre_image_path"], output_dir / f"{next_idx}_condition.png")
    shutil.copy(data["post_image_path"], output_dir / f"{next_idx}.png")
    return next_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Collate removal outputs into image-to-image training triples.")
    parser.add_argument("--input_primary_dir", type=str, default="outputs/synthetic_flux_qwen")
    parser.add_argument("--output_dir", type=str, default="outputs/synthetic_flux_qwen/collated_train_pairs")
    parser.add_argument("--max_samples_per_k", type=int, default=80000)
    parser.add_argument("--state_file", type=str, default=None, help="Resume-state pickle. Defaults to OUTPUT_DIR/post_processing_dirs_processed.pkl.")
    args = parser.parse_args()

    input_primary_dir = Path(args.input_primary_dir)
    output_dir = Path(args.output_dir)
    state_file = Path(args.state_file) if args.state_file else output_dir / "post_processing_dirs_processed.pkl"

    if not input_primary_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_primary_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_processed, rel_transition_counts = load_state(state_file)
    input_dirs = sorted(
        [
            path
            for path in input_primary_dir.iterdir()
            if path.is_dir() and "512_sbatch" in path.name and "final_inpainting_" in path.name
        ],
        key=lambda path: sort_key(path.name),
    )
    print("valid_input_dirs")
    print([str(path) for path in input_dirs])

    current_idx = next_index(output_dir)
    for input_dir in input_dirs:
        sample_dirs = sorted([path for path in input_dir.iterdir() if path.is_dir()], key=lambda path: sort_key(path.name))
        for sample_dir in tqdm(sample_dirs):
            for json_path in sorted(sample_dir.glob("*.json")):
                record_id = str(json_path)
                if record_id in files_processed:
                    continue

                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                _, second_step_data, num_rels_pre, num_rels_post = build_records(data)
                first_transition = f"0 -> {num_rels_pre}"
                if not transition_allowed(first_transition, rel_transition_counts, args.max_samples_per_k):
                    continue

                if second_step_data is not None:
                    second_transition = f"{num_rels_pre} -> {num_rels_post}"
                    if not transition_allowed(second_transition, rel_transition_counts, args.max_samples_per_k):
                        continue
                else:
                    second_transition = None

                current_idx = collate_record(data, output_dir, current_idx)
                mark_transition(first_transition, rel_transition_counts)
                if second_transition is not None:
                    mark_transition(second_transition, rel_transition_counts)
                files_processed.add(record_id)
                save_state(state_file, files_processed, rel_transition_counts)


if __name__ == "__main__":
    main()
