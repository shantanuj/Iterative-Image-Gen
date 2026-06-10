from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from qwen_edit_utils import QwenImageEditClient
from generate_flux_images_and_detect_objects import RELATIONS_DICT


def relation_touches_any_object(rel: list[str] | tuple[str, str, str], objects: set[str]) -> bool:
    return rel[0] in objects or rel[2] in objects


def build_item_to_relations(rel_tuples: list[list[str] | tuple[str, str, str]]) -> dict[str, dict[str, list[Any]]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for rel in rel_tuples:
        subj, _, obj = rel
        out.setdefault(subj, {"sub": [], "obj": []})["sub"].append(tuple(rel))
        out.setdefault(obj, {"sub": [], "obj": []})["obj"].append(tuple(rel))
    return out


def get_addition_prompt(
    removed_objects: list[str],
    removed_relations: list[tuple[str, str, str]],
    all_relations: list[list[str] | tuple[str, str, str]],
) -> str:
    if not removed_objects:
        return ""

    if len(removed_objects) == 1:
        prompt = f"Put a {removed_objects[0]}. "
    else:
        prompt = "Put " + ", ".join(f"a {obj}" for obj in removed_objects[:-1])
        prompt += f", and a {removed_objects[-1]}. "

    item_to_relations = build_item_to_relations(all_relations)
    covered: set[str] = set()
    relation_parts: list[str] = []
    removed_set = set(removed_objects)

    for item in removed_objects:
        rels = item_to_relations.get(item, {"sub": [], "obj": []})
        covered.add(item)
        for rel in rels["sub"]:
            if rel[2] in covered:
                continue
            phrase = RELATIONS_DICT[rel[1]]["subj_phrase"]
            aux = "the" if rel[2] in removed_set else "the existing"
            relation_parts.append(f"The {rel[0]} is {phrase} {aux} {rel[2]}.")
        for rel in rels["obj"]:
            if rel[0] in covered:
                continue
            phrase = RELATIONS_DICT[rel[1]]["obj_phrase"]
            aux = "the" if rel[0] in removed_set else "the existing"
            relation_parts.append(f"The {rel[2]} is {phrase} {aux} {rel[0]}.")

    return (prompt + " ".join(relation_parts)).strip()


def get_addition_prompt_v2(
    removed_objects: list[str],
    removed_relations: list[tuple[str, str, str]],
    all_relations: list[list[str] | tuple[str, str, str]],
) -> str:
    item_to_relations = build_item_to_relations(all_relations)
    covered: set[str] = set()
    removed_set = set(removed_objects)
    parts: list[str] = []

    for item in removed_objects:
        rels = item_to_relations.get(item, {"sub": [], "obj": []})
        covered.add(item)
        item_parts: list[str] = []
        for rel in rels["sub"]:
            if rel[2] in covered:
                continue
            phrase = RELATIONS_DICT[rel[1]]["subj_phrase"]
            aux = "the" if rel[2] in removed_set else "the existing"
            item_parts.append(f"Put a {rel[0]} {phrase} {aux} {rel[2]}")
        for rel in rels["obj"]:
            if rel[0] in covered:
                continue
            phrase = RELATIONS_DICT[rel[1]]["obj_phrase"]
            aux = "the" if rel[0] in removed_set else "the existing"
            item_parts.append(f"Put a {rel[2]} {phrase} {aux} {rel[0]}")
        if item_parts:
            parts.append(", and ".join(item_parts) + ".")

    if parts:
        return " ".join(parts).replace("  ", " ").strip()
    return get_addition_prompt(removed_objects, removed_relations, all_relations)


def choose_removed_objects(items: list[str], max_items_to_select: int, rng: random.Random) -> list[str]:
    unique_items = list(dict.fromkeys(items))
    if len(unique_items) <= 1:
        return []
    max_remove = min(max_items_to_select, len(unique_items) - 1)
    num_remove = rng.randint(1, max_remove)
    return rng.sample(unique_items, num_remove)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_direct_record(data: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return {
        "original_items": data["items"],
        "original_rel_tuples": data["rel_tuples"],
        "background": data.get("background"),
        "style": data.get("style"),
        "removed_relations": [],
        "removed_objects": [],
        "addition_prompt": data.get("prompt", ""),
        "addition_promptv2": data.get("prompt", ""),
        "is_direct_0th_step": True,
        "pre_image_path": None,
        "post_image_path": str(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove objects with Qwen-Image-Edit and write step-by-step training metadata.")
    parser.add_argument("--source_folder", required=True, help="Folder containing Step 1 images and Step 2 *_relabelled.json files.")
    parser.add_argument("--output_folder", default=None, help="Defaults to SOURCE_final_inpainting_qwenedit.")
    parser.add_argument("--qwen-edit-base-url", default=os.environ.get("QWEN_EDIT_BASE_URL", "http://localhost:8092/v1"))
    parser.add_argument("--qwen-edit-model", default=os.environ.get("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit"))
    parser.add_argument("--qwen-edit-api-key", default=os.environ.get("QWEN_EDIT_API_KEY", "none"))
    parser.add_argument("--max_items_to_select", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--true_cfg_scale", type=float, default=None)
    parser.add_argument("--skip_direct_0th_step", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_folder = Path(args.source_folder)
    output_folder = Path(args.output_folder) if args.output_folder else Path(f"{args.source_folder}_final_inpainting_qwenedit")
    output_folder.mkdir(parents=True, exist_ok=True)

    client = QwenImageEditClient(
        base_url=args.qwen_edit_base_url,
        model=args.qwen_edit_model,
        api_key=args.qwen_edit_api_key,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
    )
    rng = random.Random(args.seed)

    image_paths = sorted(source_folder.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    for image_path in tqdm(image_paths):
        file_id = image_path.stem
        relabelled_path = source_folder / f"{file_id}_relabelled.json"
        if not relabelled_path.exists():
            continue

        sample_dir = output_folder / file_id
        if (sample_dir / "qwen_remove.json").exists():
            continue

        with relabelled_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not args.skip_direct_0th_step:
            write_json(sample_dir / "direct_0th.json", build_direct_record(data, image_path))

        removed_objects = choose_removed_objects(data["items"], args.max_items_to_select, rng)
        if not removed_objects:
            continue

        removed_set = set(removed_objects)
        removed_relations = [tuple(rel) for rel in data["rel_tuples"] if relation_touches_any_object(rel, removed_set)]
        instruction = (
            "Remove "
            + ", ".join(f"the {obj}" for obj in removed_objects)
            + ". Preserve the rest of the scene, style, background, camera view, lighting, and all remaining objects."
        )

        edited_path = sample_dir / "qwen_removed.png"
        client.edit(image_path=image_path, instruction=instruction, output_path=edited_path, seed=args.seed)

        # Normalize to RGB PNG so downstream PIL loading is predictable.
        Image.open(edited_path).convert("RGB").save(edited_path)

        record = {
            "original_items": data["items"],
            "original_rel_tuples": data["rel_tuples"],
            "background": data.get("background"),
            "style": data.get("style"),
            "removed_relations": [list(rel) for rel in removed_relations],
            "removed_objects": removed_objects,
            "addition_prompt": get_addition_prompt(removed_objects, removed_relations, data["rel_tuples"]),
            "addition_promptv2": get_addition_prompt_v2(removed_objects, removed_relations, data["rel_tuples"]),
            "remove_instruction": instruction,
            "is_direct_0th_step": False,
            "pre_image_path": str(edited_path),
            "post_image_path": str(image_path),
        }
        write_json(sample_dir / "qwen_remove.json", record)


if __name__ == "__main__":
    main()
