import json
import os
# append the path to the root directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './Grounded_SAM2')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

import random
import torch
from diffusers import FluxPipeline
from Grounded_SAM2.sam2.build_sam import build_sam2
from Grounded_SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import numpy as np
import cv2
import supervision as sv
import argparse
import matplotlib.pyplot as plt
import gc
import textwrap
import pickle

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")

ITEMS_DICT = {
    "basketball": ["physical_object", "inanimate_objects", "play_object"],
    "soccer ball": ["physical_object", "inanimate_objects", "play_object"],
    #""
    "refrigerator": ["physical_object", "inanimate_objects", "household_appliance"],
    "box": ["physical_object", "inanimate_objects", "container", "play_object"],
    "watermelon": ["physical_object", "inanimate_objects", "food", "play_object"],
    #"melon": ["physical_object", "inanimate_objects", "food"],
    "pumpkin": ["physical_object", "inanimate_objects", "food", "play_object"],
    "tomato": ["physical_object", "inanimate_objects", "food", "play_object"],
    "pyramid": ["physical_object", "inanimate_objects", "structure"],
    "cactus": ["physical_object", "inanimate_objects", "plant", "play_object"],
    #"doll": ["physical_object", "inanimate_objects", "toy", "play_object"],
    "teddy bear": ["physical_object", "inanimate_objects", "toy", "play_object"],
    "trashcan": ["physical_object", "inanimate_objects", "container", "play_object"],
    "bottle": ["physical_object", "inanimate_objects", "container", "play_object"],
    #""
    #"toy": ["physical_object", "inanimate_objects", "play_object"],
    
    "ambulance": ["vehicles", "physical_object", "emergency_vehicle"],
    #"car": ["vehicles", "physical_object", "transportation"],
    #"trashcan": ["vehicles", "physical_object", "transportation"],
    "car": ["vehicles", "physical_object", "transportation"],
    #"suv": ["vehicles", "physical_object", "transportation"],
    #"truck": ["vehicles", "physical_object", "transportation"],
    
    "firetruck": ["vehicles", "physical_object", "emergency_vehicle"],
    "school bus": ["vehicles", "physical_object", "transportation"],
    #""
    "dog": ["animals", "physical_object", "play_agent", "pet", "play_object"],
    "baloon": ["physical_object", "inanimate_objects", "play_object"],
    #""
    #"deer": ["animals", "physical_object", "farm_animal", "play_object"],
    "penguin": ["animals", "physical_object", "play_agent", "pet", "play_object"],
    "panda": ["animals", "physical_object", "play_agent", "pet", "play_object"],
    "rabbit": ["animals", "physical_object", "play_agent", "pet", "play_object"],
    #"fox": ["animals", "physical_object", "play_agent", "pet", "play_object"],
    "zebra": ["animals", "physical_object", "play_agent", "pet", "play_object"],

    #""
    #"corgi": ["animals", "physical_object", "play_agent", "dog", "pet", "play_object"],
    #"dalmation": ["animals", "physical_object", "play_agent", "dog", "pet", "play_object"],
    #"poodle": ["animals", "physical_object", "play_agent", "dog", "pet", "play_object"],
    #"beagle": ["animals", "physical_object", "play_agent", "dog", "pet", "play_object"],
    "cat": ["animals", "physical_object", "play_agent", "pet", "play_object"],
    "sheep": ["animals", "physical_object", "farm_animal", "play_object"],
    "pig": ["animals", "physical_object", "farm_animal", "play_object"],
    "horse": ["animals", "physical_object", "farm_animal", "play_object"],
    "cow": ["animals", "physical_object", "farm_animal", "play_object"],
    "duck": ["animals", "physical_object", "farm_animal", "play_object"],
    
    
    "man": ["humans", "physical_object", "play_agent", "person"],
    "woman": ["humans", "physical_object", "play_agent", "person"],
    #"doctor": ["humans", "physical_object", "play_agent", "person"],
    #""
    
    #"robot": ["inanimate_objects", "physical_object", "artificial_agent", "play_object", "play_agent"]
}

ITEM_TO_PLURAL = {
    "basketball": "basketballs",
    "soccer ball": "soccer balls",
    "refrigerator": "refrigerators",
    "box": "boxes",
    "watermelon": "watermelons",
    "melon": "melons",
    "pumpkin": "pumpkins",
    "coconut": "coconuts",
    "pyramid": "pyramids",
    "cactus": "cacti",
    "doll": "dolls",
    "teddy bear": "teddy bears",
    "toy": "toys",
    "ambulance": "ambulances",
    "car": "cars",
    "firetruck": "firetrucks",
    "school bus": "school buses",
    "corgi": "corgis",
    "dalmation": "dalmations",
    "poodle": "poodles",
    "beagle": "beagles",
    "cat": "cats",
    "sheep": "sheep",
    "pig": "pigs",
    "horse": "horses",
    "cow": "cows",
    "man": "men",
    "woman": "women",
    "robot": "robots"
}

class_items_combinations = []
# Create NxN combinations of all objects
items = list(ITEMS_DICT.keys())
for item1 in items:
    for item2 in items:
        #if item1 != item2:  # Avoid pairing an item with itself
        class_items_combinations.append(f"{item1} {item2}")
CATEGORY_TO_ITEM_LIST = {}
for item, categories in ITEMS_DICT.items():
    for category in categories:
        if category not in CATEGORY_TO_ITEM_LIST:
            CATEGORY_TO_ITEM_LIST[category] = []
        CATEGORY_TO_ITEM_LIST[category].append(item)

# Flatten the dictionary for backward compatibility
ITEM_LIST_FLAT = []
for item, categories in ITEMS_DICT.items():
    ITEM_LIST_FLAT.extend([item] * 2)

BACKGROUND_LIST = ["lawn", "beach", "desert", "room", "road","park", None]
STYLE_LIST = ["cartoon", "sketch", "pixel art", "photorealistic", None]
RELATIONS_DICT = {
    # Spatial relations that apply to all categories
    'is on left of': {'agent': ['physical_object'], 
                'object': ['physical_object'],
                'subj_phrase': 'on left of',
                'obj_phrase': 'on right of'
                },
    'is on right of': {'agent': ['physical_object'], 
                 'object': ['physical_object'],
                 'subj_phrase': 'on right of',
                 'obj_phrase': 'on left of'
                 },
    'is above': {'agent': ['physical_object'], 
              'object': ['physical_object'],
              'subj_phrase': 'above',
              'obj_phrase': 'below'
              },
    #'is on top of': {'agent': ['physical_object'], 
     #         'object': ['physical_object']},
    #'below': {'agent': ['physical_object'], 
     #         'object': ['physical_object']},
    
    # Relations requiring animate agents
    #'looking at': {'agent': ['animals', 'humans', 'artificial_agent'], 
     #              'object': ['inanimate_objects', 'vehicles', 'animals', 'humans']},
    'is looking at': {'agent': ['animals', 'humans'],#, 'artificial_agent'], 
               'object': ['physical_object'],
               'subj_phrase': 'looking at',
               'obj_phrase': 'being looked at by'
               },
    #'is facing away from': {'agent': ['animals', 'humans', 'artificial_agent'], 
     #                     'object': ['physical_object']},
    
    # Relations requiring humans or animals as agents
    'is holding': {'agent': ['humans', 'animals'],#, 'artificial_agent'], 
                'object': ['inanimate_objects'],
                'subj_phrase': 'holding',
                'obj_phrase': 'being held by'
                },
    'is sitting on': {'agent': ['humans','animals'], 
                   'object': ['physical_object'],
                   'subj_phrase': 'sitting on',
                   'obj_phrase': 'sitting on'
                   },
    #'is placed on': {'agent': ['physical_object'], 
     #              'object': ['physical_object']},
    'is standing on': {'agent': ['humans','animals'], 
                    'object': ['physical_object'],
                    'subj_phrase': 'standing on',
                    'obj_phrase': 'being stood on by'
                    },
    'is lying on': {'agent': ['humans','animals'], 
                 'object': ['physical_object'],
                 'subj_phrase': 'lying on',
                 'obj_phrase': 'being lain on by'
                 },
    
    # Relations requiring animals as agents
    'is chasing': {'agent': ['animals', 'humans', 'play_agent'], 
                'object': ['animals', 'humans','vehicles'],
                'subj_phrase': 'chasing',
                'obj_phrase': 'being chased by'
                },
    'is eating': {'agent': ['animals', 'humans'], 
               'object': ['food'],
               'subj_phrase': 'eating',
               'obj_phrase': 'being eaten by'
               },
    
    'is touching': {'agent': ['humans','animals'], 
                'object': ['physical_object'],
                'subj_phrase': 'touching',
                'obj_phrase': 'being touched by'
                },
    'is driving': {'agent': ['humans','animals'], 
                'object': ['vehicles'],
                'subj_phrase': 'driving',
                'obj_phrase': 'being driven by'
                },
    
    
}

category_to_agent_relations = {}
for rel, rel_dict in RELATIONS_DICT.items():
    for category in rel_dict['agent']:
        if category not in category_to_agent_relations:
            category_to_agent_relations[category] = []
        category_to_agent_relations[category].append(rel)

# For backward compatibility
RELATIONS_LIST = list(RELATIONS_DICT.keys())
count_to_str = {
    1: "one",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth"
}
count_to_str_number = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten"
}

count_to_ordinal = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
}

from copy import deepcopy

def get_vocab():
    return ITEMS_DICT, RELATIONS_DICT, STYLE_LIST, BACKGROUND_LIST
def prompt_generator(num_rels=5, spatial_relation=None):
    relations = list(RELATIONS_DICT.keys())
    style = random.choice(STYLE_LIST)
    background = random.choice(BACKGROUND_LIST)
    #prompt = "Generate an im"
    rels = []
    items = []
    #num_rels = num_items -1
    item_counts = {}
    #item_categories_covered = {}
    #rels_prompt = {}
    rel_tuples = []
    max_tries = 10
    for i in range(num_rels):
        if(i==0):
            rel = random.choice(relations)
        else:
            #st()
            prev_item2_categories = ITEMS_DICT[item2]
            valid_rels = []
            for cat in prev_item2_categories:
                if(cat in category_to_agent_relations):
                    valid_rels.extend(category_to_agent_relations[cat])
            rel = random.choice(valid_rels)
        rels.append(rel)
        agent_categories = RELATIONS_DICT[rel]['agent']
        item_categories = RELATIONS_DICT[rel]['object']
        if(i==0):
            
            valid=False
            num_tries = 0
            while(not valid and num_tries < max_tries):
                item1_category = random.choice(agent_categories)
                item1 = random.choice(CATEGORY_TO_ITEM_LIST[item1_category])
                if(item1 not in item_counts):
                    valid=True
                num_tries += 1
            if(num_tries >= max_tries):
                print("Failed to find a valid item 1")
                return -1
            items.append(item1)
            if item1 not in item_counts:
                item_counts[item1] = 1
            else:
                item_counts[item1] += 1
            
        
        else:
            #st()
            #item1_category = agent_ca
            item1 = item2
        #st()
        valid=False
        num_tries = 0
        while(not valid and num_tries < max_tries):
            item2_category = random.choice(item_categories)
            item2 = random.choice(CATEGORY_TO_ITEM_LIST[item2_category])
            if(item2 not in item_counts):
                valid=True
            num_tries += 1
        if(num_tries >= max_tries):
            print("Failed to find a valid item 2")
            return -1
        items.append(item2)
        if item2 not in item_counts:
            item_counts[item2] = 1
        else:
            item_counts[item2] += 1
        rel_tuples.append((item1, rel, item2))

    prompt = gen_prompt_str(rel_tuples, style, background, item_counts)

        
        
    items_set = list(set(items))
    #st()
    return prompt, rel_tuples, background, style, items_set, item_counts


def gen_prompt_str(rel_tuples, style, background, item_counts):
    txt = ""

    for i, item in enumerate(item_counts):
        count_str = count_to_str_number[item_counts[item]]
        if(item_counts[item] > 1):
            item = ITEM_TO_PLURAL[item]
        if(i == len(item_counts) - 1 and len(item_counts) > 1):
            txt += f"and {count_str} {item}"
        else:
            if(len(item_counts) == 1):
                txt += f"{count_str} {item}"
            else:
                txt += f"{count_str} {item}, "
        
    txt = txt.strip()    
    if(style is None):
        prompt = f"Generate an image with {txt}."
    else:
        prompt = f"Generate an image in {style} style with {txt}."
    if(background is not None):
        prompt += f" The background is an empty clean {background}."
    
    covered_counts = {}
    rel_prompt = " "
    #random.shuffle(rel_tuples)
    for rel_i, rel_item in enumerate(rel_tuples):
        if(item_counts[rel_item[0]] == 1):
            first_item = rel_item[0]
        else:
            if(rel_item[0] not in covered_counts):
                covered_counts[rel_item[0]] = 0
            if(rel_i == 0):
                covered_counts[rel_item[0]] += 1
            first_item = count_to_ordinal[covered_counts[rel_item[0]]] + " " + rel_item[0]
        if(item_counts[rel_item[2]] == 1):
            second_item = rel_item[2]
        else:
            if(rel_item[2] not in covered_counts):
                covered_counts[rel_item[2]] = 0
            covered_counts[rel_item[2]] += 1
            second_item = count_to_ordinal[covered_counts[rel_item[2]]] + " " + rel_item[2]
        
        rel_prompt += f"The {first_item} {rel_item[1]} the {second_item}. "

    prompt += rel_prompt
    prompt = prompt.strip()
    return prompt


def number_mapping(count):
    if count == 1:
        t = "one"
    elif count == 2:
        t = "two"
    elif count == 3:
        t = "three"
    elif count == 4:
        t = "four"
    elif count == 5:
        t = "five"
    elif count == 6:
        t = "six"
    elif count == 7:
        t = "seven"
    elif count == 8:
        t = "eight"
    elif count == 9:
        t = "nine"
    else:
        t = "many"
    return t


inv_relation_dict = {
    'is on right of': 'is on left of',
    'is on left of': 'is on right of',
    'is above': 'is below',
    'is below': 'is above',
    'is on top of': 'is on bottom of',
    'is on bottom of': 'is on top of',
    
}
if __name__ == "__main__":
    # add arguments to the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Number of items in the prompt")
    parser.add_argument("--dataset_folder_name", type=str, default="./dataset_multi_object_expanded_vocab/dataset_512_1", help="Name of the dataset folder")
    parser.add_argument("--max_num_rels", type=int, default=4, help="Number of max relations in the prompt")
    parser.add_argument("--num_samples_to_generate_per_k", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--verbose", type=str2bool, default=False, help="Whether to print verbose output")
    args = parser.parse_args()

    DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    
    
    # directory to save the dataset
    dataset_folder_name = args.dataset_folder_name
    if not os.path.exists(dataset_folder_name):
        os.makedirs(dataset_folder_name, exist_ok=True)
        rels_counts_covered = {k:0 for k in range(1, args.max_num_rels+1)}
        rels_counts_tried = {k:0 for k in range(1, args.max_num_rels+1)}
        num_files = 0
    else:
        files = os.listdir(dataset_folder_name)
        num_files = len([f for f in files if f.endswith(".json")])
        rels_counts_covered = pickle.load(open(os.path.join(dataset_folder_name, "rels_counts_covered.pkl"), "rb"))
        rels_counts_tried = pickle.load(open(os.path.join(dataset_folder_name, "rels_counts_tried.pkl"), "rb"))
        for k in range(1, args.max_num_rels+1):
            if k not in rels_counts_covered:
                rels_counts_covered[k] = 0
            if k not in rels_counts_tried:
                rels_counts_tried[k] = 0

    # Load the FLUX pipeline
    model_directory = None
    

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir=model_directory,
    ).to(DEVICE)

    # build SAM2 image predictor
    # create the text prompt for grounding dino
    #TEXT_PROMPT = ". ".join(ITEM_LIST) + "."
    TEXT_PROMPT = ". ".join(list(set(ITEMS_DICT.keys()))) + "."
    SAM2_CHECKPOINT = "Grounded_SAM2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "Grounded_SAM2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "Grounded_SAM2/gdino_checkpoints/groundingdino_swint_ogc.pth"

    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_directory)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

    # only use 2 prompts for now
    # prompt_pairs = random.sample(prompt_pairs.items(), 10)
    #key = 10000 * (args.cuda)
    #st()
    key = num_files #overwrite the last one just in case
    while True:#key < args.num_samples_to_generate:
        rels_to_consider = [k for k in range(1, args.max_num_rels+1) if rels_counts_covered[k] < args.num_samples_to_generate_per_k]
        #print("Rels counts covered: ", rels_counts_covered)
        if(len(rels_to_consider) == 0):
            print(f"Done generating all relations")
            break
        num_rels = random.choice(rels_to_consider)
        #rels_counts_covered[num_rels] += 1

        #num_rels = random.randint(2, args.max_num_rels)
        
        out = prompt_generator(num_rels=num_rels)
        if(out==-1):
            continue
        prompt, rel_tuples, background, style, objs_set, item_counts = out
        #if(len(objs) != num_rels+1):
        
        # generate image with prompt using FLUX
        images = pipe(
                    prompt,
                    num_images_per_prompt = 4,
                    height=512,
                    width=512,
                    guidance_scale=random.choice([3.5, 4.0, 4.5, 5.0]),
                    num_inference_steps=20,
                    max_sequence_length=512,
                    # generator=torch.Generator("cpu").manual_seed(0)
                ).images        # list of PIL images
        
        
        #st()
        # release the memory
        torch.cuda.empty_cache()  # clear the cache to prevent memory leak

        """
        Start to check if the image is correct
        """

        for image in images:
            prompt_pairs = {
                'prompt': prompt,
                'rel_tuples': rel_tuples,
                'background': background,
                'style': style
            }
            rels_counts_tried[num_rels] += 1
            with open(os.path.join(dataset_folder_name, "rels_counts_tried.pkl"), "wb") as f:
                pickle.dump(rels_counts_tried, f)
            

            # setup the input image and text prompt for SAM 2 and Grounding DINO
            sam2_predictor.set_image(np.array(image.convert("RGB")))
            input_boxes = []
            input_scores = []
            input_text_labels = []
            text_prompt = ". ".join(objs_set) + "."
            text_prompt = text_prompt.strip()
            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )
            input_boxes.append(results[0]["boxes"].cpu().numpy())
            input_scores.append(results[0]["scores"].cpu().numpy())
            input_text_labels.append(results[0]["labels"])
            ##input_boxes = np.concatenate(input_boxes, axis=0)
            #input_scores = np.concatenate(input_scores, axis=0)
           # input_text_labels = np.concatenate(input_text_labels, axis=0)
            #if(input_boxes.shape[0] == 0):
             #   print(f"WRONG: No boxes are detected in the image")
              #  print(f"Image {key} is wrong")
               # continue

            #image.save("temp_image.png")
            #st()

            #"""
            missing_labels = []
            if(rels_counts_covered[num_rels]/rels_counts_tried[num_rels]<0.3): #if less than 30% acceptance rate, then consider adding the missing labels (to be more lenient)
                for label in objs_set:
                    if label not in input_text_labels[0]:
                        missing_labels.append(label)
            
            #st()

            for obj in missing_labels:
                text_prompt = f"{obj}."
            #text_prompt = '. '.join(objs_set)
            #text_prompt+= '.'
                text_prompt = text_prompt.strip()
                inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
            
                # run the models
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]]
                )
                input_boxes.append(results[0]["boxes"].cpu().numpy())
                input_scores.append(results[0]["scores"].cpu().numpy())
                input_text_labels.append(results[0]["labels"])
            #"""

            
            input_boxes = np.concatenate(input_boxes, axis=0)
            input_scores = np.concatenate(input_scores, axis=0)
            input_text_labels = np.concatenate(input_text_labels, axis=0)
            #st()
            if(input_boxes.shape[0] == 0):
                if(args.verbose):
                    print(f"WRONG: No boxes are detected in the image")
                    print(f"Image {key} is wrong")
                continue
            missing_labels = []
            for label in objs_set:
                if label not in input_text_labels:
                    missing_labels.append(label)
            if(len(missing_labels) > 0):
                if(args.verbose):
                    print(f"WRONG: Missing labels in the image: {missing_labels}")
                    print(f"Image {key} is wrong")
                continue
            if(len(input_text_labels) != len(objs_set) and (rels_counts_covered[num_rels]/rels_counts_tried[num_rels]>0.3)): #if the number of labels is not equal to the number of objects, and the percentage of times the relation has been accepted is more than 25%
                #st()
                if(args.verbose):
                    print(f"WRONG: Number of labels in the image: {len(input_text_labels)} is not equal to the number of objects in the prompt: {len(objs_set)}")
                    print(f"Image {key} is wrong")
                continue
            #for label in input_text_labels:

            #else:
             #   print(f"")
            #print("Results: ", results)
            
            """
            Results is a list of dict with the following structure:
            [
                {
                    'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
                    'labels': ['car', 'tire', 'tire', 'tire'], 
                    'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                                    [1392.4701,  554.4064, 1628.6133,  777.5872],
                                    [ 436.1182,  621.8940,  676.5255,  851.6897],
                                    [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
                }
            ]
            """

            # get the box prompt for SAM 2
            #input_boxes = results[0]["boxes"].cpu().numpy()

            # if no boxes are detected, skip the image
            #if input_boxes.shape[0] == 0:
             #   print("No boxes are detected in the image")
              #  print(f"Image {key} is wrong")
               # continue

            # run SAM 2
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            #st()

            # Post process the output of the model
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = input_scores.tolist()
            class_names = [str(text) for text in input_text_labels]
            #st()

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]
            #print(labels)

            # use the class names as the ground truth items
            prompt_pairs["items"] = class_names

            for class_name in class_names:
                if class_name in class_items_combinations:
                    # skip the image
                    print(f"{class_names} is wrong")
                    continue

            # get the spatial relation among the detected items

            def predict_spatial_relation(box_A, box_B):
                # predict the spatial relation between two bounding boxes
                # also return the distance between the two bounding boxes
                # get the center of the bounding box
                center_A = [(box_A[0]+box_A[2])/2, (box_A[1]+box_A[3])/2]
                center_B = [(box_B[0]+box_B[2])/2, (box_B[1]+box_B[3])/2]
                x_A, y_A = center_A
                x_B, y_B = center_B
                height_A = box_A[3] - box_A[1]
                width_A = box_A[2] - box_A[0]
                height_B = box_B[3] - box_B[1]
                width_B = box_B[2] - box_B[0]

                
                # Check for x-axis overlap
                x_overlap = not (box_A[2] < box_B[0] or box_A[0] > box_B[2])

                if (box_A[3] < (y_B - height_B/4)) and x_overlap:   # is on the top of
                    distance = y_B - box_A[3]
                    return "is above", distance
                #elif
                #elif (box_A[1] > (y_B + height_B*2/5)) and x_overlap:   # is under
                 #   distance = box_A[1] - y_B
                  #  return "is under", distance
                elif box_A[0] > (box_B[2] - width_B/4):   # is on the right side of
                    distance = box_A[2] - box_B[2]
                    return "is on right of", distance
                elif box_A[2] < (box_B[0] + width_B/4):   # is on the left side of
                    distance = box_B[0] - box_A[0]
                    return "is on left of", distance
                else:
                    return None, None # not sure about the spatial relation

            used_class_names = [0 for _ in class_names]
            
            Wrong_image = False
            
            #st()
            do_relabel = False
            new_rel_tuples = []
            for r_i, rel_tuple in enumerate(rel_tuples):
                new_rel_tuples.append(rel_tuple)
                if(rel_tuple[1] in ['is on top of', 'is on right of', 'is on left of', 'is above', 'is placed on']):
                    obj1 = rel_tuple[0]
                    obj2 = rel_tuple[2]
                    if(obj1 == 'pyramid'):
                        continue #skip pyramid as there are many false rejections for it
                    candidate_obj1_bboxes = []
                    candidate_obj2_bboxes = []
                    found_correct_relation = False
                    predicted_relations = []
                    for class_name in class_names:
                        if(obj1 in class_name):
                            candidate_obj1_bboxes.append(input_boxes[class_names.index(class_name)])
                        if(obj2 in class_name):
                            candidate_obj2_bboxes.append(input_boxes[class_names.index(class_name)])

                    if(len(candidate_obj1_bboxes) == 0 or len(candidate_obj2_bboxes) == 0):
                        if(args.verbose):
                            print("WRONG: Obj1 or Obj2 not detected in the image")
                        Wrong_image = True
                        #st()
                        break
                    else:
                        for obj1_bbox in candidate_obj1_bboxes:
                            for obj2_bbox in candidate_obj2_bboxes:
                                spatial_relation, distance = predict_spatial_relation(obj1_bbox, obj2_bbox)
                                if(spatial_relation is None):
                                    continue
                                else:
                                    predicted_relations.append(spatial_relation)
                                
                                    if(spatial_relation == rel_tuple[1] or (spatial_relation =='is above' and rel_tuple[1] in ['is on top of', 'is placed on','is above'])):
                                        found_correct_relation = True
                                        break
                    if(not found_correct_relation):


                        if(len(predicted_relations) == 0): #no chance to relabel
                            predicted_relation, distance = predict_spatial_relation(obj2_bbox, obj1_bbox) #try switching the order of the objects
                            if(predicted_relation is None): #if still no relation found, then it is a wrong image
                                if(args.verbose):
                                    print(f"WRONG: No correct relation found to relabel original relation: {rel_tuple}")
                                Wrong_image = True
                                #st()
                                break
                            else:
                                Wrong_image = True
                                if(args.verbose):
                                    print(f"DOING INVERSE RELABELING with predicted relations: {predicted_relation} for original relation: {rel_tuple}")
                                do_relabel = True
                                Wrong_image = False
                                new_rel_tuples[r_i] = (obj2, predicted_relation, obj1)
                                #break
                            
                        else: #do relabeling
                            Wrong_image = True
                            if(args.verbose):
                                print(f"DOING RELABELING with predicted relations: {predicted_relations} for original relation: {rel_tuple}")
                            #st()
                            for pred_rel in list(set(predicted_relations))[:1]:
                                #for temp_rel_tuple in rel_tuples:
                                    #st()
                                    #if((pred_rel == temp_rel_tuple[1] and temp_rel_tuple[0] == obj1 and temp_rel_tuple[2] == obj2) or (temp_rel_tuple[1] == inv_relation_dict[pred_rel] and temp_rel_tuple[0] == obj2 and temp_rel_tuple[2] == obj1)):
                                 #       continue #can't use this relation as it already exists in the rel_tuples
                                  #  else:
                                do_relabel = True
                                Wrong_image = False
                                new_rel_tuples[r_i] = (obj1, pred_rel, obj2)
                                break
                                #if(do_relabel):
                                 #   break

                                        
                        if(Wrong_image):
                            break
                        #Wrong_image = True
                        #break

                #check if object is detected -- instead of rejecting if the object is not detected, we can later remove the object from the prompt
                if(rel_tuple[0] not in class_names or rel_tuple[2] not in class_names):
                    if(args.verbose):
                        print(f"WRONG: Object not detected in the image: {rel_tuple}")
                    Wrong_image = True
                    break
                

            
            if(do_relabel):
                #st()
                caption = gen_prompt_str(new_rel_tuples, style, background, item_counts)
            else:
                caption = prompt
            if(args.verbose):
                print("updated prompt: ", caption)
                print("rel_tuples: ", rel_tuples)
                print("class_names: ", class_names)
                print("new rel_tuples: ", new_rel_tuples)
                print("Is wrong image: ", Wrong_image)
                image.save("temp_image.png")
            #st()
            if Wrong_image:
                # skip the image
                continue

            
                #st()



            
            #caption = prompt
            caption = caption.strip()
            prompt_pairs["prompt"] = caption
            if(do_relabel):
                prompt_pairs['rel_tuples'] = new_rel_tuples
            else:
                prompt_pairs['rel_tuples'] = rel_tuples
            prompt_pairs['detected_class_names'] = class_names
            prompt_pairs['style'] = style
            prompt_pairs['background'] = background
            prompt_pairs['num_rels'] = num_rels
            rels_counts_covered[num_rels] += 1
            #st()

            # save degug image
            image_np = image.copy()
            # Convert the PIL image to a format compatible with Matplotlib
            image_np = np.array(image_np)
            # Plot the image and add text
            plt.imshow(image_np)
            plt.axis('off')  # Hide axes
            # Define the text and position
            caption = prompt_pairs["prompt"]
            # Wrap the caption to fit within a certain character width
            wrapped_caption = "\n".join(textwrap.wrap(caption, width=40))

            # Define figure size based on the image size, adding extra space for caption
            fig_height = image_np.shape[0] + 40  # Adding space for caption
            fig, ax = plt.subplots(figsize=(image_np.shape[1] / 100, fig_height / 100))
            ax.imshow(image_np)
            ax.axis('off')

            # Add a black rectangle below the image for the caption
            plt.gcf().patch.set_facecolor('black')  # Set background color for the figure
            plt.text(
                0.5, -0.1, wrapped_caption,  # Position below the image
                fontsize=10, color="white", ha='center', va='top', transform=ax.transAxes
            )
            plt.savefig(f"{dataset_folder_name}/{key}_debug.jpg", bbox_inches='tight', pad_inches=0)
            plt.close()



            # save data
            image.save(f"{dataset_folder_name}/{key}.png")

            with open(f"{dataset_folder_name}/{key}.json", "w") as f:
                json.dump(prompt_pairs, f, indent=4)
            
            # save mask and bounding box for inpainting purpuse
            np.save(f"{dataset_folder_name}/{key}_mask.npy", masks)
            np.save(f"{dataset_folder_name}/{key}_box.npy", input_boxes)

            key += 1
            with open(os.path.join(dataset_folder_name, "rels_counts_covered.pkl"), "wb") as f:
                pickle.dump(rels_counts_covered, f)
            
            #image.save('temp.png')
            #print(prompt_pairs)
            
            #st()

        
        # release the memory
        torch.cuda.empty_cache()
        plt.close()
        gc.collect()
            
