import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img

import pickle
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False):
        images = []
        json_files = []
        key = '_'.join(img_dir.split('/')[-3:]) + "_cached.pkl"
        if(os.path.exists(os.path.join('./', key))):
            images = pickle.load(open(os.path.join('./', key), "rb"))
        else:
            for file in os.listdir(img_dir):
                if file.endswith('.json') and os.path.exists(os.path.join(img_dir, file.replace('.json', '.png'))):
                    #json_files.append(os.path.join(img_dir, file))
                    images.append(os.path.join(img_dir, file.replace('.json', '.png')))
            with open(os.path.join('./', key), "wb") as f:
                pickle.dump(images, f) 
        #for file in os.listdir(img_dir):
         #   if file.endswith('.json') and os.path.exists(os.path.join(img_dir, file.replace('.json', '.png'))):
                #json_files.append(os.path.join(img_dir, file))
          #      images.append(os.path.join(img_dir, file.replace('.json', '.png')))
           # elif file.endswith('.jpg') or file.endswith('.png'):
            #    images.append(os.path.join(img_dir, file))
        self.images = images
        #self.json_files = json_files
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
            img = image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            data = json.load(open(self.images[idx].replace('.png', '.json')))
            prompt = (
                data.get('prompt')
                or data.get('actual_step_by_step_prompt')
                or data.get('full_context_prompt')
            )
            if prompt is None:
                raise KeyError("JSON must contain prompt, actual_step_by_step_prompt, or full_context_prompt")

            #json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            #if self.caption_type == "json":
            #    prompt = json.load(open(json_path))['caption']
            #else:
             #   prompt = open(json_path).read()
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))

class CustomI2IDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False, eval_img_dir=None, filter_step0_images_with_max_k=None, used_cached_dataloader=True, skip_conditioning_on_0th_step=False, train_only_for_steps=None, use_addition_prompt_v1=False):
        self.condition_images = []
        key = '_'.join(img_dir.split('/')[-3:]) + "_cached_condition_images.pkl"
        print("Loading dataloader")
        if(filter_step0_images_with_max_k is not None):
            save_str = f"{key}_max_k_{filter_step0_images_with_max_k}"
        else:
            save_str = key

        if(train_only_for_steps in ['zero_and_first']):
            save_str = f"{save_str}_zero_and_first"
        elif(train_only_for_steps in ['zero_only']):
            save_str = f"{save_str}_zero_only"
        elif(train_only_for_steps in ['first_only']):
            save_str = f"{save_str}_first_only"
        else:
            train_only_for_steps = 'zero_and_first'
            save_str = f"{save_str}_zero_and_first"
        print("save str: ", save_str)
        print("filter_step0_images_with_max_k: ", filter_step0_images_with_max_k)
        

        print("train only for steps: ", train_only_for_steps)
        if(os.path.exists(os.path.join('./', save_str)) and used_cached_dataloader):
            print("Loading cached dataloader from: ", os.path.join('./', save_str))
            self.condition_images = pickle.load(open(os.path.join('./', save_str), "rb"))
             
        else:
            print("Loading new dataloader")
            for file in os.listdir(img_dir):
                if('_condition' in file and (file.endswith('.jpg') or file.endswith('.png')) and os.path.exists(os.path.join(img_dir, file.replace('_condition.', '.')))):
                    json_file = os.path.join(img_dir, file.replace('_condition.', '.').replace('.jpg', '.json').replace('.png', '.json'))
                    if(os.path.exists(json_file)):
                        json_data = json.load(open(json_file, "r"))
                        if(train_only_for_steps in ['zero_and_first']):
                            self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        elif(train_only_for_steps in ['zero_only']):
                            if(json_data['is_first_step']): #is_first_step corresponds to step0
                                self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        elif(train_only_for_steps in ['first_only']):
                            if(not json_data['is_first_step']):
                                self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        else:
                            self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        
                        #continue
                    else:
                        continue
                    """
                    if(filter_step0_images_with_max_k not in ['None', None]):
                        json_file = os.path.join(img_dir, file.replace('_condition.', '.').replace('.jpg', '.json').replace('.png', '.json'))
                        if(os.path.exists(json_file)):
                            json_data = json.load(open(json_file, "r"))
                            if(filter_step0_images_with_max_k is not None and len(json_data['original_items']) > filter_step0_images_with_max_k):
                                continue
                        else:
                            continue
                    """
                    #self.condition_images 
                        
                    self.condition_images.append(os.path.join(img_dir, file))
                
                #for file in os.listdir(eval_img_dir):
                    

            with open(os.path.join('./', save_str), "wb") as f:
                pickle.dump(self.condition_images, f)
        
            
        
        
        #self.condition_images.sort()
        #self.images = [image.replace('_condition.', '.') for image in self.condition_images]  # target image. eg. '1.jpg'
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.use_addition_prompt_v1 = use_addition_prompt_v1
        self.train_only_for_steps = train_only_for_steps
        self.skip_conditioning_on_0th_step = skip_conditioning_on_0th_step

    def __len__(self):
        return len(self.condition_images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.condition_images[idx].replace('_condition.', '.')).convert('RGB')
            condition_img = Image.open(self.condition_images[idx]).convert('RGB')
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
                    condition_img = crop_to_aspect_ratio(condition_img, ratio)
            img = image_resize(img, self.img_size)
            condition_img = image_resize(condition_img, self.img_size)

            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)  # (C, H, W)

            w, h = condition_img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            condition_img = condition_img.resize((new_w, new_h))
            condition_img = torch.from_numpy((np.array(condition_img) / 127.5) - 1)
            condition_img = condition_img.permute(2, 0, 1)

            json_path = self.condition_images[idx].replace('_condition.', '.').split('.')[0] + '.' + self.caption_type
            #if self.caption_type == "json":
            data = json.load(open(json_path))
            prompt = data['actual_step_by_step_prompt']
            if self.use_addition_prompt_v1 and not data.get('is_first_step', False):
                prompt = data.get('addition_prompt_v1', prompt)
            global_caption = data['full_context_prompt']

            
            is_0th_step = bool(data.get('is_first_step', False)) if self.skip_conditioning_on_0th_step else False
            return img, condition_img, prompt, global_caption, is_0th_step
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.condition_images) - 1))


class CustomI2IDataset_Eval(Dataset):
    def __init__(self, eval_img_dir, img_size=512, caption_type='json', random_ratio=False, img_dir=None, filter_step0_images_with_max_k=None, used_cached_dataloader=True, skip_conditioning_on_0th_step=False, train_only_for_steps=None, use_addition_prompt_v1=False):
        self.condition_images = []
        img_dir = eval_img_dir
        key = '_'.join(img_dir.split('/')[-3:]) + "_cached_eval_condition_images.pkl"
        print("Loading dataloader")
        if(filter_step0_images_with_max_k is not None):
            save_str = f"{key}_max_k_{filter_step0_images_with_max_k}"
        else:
            save_str = key

        if(train_only_for_steps in ['zero_and_first']):
            save_str = f"{save_str}_zero_and_first"
        elif(train_only_for_steps in ['zero_only']):
            save_str = f"{save_str}_zero_only"
        elif(train_only_for_steps in ['first_only']):
            save_str = f"{save_str}_first_only"
        else:
            train_only_for_steps = 'zero_and_first'
            save_str = f"{save_str}_zero_and_first"
        print("save str: ", save_str)
        print("filter_step0_images_with_max_k: ", filter_step0_images_with_max_k)
        

        print("train only for steps: ", train_only_for_steps)
        if(os.path.exists(os.path.join('./', save_str)) and used_cached_dataloader):
            print("Loading cached dataloader from: ", os.path.join('./', save_str))
            self.condition_images = pickle.load(open(os.path.join('./', save_str), "rb"))
             
        else:
            print("Loading new dataloader")
            for file in os.listdir(img_dir):
                if('_condition' in file and (file.endswith('.jpg') or file.endswith('.png')) and os.path.exists(os.path.join(img_dir, file.replace('_condition.', '.')))):
                    json_file = os.path.join(img_dir, file.replace('_condition.', '.').replace('.jpg', '.json').replace('.png', '.json'))
                    if(os.path.exists(json_file)):
                        json_data = json.load(open(json_file, "r"))
                        if(train_only_for_steps in ['zero_and_first']):
                            self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        elif(train_only_for_steps in ['zero_only']):
                            if(json_data['is_first_step']): #is_first_step corresponds to step0
                                self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        elif(train_only_for_steps in ['first_only']):
                            if(not json_data['is_first_step']):
                                self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        else:
                            self.condition_images.append(os.path.join(img_dir, file))
                            continue
                        
                        #continue
                    else:
                        continue
                    """
                    if(filter_step0_images_with_max_k not in ['None', None]):
                        json_file = os.path.join(img_dir, file.replace('_condition.', '.').replace('.jpg', '.json').replace('.png', '.json'))
                        if(os.path.exists(json_file)):
                            json_data = json.load(open(json_file, "r"))
                            if(filter_step0_images_with_max_k is not None and len(json_data['original_items']) > filter_step0_images_with_max_k):
                                continue
                        else:
                            continue
                    """
                    #self.condition_images 
                        
                    self.condition_images.append(os.path.join(img_dir, file))
                
                #for file in os.listdir(eval_img_dir):
                    

            with open(os.path.join('./', save_str), "wb") as f:
                pickle.dump(self.condition_images, f)
        
            
        
        
        #self.condition_images.sort()
        #self.images = [image.replace('_condition.', '.') for image in self.condition_images]  # target image. eg. '1.jpg'
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.use_addition_prompt_v1 = use_addition_prompt_v1
        self.train_only_for_steps = train_only_for_steps
        self.skip_conditioning_on_0th_step = skip_conditioning_on_0th_step

    def __len__(self):
        return len(self.condition_images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.condition_images[idx].replace('_condition.', '.')).convert('RGB')
            condition_img = Image.open(self.condition_images[idx]).convert('RGB')
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
                    condition_img = crop_to_aspect_ratio(condition_img, ratio)
            img = image_resize(img, self.img_size)
            condition_img = image_resize(condition_img, self.img_size)

            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)  # (C, H, W)

            w, h = condition_img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            condition_img = condition_img.resize((new_w, new_h))
            condition_img = torch.from_numpy((np.array(condition_img) / 127.5) - 1)
            condition_img = condition_img.permute(2, 0, 1)

            json_path = self.condition_images[idx].replace('_condition.', '.').split('.')[0] + '.' + self.caption_type
            #if self.caption_type == "json":
            data = json.load(open(json_path))
            prompt = data['actual_step_by_step_prompt']
            if self.use_addition_prompt_v1 and not data.get('is_first_step', False):
                prompt = data.get('addition_prompt_v1', prompt)
            global_caption = data['full_context_prompt']

            
            is_0th_step = bool(data.get('is_first_step', False)) if self.skip_conditioning_on_0th_step else False
            return img, condition_img, prompt, global_caption, is_0th_step
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.condition_images) - 1))




        
from copy import deepcopy
class Eval_I2IDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False, 
                 #num_gen_steps = None, #<int> number of steps to generate, if None then will be random
                 max_concepts_per_step = 2, #number of max concepts to generate at each step
                 dynamic_concepts_per_step = True, #if True, then the number of concepts to generate at each step will be dynamic
                 num_concepts_per_step = 1, #number of concepts to generate at each step
                 #img_dir=None, 
                 filter_step0_images_with_max_k=None,
                 num_max_broken_down_prompts=None,
                 use_fixed_prompts_order=False
                 ):
        #self.condition_images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '_condition.jpg' in i or '_condition.png' in i]   # eg. '1_condition.jpg'
        #self.condition_images.sort()
        #self.images = [image.replace('_condition.', '.') for image in self.condition_images]  # target image. eg. '1.jpg'
        
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.baseline_images = []
        self.image_info_dicts = []
        eval_img_dir = img_dir
        print(f"Loading from {eval_img_dir}")
        for img_path in os.listdir(eval_img_dir):
            if(img_path.endswith('.jpg') or img_path.endswith('.png')):
                self.baseline_images.append(os.path.join(eval_img_dir, img_path))
                image_info_dict_path = os.path.join(eval_img_dir, img_path.replace('baseline_', '').replace('.jpg', '.json').replace('.png', '.json'))
                self.image_info_dicts.append(json.load(open(image_info_dict_path)))

        #self.num_gen_steps = num_gen_steps
        self.max_concepts_per_step = max_concepts_per_step
        self.dynamic_concepts_per_step = dynamic_concepts_per_step
        self.num_concepts_per_step = num_concepts_per_step
        self.num_max_broken_down_prompts = num_max_broken_down_prompts
        self.use_fixed_prompts_order = use_fixed_prompts_order



    def __len__(self):
        return len(self.baseline_images)

    def __getitem__(self, idx):
        try:
            baseline_image = Image.open(self.baseline_images[idx]).convert('RGB')
            size = baseline_image.size
            initial_image = Image.new('RGB', size, (0,0,0))
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    baseline_image = crop_to_aspect_ratio(baseline_image, ratio)
                    initial_image = crop_to_aspect_ratio(initial_image, ratio)
            baseline_image = image_resize(baseline_image, self.img_size)
            initial_image = image_resize(initial_image, self.img_size)

            w, h = baseline_image.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            baseline_image = baseline_image.resize((new_w, new_h))
            baseline_image = torch.from_numpy((np.array(baseline_image) / 127.5) - 1)
            baseline_image = baseline_image.permute(2, 0, 1)  # (C, H, W)

            w, h = initial_image.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            initial_image = initial_image.resize((new_w, new_h))
            initial_image = torch.from_numpy((np.array(initial_image) / 127.5) - 1)
            initial_image = initial_image.permute(2, 0, 1)

            context_prompt = self.image_info_dicts[idx]['prompt']
            items = self.image_info_dicts[idx]['items']
            spatial_relation = self.image_info_dicts[idx]['spatial_relation']
            questions = []
            obj_counts = {}
            for item in items:
                if(item not in obj_counts): 
                    obj_counts[item] = 0
                obj_counts[item] += 1

            for obj, count in obj_counts.items():
                questions.append(f"Are there exactly {count} {obj}?")
            
            num_relations = 0
            for relation in spatial_relation:
                for obj_pair in spatial_relation[relation]:
                    obj1, obj2 = obj_pair
                    relation_str = relation.replace('is', '').strip()
                    questions.append(f"Is the {obj1} {relation_str} the {obj2}?")
                    num_relations += 1
            step_wise_prompts = []
            gpt_broken_down_prompts = deepcopy(self.image_info_dicts[idx]['gpt_prompt'])



            #rels_to_generate = []
            #for relation in spatial_relation:
             #   for obj_pair in spatial_relation[relation]:
              #      rels_to_generate.append((relation.replace('is', '').strip(), obj_pair))
            max_concepts_per_step = self.max_concepts_per_step
            if(max_concepts_per_step is None):
                max_concepts_per_step = len(gpt_broken_down_prompts)
            

            print("gpt_broken_down_prompts: ", gpt_broken_down_prompts)
            print("len(gpt_broken_down_prompts): ", len(gpt_broken_down_prompts))
            if(self.use_fixed_prompts_order):
                if(len(gpt_broken_down_prompts) ==1):
                    prompt = gpt_broken_down_prompts[0]
                    step_wise_prompts.append(prompt)
                elif(len(gpt_broken_down_prompts) ==2):
                    prompt = gpt_broken_down_prompts[0] #+ " " + gpt_broken_down_prompts[1]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[1]
                    step_wise_prompts.append(prompt)
                elif(len(gpt_broken_down_prompts) ==3):
                    prompt = gpt_broken_down_prompts[0]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[1] + " " + gpt_broken_down_prompts[2]
                    step_wise_prompts.append(prompt)

                elif(len(gpt_broken_down_prompts) ==4):
                    prompt = gpt_broken_down_prompts[0] + " " + gpt_broken_down_prompts[1]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[2] + " " + gpt_broken_down_prompts[3]
                    step_wise_prompts.append(prompt)
                elif(len(gpt_broken_down_prompts) ==5):
                    prompt = gpt_broken_down_prompts[0] + " " + gpt_broken_down_prompts[1]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[2] + " " + gpt_broken_down_prompts[3]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[4]
                    step_wise_prompts.append(prompt)
                elif(len(gpt_broken_down_prompts) ==6): 
                    prompt = gpt_broken_down_prompts[0] + " " + gpt_broken_down_prompts[1]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[2] + " " + gpt_broken_down_prompts[3] + " " + gpt_broken_down_prompts[4]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[5]
                    step_wise_prompts.append(prompt)
                elif(len(gpt_broken_down_prompts) ==7):
                    prompt = gpt_broken_down_prompts[0] + " " + gpt_broken_down_prompts[1] + " " + gpt_broken_down_prompts[2]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[3] + " " + gpt_broken_down_prompts[4] + " " + gpt_broken_down_prompts[5]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[6]
                    step_wise_prompts.append(prompt)
                else:
                    print("Warning: len(gpt_broken_down_prompts) > 7")
                    prompt = gpt_broken_down_prompts[0] + " " + gpt_broken_down_prompts[1]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[2] + " " + gpt_broken_down_prompts[3] + " " + gpt_broken_down_prompts[4]
                    step_wise_prompts.append(prompt)
                    prompt = gpt_broken_down_prompts[5] + " " + gpt_broken_down_prompts[6]
                    for i_ in range(7, len(gpt_broken_down_prompts)):
                        step_wise_prompts.append(gpt_broken_down_prompts[i_])

            elif(self.dynamic_concepts_per_step):
                while(len(gpt_broken_down_prompts) > 0):
                    num_items_to_generate = random.randint(1, max_concepts_per_step)
                    num_items_to_generate = min(num_items_to_generate, len(gpt_broken_down_prompts))
                    prompt = ""
                    for i_ in range(num_items_to_generate):
                        prompt += gpt_broken_down_prompts.pop(0) + " "
                    step_wise_prompts.append(prompt.strip())
            else:
                assert(self.num_concepts_per_step is not None)
                while(len(gpt_broken_down_prompts) > 0):
                    num_items_to_generate = min(self.num_concepts_per_step, len(gpt_broken_down_prompts))
                    prompt = ""
                    for i_ in range(num_items_to_generate):
                        prompt += gpt_broken_down_prompts.pop(0) + " "
                    step_wise_prompts.append(prompt.strip())

            full_info_dict = {
                'context_prompt': context_prompt,
                'step_wise_prompts': step_wise_prompts,
                'questions': questions,
                'num_objects': int(sum(obj_counts.values())),
                'num_relations': int(num_relations),
                'baseline_image_path': self.baseline_images[idx],
                'gpt_broken_down_prompts': gpt_broken_down_prompts,
                #'initial_image_path': self.baseline_images[idx]
            }                        
            return initial_image, step_wise_prompts, context_prompt, baseline_image, full_info_dict
            #json_path = self.image_info_dicts[idx]
            #if self.caption_type == "json":
             #   data = json.load(open(json_path))
              #  prompt = data['caption']
               # global_caption = data['global_caption']
            #else:
             #   prompt = open(json_path).read()
            #return img, condition_img, prompt, global_caption
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.baseline_images) - 1))

def single_image_loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

def loader(train_batch_size, num_workers, **args):
    return single_image_loader(train_batch_size=train_batch_size, num_workers=num_workers, **args)

def i2i_loader(train_batch_size, num_workers, **args):
    #dataset = NewCustomI2IDataset(img_dirs, **args)
    dataset = CustomI2IDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

def i2i_val_loader(train_batch_size, num_workers, **args):
    #dataset = NewCustomI2IDataset(img_dirs, **args)
    dataset = CustomI2IDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

def i2i_inference_loader(train_batch_size, num_workers, **args):
    #dataset = CustomI2IDataset_Eval(**args)
    dataset = Eval_I2IDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=False)

def i2i_eval_loader_during_training(train_batch_size, num_workers, **args):
    #dataset = NewCustomI2IDataset(img_dirs, **args)
    dataset = CustomI2IDataset_Eval(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=False)
