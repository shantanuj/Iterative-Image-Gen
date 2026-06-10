import json
import os
import sys
import shutil
import argparse

import torch
from t2v_metrics import VQAScore
import numpy as np
import random
from PIL import Image
from generate_flux_images_and_detect_objects import get_vocab, gen_prompt_str

ITEMS_DICT, RELATIONS_DICT, STYLE_LIST, BACKGROUND_LIST = get_vocab()
CANDIDATE_REPLACEMENT_RELATIONS = [rel for rel in RELATIONS_DICT.keys() if rel not in ['is above', 'is on left of', 'is on right of', 'is on top of']]

            
def get_vqa_score_model(vqa_score_model_type = 'clip-flant5-xxl'):
    print(f"Loading VQA score model: {vqa_score_model_type}")
    if(vqa_score_model_type == 'clip-flant5-xxl'):
        return VQAScore(model='clip-flant5-xxl')
    else:
        raise ValueError(f"Invalid VQA score model type: {vqa_score_model_type}") 

def get_vqa_score(img_file_path,
                  #prompt,
                  vqa_score_model, 
                  pregiven_questions_list,   
                  verbose=False):
    #print("Prompt: ", prompt)
    if(img_file_path is None):
        print(f"No image file path provided. Skipping VQA score calculation.")
        return -1, [], []
    #if(pregiven_questions_list is None):
     #   print(f"Input prompt is: {prompt}; using GPT to generate questions.")
      #  _, questions_list = get_gpt_step_scoring_questions(prompt, gpt_client, seed, gpt_model)
    #else:
    questions_list = pregiven_questions_list
    q_wise_scores = []
    for q in questions_list:
        if(verbose):
            print("Question: ", q)
        score = vqa_score_model(images=[img_file_path], texts=[q.strip()])
        if(verbose):
            print("Score: ", score)
        q_wise_scores.append(score)
    if(len(q_wise_scores) == 0):
        print(f"No questions were generated for image: {img_file_path}")
        return 0, [], []
    cuml_score = sum(q_wise_scores)/len(q_wise_scores)
    return cuml_score, questions_list, q_wise_scores

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")

def main():
    parser = argparse.ArgumentParser(description='Run rejection and relabelling with VQA score')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images and JSON files')
    parser.add_argument('--verbose', type=str2bool, required=False, help='Verbose output', default=True)
    parser.add_argument('--skip_if_item_not_detected', type=str2bool, required=False, help='Skip if item not detected', default=True)
    parser.add_argument('--use_spatial_relations_if_norelabel', type=str2bool, required=False, help='Use spatial relations if no relabel', default=True)
    args = parser.parse_args()
    
    vqa_score_model = get_vqa_score_model()
    input_dir = args.input_dir

    indv_score_thresh = 0.9
    cuml_score_thresh = 0.9

    obj_existence_thresh = random.choice([0.875, 0.86,0.9]) #set score threshold for object existence (let's set to slightly lower since already checked through object detector)
    obj_rel_thresh = random.choice([0.875, 0.86,0.9]) #set score threshold for object relations (should be moderately high since this is a bit hard)
    obj_style_thresh = 0.78 #set score threshold for object style (should be moderately high since this is a bit hard)
    obj_background_thresh = 0.8450 #set score threshold for object background (should be moderately high since this is a bit hard)

    obj_relabel_rel_thresh = random.choice([0.875, 0.86,0.9])
    obj_relabel_style_thresh = 0.9
    obj_relabel_background_thresh = 0.9

    thresh_num_before_and_after_rels = 2 #difference between number of relations before and after relabelling (maximum drop rels)
    thresh_num_before_and_after_items = 2 #difference between number of items before and after relabelling (maximum drop thresh items)

    processed_files = []
    scores = []
    num_total = 0
    num_rejected = 0
    k_wise_rejection_rates = {}
    verbose = args.verbose
    skip_if_item_not_detected = args.skip_if_item_not_detected
    use_spatial_relations_if_norelabel = args.use_spatial_relations_if_norelabel


    if(os.path.exists(f"{input_dir}/step2_processed_files.json")):
        with open(f"{input_dir}/step2_processed_files.json", 'r') as f:
            processed_files = json.load(f)

    if(os.path.exists(f"{input_dir}/step2_rejection_stats.json")):
        with open(f"{input_dir}/step2_rejection_stats.json", 'r') as f:
            k_wise_rejection_rates = json.load(f)

    print("Loaded VQA score model. now processing files in the input directory: ", input_dir)
    while(True):
        # Get all files in the input directory
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        # Filter out files that have already been processed
        new_files = [f for f in all_files if f not in processed_files]
        
        if not new_files:
            print("No new files to process. Waiting for more files...")
            import time
            time.sleep(60)  # Wait for 60 seconds before checking again
            continue
        
        print(f"Found {len(new_files)} new files to process.")
        #random.shuffle(new_files)
        
        for i, file in enumerate(new_files):
            file_id = file.split('.')[0]
            #st()
            img_file_path = os.path.join(input_dir, file)
            json_file_path = os.path.join(input_dir, f"{file_id}.json")
            
            if not os.path.exists(json_file_path):
                print(f"JSON file not found for {file}. Skipping.")
                processed_files.append(file)
                continue
                
            
            # Load the JSON file to get the prompt
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                prompt = data['prompt']
                rel_tuples = data['rel_tuples']
                num_orig_rels = len(rel_tuples)
                num_orig_items = len(data['items'])
                background = data['background']
                style = data['style']
                items = data['items']
                detected_class_names = data['detected_class_names']
                num_rels = data['num_rels']

            #num_rels_before = len(data['rel_tuples'])
            #num_items_before = len(data['item'])


            relabelled_data = {}
            relabelled_data['detected_class_names'] = detected_class_names

            
            new_items = []
            final_questions = []
            reject_image = False

            if(verbose):
                img = Image.open(img_file_path) 
                img.save("./temp_image_step2.png")

            #bbox_path = os.path.join(input_dir, f"{file_id}_bbox.json")
            mask_path = img_file_path.replace('.png', '_mask.npy')
            bounding_box_path = img_file_path.replace('.png', '_box.npy')
            all_mask = np.load(mask_path)
            all_boxes = np.load(bounding_box_path)
            #all_class_names = data['items']

            new_mask = []
            new_boxes = []




            for i, item in enumerate(items):
                question = f"Is there a {item}?"
                
                #st()
                score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, [question], verbose=verbose)
                #st()
                if(score >= obj_existence_thresh):
                    new_items.append(item)
                    final_questions.append(question)
                    new_mask.append(all_mask[i])
                    new_boxes.append(all_boxes[i])
                else:
                    if(verbose):
                        print(f"Object {item} does not exist in the image. Skipping.")
                    if(skip_if_item_not_detected):
                        reject_image = True
                        break

            new_mask = np.array(new_mask)
            new_boxes = np.array(new_boxes)
            if(reject_image):
                processed_files.append(file)
                num_total += 1
                num_rejected += 1
                #k_wise_rejection_rates[len(items) + len(rel_tuples)][0] += 1
                k = len(items) + len(rel_tuples)
                if(data['background'] is not None):
                    k += 1
                if(data['style'] is not None):
                    k += 1
                if(k not in k_wise_rejection_rates):
                    k_wise_rejection_rates[k] = [0, 0, 0]
                k_wise_rejection_rates[k][0] += 1
                #st()
                #if(not passes_threshold):
                k_wise_rejection_rates[k][1] += 1
                k_wise_rejection_rates[k][2] = 100*k_wise_rejection_rates[k][0]/k_wise_rejection_rates[k][1]
                continue
                
            relabelled_data['items'] = new_items
            #relabelled_data['mask'] = new_mask
            #relabelled_data['boxes'] = new_boxes

            
            new_style = None
            if style not in ['None', None, "", " "]:
                #st()
                question = f"Is the image {style}?"
                score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, [question], verbose=verbose)
                #st()
                if(score >= obj_style_thresh):
                    new_style = style
                    final_questions.append(question)
                
                else:
                    if(style in ['photorealistic', 'photo-realistic', 'photo realistic', 'photo realistic style', 'photo realistic style']):
                        question = f"Is the image a photo?"
                        score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, [question], verbose=verbose)
                        #st()
                        if(score >= 0.7):
                            new_style = style
                            final_questions.append(question)
                        else:
                            new_style = None
                    else:
                        new_style = None
            relabelled_data['style'] = new_style


            new_background = None

            if background not in ['None', None, "", " "]:
                #st()
                question = f"Is the background a {background}?"
                score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, [question], verbose=verbose)
                #st()
                if(score >= obj_background_thresh):
                    new_background = background
                    final_questions.append(question)
                
                else:
                    new_background = None
            relabelled_data['background'] = new_background


            new_rel_tuples = []
            #relabelled_data['rel']

            #st()
            
            for original_rel_tuple in rel_tuples:
                if(original_rel_tuple[0].strip() not in new_items or original_rel_tuple[2].strip() not in new_items):
                    if(verbose):
                        print(f"Object {original_rel_tuple[0]} or {original_rel_tuple[2]} does not exist in the image for the relation {original_rel_tuple}. Skipping.")
                    continue
                question = f"Is the {original_rel_tuple[0]} {original_rel_tuple[1].replace('is ', '').replace('Is ', '').strip()} the {original_rel_tuple[2]}?"
                #st()
                
                score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, [question], verbose=verbose)
                #st()
                if(score >= obj_rel_thresh): #or (original_rel_tuple[1] == 'is looking at' and score >= 0.82)):
                    new_rel_tuples.append(original_rel_tuple)
                    final_questions.append(question)
                else:
                    #Try to relabel the relation
                    candidate_relations = [rel for rel in CANDIDATE_REPLACEMENT_RELATIONS if rel.strip() not in [original_rel_tuple[1].strip()]]
                    questions_list = [f"Is the {original_rel_tuple[0]} {candidate_relation.replace('is ', '').replace('Is ', '').strip()} the {original_rel_tuple[2]}?" for candidate_relation in candidate_relations]
                    #st()
                    score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, questions_list, verbose=verbose)
                    #st()
                    candidate_relations_sorted, q_wise_scores_sorted = zip(*sorted(zip(candidate_relations, q_wise_scores), key=lambda x: x[1], reverse=True))
                    
                    found_relabel = False

                    #Try to relabel the relation with the highest score
                    for i, (candidate_relation, score) in enumerate(zip(candidate_relations_sorted, q_wise_scores_sorted)):
                        #if(candidate_relation in ['is on left of', 'is on right of', ''])
                        if(score >= obj_relabel_rel_thresh):
                            new_rel_tuples.append((original_rel_tuple[0], candidate_relation, original_rel_tuple[2]))
                            final_questions.append(questions_list[i])
                            if(verbose):
                                print(f"Relabeled relation {original_rel_tuple[1]} to {candidate_relation} with score {score}.")
                            found_relabel = True
                            break
                            
                    if(not found_relabel):
                        #Try to relabel the object source
                        obj_candidates_source = [obj for obj in new_items if obj != original_rel_tuple[0]]
                        questions_list = []
                        for obj_source in obj_candidates_source:
                            question = f"Is the {obj_source} {original_rel_tuple[1].replace('is ', '').replace('Is ', '').strip()} the {original_rel_tuple[2]}?"
                            questions_list.append(question)
                        #st()
                        score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, questions_list, verbose=verbose)
                        #st()
                        candidate_obj_source_sorted, q_wise_scores_sorted = zip(*sorted(zip(obj_candidates_source, q_wise_scores), key=lambda x: x[1], reverse=True))
                        for i, (obj_source, score) in enumerate(zip(candidate_obj_source_sorted, q_wise_scores_sorted)):
                            if(score >= obj_relabel_rel_thresh): #or (original_rel_tuple[1] == 'is looking at' and score >= 0.82)):
                                new_rel_tuples.append((obj_source, original_rel_tuple[1], original_rel_tuple[2]))
                                final_questions.append(questions_list[i])
                                if(verbose):
                                    print(f"Relabeled object source {original_rel_tuple[0]} to {obj_source} with score {score}.")
                                found_relabel = True
                                break

                        
                        
                    
                    if(not found_relabel):
                        #Try to relabel the object target
                        obj_candidates_target = [obj for obj in new_items if obj != original_rel_tuple[2]]
                        questions_list = []
                        for obj_target in obj_candidates_target:
                            question = f"Is the {original_rel_tuple[0]} {original_rel_tuple[1].replace('is ', '').replace('Is ', '').strip()} the {obj_target}?"
                            questions_list.append(question)

                        #st()
                        score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, questions_list, verbose=verbose)
                        #st()
                        candidate_obj_target_sorted, q_wise_scores_sorted = zip(*sorted(zip(obj_candidates_target, q_wise_scores), key=lambda x: x[1], reverse=True))
                        for i, (obj_target, score) in enumerate(zip(candidate_obj_target_sorted, q_wise_scores_sorted)):
                            if(score >= obj_relabel_rel_thresh): #or (original_rel_tuple[1] == 'is looking at' and score >= 0.82)):
                                new_rel_tuples.append((original_rel_tuple[0], original_rel_tuple[1], obj_target))
                                final_questions.append(questions_list[i])
                                #final_
                                if(verbose):
                                    print(f"Relabeled object target {original_rel_tuple[2]} to {obj_target} with score {score}.")
                                found_relabel = True
                                break

                    if(not found_relabel and use_spatial_relations_if_norelabel):
                        #Try to relabel the relation with a spatial relation with the highest score
                        candidate_relations = [rel for rel in ['is above', 'is on right of', 'is on left of'] if rel.strip() not in [original_rel_tuple[1].strip()]]
                        questions_list = [f"Is the {original_rel_tuple[0]} {candidate_relation.replace('is ', '').replace('Is ', '').strip()} the {original_rel_tuple[2]}?" for candidate_relation in candidate_relations]
                        #st()
                        score, questions, q_wise_scores = get_vqa_score(img_file_path, vqa_score_model, questions_list, verbose=verbose)
                        #st()
                        candidate_relations_sorted, q_wise_scores_sorted = zip(*sorted(zip(candidate_relations, q_wise_scores), key=lambda x: x[1], reverse=True))
                        
                        found_relabel = False

                        #Try to relabel the relation with the highest score
                        for i, (candidate_relation, score) in enumerate(zip(candidate_relations_sorted, q_wise_scores_sorted)):
                            #if(candidate_relation in ['is on left of', 'is on right of', ''])
                            if(score >= obj_relabel_rel_thresh):
                                new_rel_tuples.append((original_rel_tuple[0], candidate_relation, original_rel_tuple[2]))
                                final_questions.append(questions_list[i])
                                if(verbose):
                                    print(f"Relabeled relation {original_rel_tuple[1]} to {candidate_relation} with score {score}.")
                                found_relabel = True
                                break

                    if(not found_relabel):
                        print(f"No relabeling found for relation {original_rel_tuple}.")
                        #processed_files.append(file)
                        continue
                        
            relabelled_data['rel_tuples'] = new_rel_tuples
            relabelled_data['question'] = final_questions
            
            #relabelled_data['k'] = len(new_rel_tuples) + len()
            relabelled_data['num_rels'] = len(new_rel_tuples)

            #items_with_rels = []
            #for rel_tuple in new_rel_tuples:
                #   items_with_rels.append(rel_tuple[0])
                #  items_with_rels.append(rel_tuple[2])
            
            relabelled_data['k'] = len(new_rel_tuples) + len(items) 
            if(relabelled_data['background'] is not None):
                relabelled_data['k'] += 1
            if(relabelled_data['style'] is not None):
                relabelled_data['k'] += 1

            new_prompt = gen_prompt_str(relabelled_data['rel_tuples'], relabelled_data['style'], relabelled_data['background'], {item: 1 for item in relabelled_data['items']})

            relabelled_data['prompt'] = new_prompt
            
            #passes_threshold = 
            diff_num_rels = len(relabelled_data['rel_tuples']) - num_orig_rels
            diff_num_items = len(relabelled_data['items']) - num_orig_items
            if(diff_num_rels > 0 and diff_num_items > 0 and diff_num_rels <= thresh_num_before_and_after_rels and diff_num_items <= thresh_num_before_and_after_items):
                if((num_orig_rels < 4 and diff_num_rels == 2) or (num_orig_items < 4 and diff_num_items == 2)):
                    if(verbose):
                        print(f"Does not pass threshold as: Diff num rels: {diff_num_rels}, diff num items: {diff_num_items}")
                    passes_threshold = False
                else:
                    if(verbose):
                        print("Passes threshold as: Diff num rels: {diff_num_rels}, diff num items: {diff_num_items}")
                    passes_threshold = True
            else:
                if(diff_num_rels > 0 and diff_num_items > 0):
                    if(verbose):
                        print(f"Does not pass threshold as: Diff num rels: {diff_num_rels}, diff num items: {diff_num_items}")
                    
                    passes_threshold = False
                else:
                    passes_threshold = True


            if(len(relabelled_data['rel_tuples']) > 2 and (num_rejected / (num_total + 1e-5) < 0.2) and len(relabelled_data['items']) <5 and len(relabelled_data['rel_tuples']) < 5): #only do this when images that have been rejected less than 50% of the time and have less than 5 items and less than 5 relations

                
                is_all_spatial_relations= True
                for rel_tuple in relabelled_data['rel_tuples']:
                    if(rel_tuple[1] not in ['is above', 'is on right of', 'is on left of', 'is on top of']):
                        is_all_spatial_relations = False
                        break
                if(is_all_spatial_relations):
                    passes_threshold = False
                    if(verbose):
                        print(f"Image has all spatial relations. Skipping.")


            #st()

            if(passes_threshold):
                if(verbose):
                    print(f"Passes threshold. Saving relabelled data for {file_id}.")
                with open(os.path.join(input_dir, f"{file_id}_relabelled.json"), 'w') as f:
                    json.dump(relabelled_data, f)
                with open(os.path.join(input_dir, f"{file_id}_relabelled_box.npy"), 'wb') as f:
                    np.save(f, new_boxes)
                with open(os.path.join(input_dir, f"{file_id}_relabelled_mask.npy"), 'wb') as f:
                    np.save(f, new_mask)
                
                #print("Relabelled data", relabelled_data)
                
            processed_files.append(file)
            #st()
            if(not passes_threshold):
                num_rejected += 1
                if(relabelled_data['k'] not in k_wise_rejection_rates):
                    k_wise_rejection_rates[relabelled_data['k']] = [0, 0, 0]
                k_wise_rejection_rates[relabelled_data['k']][0] += 1
            num_total += 1
            #st()
            #if(not passes_threshold):
            if(relabelled_data['k'] not in k_wise_rejection_rates):
                k_wise_rejection_rates[relabelled_data['k']] = [0, 0, 0]
            k_wise_rejection_rates[relabelled_data['k']][1] += 1
            k_wise_rejection_rates[relabelled_data['k']][2] = 100*k_wise_rejection_rates[relabelled_data['k']][0]/k_wise_rejection_rates[relabelled_data['k']][1]

            #st()

            if(i%100 == 0):
                print(f"Processed {i} files")
                

                #st()    
                print(f"Rejected {num_rejected} out of {num_total} files which is {100*num_rejected/num_total:.3f}%")
                print(f"K-wise rejection rates: {k_wise_rejection_rates}")
                
                with open(f"{input_dir}/step2_rejection_stats.json", 'w') as f:
                    json.dump({
                        "num_rejected": num_rejected, 
                        "num_total": num_total, 
                        "percent_rejected": 100*num_rejected/num_total if num_total > 0 else 0, 
                        "k_wise_rejection_rates": k_wise_rejection_rates
                    }, f)
                processed_files = list(set(processed_files))
                with open(f"{input_dir}/step2_processed_files.json", 'w') as f:
                    json.dump(processed_files, f)

if __name__ == "__main__":
    main()
