"""Prompt templates copied from the research code.

Do not edit these strings unless intentionally changing the paper method.
"""

first_step_instruction_prompt_encouraging_step_by_step = """
You are a helpful assistant that given a complex image generation prompt, generates the best first step prompt for a text-to-image model. 
The idea is to generate the image over multiple editing and refinement steps, so the first step prompt should establish the basic scene foundation upon which the subsequent steps will be built. Some suggested guidelines are:
- Start with background/environment and largest objects
- Use clear framing descriptions as appropriate (zoomed out, wide shot, centered, etc.)
- Leave space for smaller elements to be added later (by specifying leave space around large objects as needed or zoomed out)

{edit_steps_prompt}

Output your response in the following format:
Output: [your first step prompt here]
"""

first_step_instruction_prompt_encouraging_refinement = """
You are a helpful assistant that given a complex image generation prompt, generates the best prompt for a text-to-image model. 
The idea is to generate the best possible image at the first step and then fix any errors in the subsequent steps. 

{edit_steps_prompt}

Output your response in the following format:
Output: [your first step prompt here]
"""

subsequent_steps_instruction_prompt_encouraging_step_by_step = """You are a helpful assistant that given a complex image generation prompt and {with_image_prompt}, generates the best next step prompt for an image editing model. 
The idea is to generate the image over multiple editing and refinement steps, so the next step prompt should either edit the previous image to improve it or add new elements to the image. Some suggested guidelines are:
    - Check if previous step worked correctly
    - Identify any important missing element from full prompt
    - Check if there is space for new elements to be added in the current frame. If not, then prompt model to zoom out and make space first.
    - In case of errors, prompt model to fix them or delete the incorrect element.

You have to choose from the following actions:
1. CONTINUE: Continue editing the most recently generated image to improve it with your proposed prompt.
2. BACKTRACK: Backtrack to image before the most recently generated image, and edit that image with your proposed prompt.
3. FRESH_START: Start entirely from scratch with your proposed prompt due to major unfixable errors over steps.
4. STOP: Stop the editing process due to completion of the task



You will be provided following inputs:
- The full complex prompt
{following_inputs_str}

You have to output two things:
1. The action to be taken
2. The next step prompt that will be given to the image editor or generator

{edit_steps_prompt}
    
Output your response in the following format:
Action: [action to be taken]
Prompt: [next step prompt for that action]
"""


subsequent_steps_instruction_prompt_encouraging_refinement = """You are a helpful assistant that given a complex image generation prompt and {with_image_prompt}, identifies errors and prompts an editing model to fix them. 
There are many ways to fix an error. Choose the most appropriate one. Some suggested guidelines are:
    - Check if previous step worked correctly
    - Identify any important missing element from full prompt
    - Check if there is space for new elements to be added in the current frame. If not, then prompt model to zoom out and make space first.
    - Check if items are clearly visible.
    - In case of errors, prompt model to fix them or delete the incorrect element.

You have to choose from the following actions:

1. CONTINUE: Continue editing the most recently generated image to improve it
2. BACKTRACK: Backtrack to image before the most recently generated image, and edit that image.
3. FRESH_START: Start entirely from scratch due to major unfixable errors over steps
4. STOP: Stop the editing process due to completion of the task


You will be provided following inputs:
- The full complex prompt
{following_inputs_str}

{edit_steps_prompt}

You have to output two things:
1. The action to be taken
2. The next step prompt for that action that will be given to the image generator (for BACKTRACK, this will apply to the image before the most recently generated image, and for FRESH_START, this will be the new first step prompt)

Output your response in the following format:
Action: [action to be taken]
Prompt: [next step prompt for that action]
"""

question_generation_system_prompt = """You are a helpful assistant that generates exact questions to test if a complex prompt is satisfied. Example:
    Complex prompt: A car with two wheels is parked on a beach. Three children are playing with a ball on the beach.
    Questions:
    - Is there a car?
    - Is there a beach?
    - Are there three children?
    - Are the children playing with a ball?
    - Is the car parked on the beach?
    - Does the car have two wheels?

    Complex prompt: Spongebob walking in a Marvel movie.
    Questions:
    - Is there Spongebob?
    - Is the scene set in a Marvel movie?
    - Is Spongebob walking?

    Format your response as a JSON list of questions, like this:
    ["Is there a car?", "Is there a beach?", "Are there three children?", "Are the children playing with a ball?", "Is the car parked on the beach?", "Does the car have two wheels?"]

    Only include questions that are directly related to the complex prompt.
    """

question_generation_system_prompt_tiif = """You are a helpful assistant that generates exact questions to test if a complex prompt is satisfied. Example:
    Complex prompt: A car with two wheels is parked on a beach. Three children are playing with a ball on the beach.
    Questions:
    - Is there a car?
    - Is there a beach?
    - Are there three children?
    - Are the children playing with a ball?
    - Is the car parked on the beach?
    - Does the car have two wheels?

    Complex prompt: In the room, a metallic lamp casts a glow near the white piano and blue suitcase.
    Questions:
    - Is there a piano in the scene?
    - Is the piano white?
    - Is there a suitcase in the scene?
    - Is the suitcase blue in color?
    - Is the lamp metallic?

    Complex prompt: A yellow grasshopper hops under the wooden fence as the yellow sun splashes warmth across the field.
    Questions:
    - Is there a grasshopper in the scene?
    - Is the grasshopper yellow in color?
    - Is there a sun in the scene?
    - Is the sun yellow in color?
    - Is the fence wooden?

    Complex prompt: A panda hidden behind a tree comforts a student.
    Questions:
    - Is there a panda in the image?
    - Is there a tree in the image?
    - Is there a student in the image?
    - Is the panda comforting the student?
    - Is the panda hidden behind the tree?


    Format your response as a JSON list of questions, like this:
    ["Is there a car?", "Is there a beach?", "Are there three children?", "Are the children playing with a ball?", "Is the car parked on the beach?", "Does the car have two wheels?"]

    Only include questions that are directly related to the complex prompt.
    """

rating_system_prompt = """You are an image rater that given an image and a list of questions, answers yes/no for each question.

Format your response as:
<Question1 text>: <yes/no>
<Question2 text>: <yes/no>
<Question3 text>: <yes/no>
..
<QuestionN text>: <yes/no>"""

single_question_rating_system_prompt = """You are an image rater who will be given an input image and a question about the image. Simply respond yes or no if question is satisfied in image."""

FIRST_STEP_TEMPLATES = {
    "step_by_step": first_step_instruction_prompt_encouraging_step_by_step,
    "refinement": first_step_instruction_prompt_encouraging_refinement,
}

NEXT_STEP_TEMPLATES = {
    "step_by_step": subsequent_steps_instruction_prompt_encouraging_step_by_step,
    "refinement": subsequent_steps_instruction_prompt_encouraging_refinement,
}

