import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)
from src.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor
from src.flux.xflux_pipeline import XFluxSampler
from accelerate.utils import gather_object
import gc
from image_datasets.dataset import i2i_loader, i2i_inference_loader
from PIL import Image
import json
if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


def get_combined_img(final_images, step_wise_prompts):
    """
    Stack all images in final_images horizontally and return the combined image.
    
    Args:
        final_images: List of PIL images to combine
        
    Returns:
        PIL.Image: Combined image with all images stacked horizontally
    """
    if not final_images:
        return Image.new('RGB', (512, 512))
        
    # Get dimensions of images
    width = final_images[0].width
    height = final_images[0].height
    
    # Create a new image with width = sum of all image widths
    combined_width = (width+10) * len(final_images)
    # Add extra space below for captions
    caption_height = 50
    combined_img = Image.new('RGB', (combined_width, height + caption_height), color=(255, 255, 255))
    
    # Paste each image at the appropriate position
    for i, img in enumerate(final_images):
        combined_img.paste(img, (i * width, 0))
        
        # Add caption below the image
        if i < len(step_wise_prompts):
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_img)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            caption = step_wise_prompts[i]
            # Truncate caption if too long
            if len(caption) > 40:
                caption = caption[:37] + "..."
            
            #draw.text((i * (width+10), height + 10), caption, fill=(0, 0, 0), font=font)
        
    return combined_img


def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config

# image2image adapter model
class ImageAdapter(nn.Module):
    def __init__(self, in_channel=16):
        super().__init__()
        # (N, 16, X, 64) -> (N, 16, X, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, in_channel, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.cnn(x)

class ImageAdapter_img_in(nn.Module):
    def __init__(self, in_dim=64, out_dim=3072):
        super().__init__()
        
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

def main():
    args = OmegaConf.load(parse_args())
    conditioning_mode = str(getattr(args, "conditioning_mode", "channel_concat")).replace("-", "_")
    if conditioning_mode not in {"channel_concat", "token_concat"}:
        raise ValueError("conditioning_mode must be 'channel_concat' or 'token_concat'")
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
        if is_wandb_available() and args.report_to == "wandb":
            # Upload config to wandb
            wandb.init(project=args.tracker_project_name, config=OmegaConf.to_container(args, resolve=True))

    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)
    image_adapter = ImageAdapter()
    image_adapter_img_in = ImageAdapter_img_in()
    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]

    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            # print("setting LoRA Processor for", name)
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
              dim=3072, rank=args.rank
            )
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            # print("setting LoRA Processor for", name)
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
              dim=3072, rank=args.rank
            )
        else:
            lora_attn_procs[name] = attn_processor

    dit.set_attn_processor(lora_attn_procs)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(accelerator.device)
    image_adapter = image_adapter.to(torch.float32)
    image_adapter_img_in = image_adapter_img_in.to(torch.float32)
    dit.train()
    image_adapter.train()
    image_adapter_img_in.train()
    optimizer_cls = torch.optim.AdamW

    for n, param in dit.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        # else:
        #     print(n)
    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'parameters')

    image_adapter_params = [p for p in image_adapter.parameters() if p.requires_grad]
    image_adapter_img_in_params = [p for p in image_adapter_img_in.parameters() if p.requires_grad]
    dit_params = [p for p in dit.parameters() if p.requires_grad]
    trainable_params = image_adapter_params + image_adapter_img_in_params + dit_params

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print("loading train dataloader")
    train_dataloader = i2i_loader(**args.data_config)
    print("loaded train dataloader")

    print("loading val dataloader")
    val_dataloader = i2i_inference_loader(**args.data_config)
    print("loaded val dataloader")
    # randomly select args.num_val_images images from train_dataloader to validate
    val_batch = []
    #random_indices = random.sample(range(len(val_dataloader)), args.num_val_samples)
    for i, batch in enumerate(val_dataloader):
        #if i in random_indices:
        val_batch.append(batch)

        if len(val_batch) == args.num_val_samples:
            break

    print("loaded val batch")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    dit, optimizer, _, lr_scheduler, image_adapter, image_adapter_img_in  = accelerator.prepare(
        dit, optimizer, deepcopy(train_dataloader), lr_scheduler, image_adapter, image_adapter_img_in
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = get_schedule(
                999,
                (1024 // 8) * (1024 // 8) // 4,
                shift=True,
            )
    total_batch_size = int(args.train_batch_size) * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")

            global_step = int(path.split("-")[1])

            # clear the memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # image_adapter_state = torch.load(os.path.join(args.output_dir, path, "image_adapter.safetensors"))
            # image_adapter.load_state_dict(image_adapter_state)

            # image_adapter_img_in_state = torch.load(os.path.join(args.output_dir, path, "image_adapter_img_in.safetensors"))
            # image_adapter_img_in.load_state_dict(image_adapter_img_in_state)

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                img, cond_img, prompts, context_prompts, is_0th_step = batch
                apply_cond = not is_0th_step
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    if(args.use_global_prompt):
                        inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts, global_prompt=context_prompts)
                    else:
                        inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)  # last dim is 64, (N, x, 64)
                    
                    x_cond = vae.encode(cond_img.to(accelerator.device).to(torch.float32))
                
                x_cond = image_adapter(x_cond)
                x_cond = rearrange(x_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                x_cond = image_adapter_img_in(x_cond)

                bs = img.shape[0]
                t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), args.guidance_train, device=x_t.device, dtype=x_t.dtype)

                # Predict the noise residual and compute loss
                model_pred = dit(img=x_t.to(weight_dtype),
                                img_ids=inp['img_ids'].to(weight_dtype),
                                txt=inp['txt'].to(weight_dtype),
                                txt_ids=inp['txt_ids'].to(weight_dtype),
                                y=inp['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),
                                x_cond=x_cond.to(weight_dtype),
                                conditioning_mode=conditioning_mode,
                                apply_cond=apply_cond
                            )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(int(args.train_batch_size))).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    accelerator.save_state(save_path)
                    unwrapped_model_state = accelerator.unwrap_model(dit).state_dict()
                    unwrapped_image_adapter_state = accelerator.unwrap_model(image_adapter).state_dict()
                    unwrapped_image_adapter_img_in_state = accelerator.unwrap_model(image_adapter_img_in).state_dict()

                    # save checkpoint of the image adapter model
                    save_file(unwrapped_image_adapter_state, os.path.join(save_path, "image_adapter.safetensors"))
                    save_file(unwrapped_image_adapter_img_in_state, os.path.join(save_path, "image_adapter_img_in.safetensors"))

                    lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                    save_file(
                        lora_state_dict,
                        os.path.join(save_path, "lora.safetensors")
                    )

                    logger.info(f"Saved state to {save_path}")

                '''
                < generate validation images >
                '''
                if not args.disable_sampling and (global_step % args.sample_val_every == 0 or global_step == 1):
                    # release the memory
                    torch.cuda.empty_cache()
                    gc.collect()

                    with torch.no_grad():
                        for split_batch in range(0, len(val_batch), accelerator.num_processes): # loop over all val_batch
                            # val_batch: [[target_image, control_image, prompt], ...]
                            split = val_batch[split_batch:split_batch + accelerator.num_processes]

                            with accelerator.split_between_processes(split) as assigned_batch:
                                results=dict(file_idx=[], image_dir=[], prompt=[], gpu_no=[], seed=[], context_prompt=[])
                                
                                v_batch = assigned_batch[0]

                                initial_image, step_wise_prompts, context_prompts, baseline_image, full_info_dict = v_batch
                                #apply_cond = 

                                # start to generate validate images
                                width = initial_image.shape[-1]  # (C, H, W)
                                height = initial_image.shape[-2]

                                zeroth_image =  (127.5 * (initial_image.squeeze(0) + 1.0)).cpu().byte().numpy().transpose(1,2,0)
                                final_images = [Image.fromarray(zeroth_image)]
                                prev_img_pt = initial_image
                                # seed as device id
                                seed =  accelerator.local_process_index * 10
                                torch.manual_seed(seed)

                                for step_i in range(len(step_wise_prompts)):
                                    if(step_i == 0):
                                        apply_cond = False
                                    else:
                                        apply_cond = True
                                    if(prev_img_pt.ndim == 3):
                                        prev_img_pt = prev_img_pt.unsqueeze(0)
                                
                                    #print("Prev img pt shape", prev_img_pt.shape)
                                    prompt = step_wise_prompts[step_i]
                                    prompts = prompt

                                    if(step_i == 0):
                                        global_prompt = context_prompts

                                    x = get_noise(
                                        1, height, width, device=accelerator.device,
                                        dtype=torch.bfloat16, seed=seed
                                    )

                                    timesteps_inference = get_schedule(
                                        20,
                                        (width // 8) * (height // 8) // (16 * 16),
                                        shift=True,
                                    )

                                #torch.manual_seed(seed)
                                    if(args.use_global_prompt):
                                        inp = prepare(t5=t5, clip=clip, img=x, prompt=prompts, global_prompt=context_prompts)
                                    else:
                                        inp = prepare(t5=t5, clip=clip, img=x, prompt=prompts)
                                    neg_inp_cond = prepare(t5=t5, clip=clip, img=x, prompt="")

                                
                                    x_cond = vae.encode(prev_img_pt.to(accelerator.device).to(torch.float32))
                                    x_cond = image_adapter(x_cond)
                                    x_cond = rearrange(x_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                                    x_cond = image_adapter_img_in(x_cond)

                                    guidance = args.guidance_eval
                                    guidance_vec = torch.full((x.shape[0],), guidance, device=accelerator.device, dtype=x.dtype)

                                    i = 0
                                
                                    for t_curr, t_prev in zip(timesteps_inference[:-1], timesteps_inference[1:]):
                                        t_vec = torch.full((img.shape[0],), t_curr, dtype=x.dtype, device=x.device)

                                        # Predict the noise residual and compute loss
                                        model_pred = dit(
                                            img=inp['img'].to(weight_dtype),
                                            img_ids=inp['img_ids'].to(weight_dtype),
                                            txt=inp['txt'].to(weight_dtype),
                                            txt_ids=inp['txt_ids'].to(weight_dtype),
                                            y=inp['vec'].to(weight_dtype),
                                            timesteps=t_vec.to(weight_dtype),
                                            guidance=guidance_vec.to(weight_dtype),
                                            x_cond=x_cond.to(weight_dtype),
                                            conditioning_mode=conditioning_mode,
                                            apply_cond = apply_cond
                                        )

                                        timestep_to_start_cfg = 100   # start to use true guidance
                                        if i >= timestep_to_start_cfg:
                                            neg_pred = dit(
                                                img=inp['img'].to(weight_dtype),
                                                img_ids=inp['img_ids'].to(weight_dtype),
                                                txt=neg_inp_cond['txt'].to(weight_dtype),
                                                txt_ids=neg_inp_cond['txt_ids'].to(weight_dtype),
                                                y=neg_inp_cond['vec'].to(weight_dtype),
                                                timesteps=t_vec.to(weight_dtype),
                                                guidance=guidance_vec.to(weight_dtype),
                                                x_cond=x_cond.to(weight_dtype),
                                                conditioning_mode=conditioning_mode,
                                                apply_cond = apply_cond
                                            )
                                            true_gs = args.guidance_eval
                                            model_pred = neg_pred + true_gs * (model_pred - neg_pred)

                                        inp['img'] = inp['img'] + (t_prev - t_curr) * model_pred
                                        i += 1

                                # only visualize the last image in the batch
                                    x = unpack(inp['img'].float(), height, width)    # b (h w) (c ph pw) -> b c (h ph) (w pw)
                                    x = vae.decode(x)   # b c h w
                                
                                    x = x.clamp(-1, 1)
                                    prev_img_pt = x[-1]
                                    x = rearrange(x[-1], "c h w -> h w c")  # b c h w -> h w c
                                    output_img_pil = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                                    final_images.append(output_img_pil)

                                    def find_tensor_index(val_batch, v_batch):
                                        for i, batch in enumerate(val_batch):
                                            # first element if the tensor of target image
                                            # second element is the tensor of control image
                                            # third tuple of prompts
                                            if torch.equal(batch[0], v_batch[0]) \
                                                and torch.equal(batch[3], v_batch[3]) \
                                                and batch[2] == v_batch[2]:

                                                return i
                                        return -1

                                    idx_batch = find_tensor_index(val_batch, v_batch)
                                    val_output_dir = os.path.join(args.val_output_dir, f"interim_visualizations_multistep")
                                    val_output_dir = os.path.join(val_output_dir, f"val_step_{global_step}")
                                    dir = os.path.join(val_output_dir, f"val_{idx_batch}")
                                    os.makedirs(dir, exist_ok=True)
                                    step_img_path = os.path.join(dir, f"step_{step_i}_img.png")
                                    output_img_pil.save(step_img_path)

                                # torch to pil
                                img_path = os.path.join(dir, f"output_combined.png")
                                baseline_img_path = os.path.join(dir, f"baseline.png")
                                baseline_img = Image.fromarray((127.5 * (baseline_image.squeeze(0) + 1.0)).cpu().byte().numpy().transpose(1,2,0))
                                baseline_img.save(baseline_img_path)

                                combined_img = get_combined_img(final_images, step_wise_prompts)
                                combined_img.save(img_path)

                                
                                #dir = os.path.join(args.val_output_dir, f"val_{idx_batch}")

                                #os.makedirs(dir, exist_ok=True)
                                #img_path = os.path.join(dir, f"val_{idx_batch}+{global_step}.png")
                                #combined_img.save(img_path)
                                results['file_idx'].append(idx_batch)
                                results['image_dir'].append(img_path)
                                results['prompt'].append(prompts[-1])
                                results['context_prompt'].append(global_prompt[-1])
                                results['gpu_no'].append(accelerator.local_process_index)
                                results['seed'].append(seed)
                                prompt_info = {'step_wise_prompts': step_wise_prompts, 'global_prompt': global_prompt}
                                with open(os.path.join(dir, "prompt_info.json"), "w") as f:
                                    json.dump(prompt_info, f)

                                full_info_dict['num_objects'] = int(full_info_dict['num_objects'])
                                full_info_dict['num_relations'] = int(full_info_dict['num_relations'])
                                with open(os.path.join(dir, "full_info_dict.json"), "w") as f:
                                    json.dump(full_info_dict, f)

                                results=[ results ] # transform to list, otherwise gather_object() will not collect correctly
                                
                            # collect inference results from all the GPUs
                            results_gathered=gather_object(results)
                            vis_dict = {}
                            if accelerator.is_main_process:
                                file_idxs = []
                                image_dirs = []
                                prompts = []
                                seeds = []
                                context_prompts = []
                                for i in range(len(results_gathered)):
                                    file_idxs.extend(results_gathered[i]["file_idx"])
                                    image_dirs.extend(results_gathered[i]["image_dir"])
                                    prompts.extend(results_gathered[i]["prompt"])
                                    seeds.extend(results_gathered[i]["seed"])
                                    context_prompts.extend(results_gathered[i]["context_prompt"])
                                # upload the video and their corresponding prompt to wandb
                                if is_wandb_available():
                                    for i, file_idx in enumerate(file_idxs):
                                        image_dir = image_dirs[i]
                                        # Verify the image file exists before trying to load it
                                        if os.path.exists(image_dir) and os.path.isfile(image_dir):
                                            try:
                                                vis_dict[f"val_{file_idx}"] = wandb.Image(
                                                    image_dir, 
                                                    caption=prompts[i]+f" Context prompt: {context_prompts[i]}; seed: {seeds[i]}"
                                                )
                                            except Exception as e:
                                                logger.warning(f"Failed to load image {image_dir}: {e}")
                                        else:
                                            logger.warning(f"Image file not found: {image_dir}")

                                accelerator.log(vis_dict, step=global_step)
                            logger.info("Validation sample saved!")

                        

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
