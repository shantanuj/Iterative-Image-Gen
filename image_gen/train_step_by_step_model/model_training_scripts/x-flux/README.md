# FLUX Step-by-Step Training

This directory contains the FLUX training code used for the training-based step-by-step image generation experiments. It is adapted from [XLabs-AI/x-flux](https://github.com/XLabs-AI/x-flux) and adds image-conditioned step-by-step LoRA training for examples produced by `synthetic_data_creation`.

The expected dataset format is the collated output from the synthetic data pipeline:

```text
collated_train_pairs/
  0_condition.png
  0.png
  0.json
  1_condition.png
  1.png
  1.json
  ...
```

Each `*_condition.png` is the current image state, each `{id}.png` is the target next state, and each JSON stores `actual_step_by_step_prompt` plus `full_context_prompt`.

## Setup

```bash
cd image_gen/train_step_by_step_model/model_training_scripts/x-flux
bash setup.sh
micromamba activate iter-refine-xflux
```

Log in to Hugging Face before launching if FLUX weights are not already cached:

```bash
huggingface-cli login
```

The loader uses the standard FLUX environment variables if you want to point to local weights:

```bash
export FLUX_DEV=/path/to/flux1-dev.safetensors
export AE=/path/to/ae.safetensors
```

## Conditioning Modes

Set `conditioning_mode` in the YAML config:

```yaml
conditioning_mode: "channel_concat"  # default
```

Supported values:

- `channel_concat`: default/recommended. The condition image is VAE-encoded, passed through a small convolutional adapter over latent channels, projected to FLUX hidden size, and added to the noisy image tokens.
- `token_concat`: experimental. The projected condition image tokens are appended to the FLUX image-token stream, then removed before the final prediction head.

The checkpoints from these two modes should be treated as separate model families.

## LoRA Step-by-Step Training

Default channel/projector conditioning:

```bash
bash scripts/launch_step_by_step_lora_training.sh \
  train_configs/step_by_step_lora_channel_concat.yaml
```

Token-concat conditioning:

```bash
bash scripts/launch_step_by_step_lora_training.sh \
  train_configs/step_by_step_lora_token_concat.yaml
```

The main step-by-step LoRA script is:

```bash
accelerate launch \
  --config_file accelerate_config_i2i.yaml \
  train_step_by_step_flux_lora.py \
  --config train_configs/step_by_step_lora_channel_concat.yaml
```

By default the release configs point to:

```text
../../synthetic_data_creation/outputs/synthetic_flux_qwen/collated_train_pairs
```

Change `data_config.img_dir`, `data_config.eval_img_dir`, `output_dir`, and `val_output_dir` for a full run.

## Full Fine-Tuning

The original full-finetuning path is included as:

```bash
bash scripts/launch_full_finetune_training.sh \
  train_configs/full_finetune_text_baseline.yaml
```

This path is a text-to-image FLUX finetune baseline. The step-by-step image-conditioned approach used in the paper is the LoRA path above.

## Important Config Fields

- `conditioning_mode`: `channel_concat` or `token_concat`.
- `use_global_prompt`: if true, uses the full context prompt for the CLIP/global embedding and the local step prompt for T5 text tokens.
- `data_config.train_only_for_steps`: `zero_and_first`, `zero_only`, or `first_only`.
- `single_blocks` / `double_blocks`: FLUX LoRA block selection.
- `rank`: LoRA rank.
- `resume_from_checkpoint`: set to `latest`, a checkpoint name, or `null`.
- `disable_sampling`: set true to skip validation image sampling during training.

## Outputs

Each checkpoint directory contains:

```text
checkpoint-N/
  lora.safetensors
  image_adapter.safetensors
  image_adapter_img_in.safetensors
```

The adapter files condition FLUX on the previous image state. Keep them with the LoRA checkpoint for inference/evaluation.

## Sample Generations

Example generations from a trained step-by-step FLUX LoRA model:

| Example 1 | Example 2 |
| --- | --- |
| <img src="outputs/sample_outputs/trained_step_by_step_generation_1.png" width="320"> | <img src="outputs/sample_outputs/trained_step_by_step_generation_2.png" width="320"> |

## Notes

The copied x-flux source is kept in `src/flux/` so the training scripts are self-contained. The `models_licence/LICENSE-FLUX1-dev` file is included for the FLUX.1-dev model license context.
