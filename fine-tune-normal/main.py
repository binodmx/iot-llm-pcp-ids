"""
Fine-tune language models using unsloth.

Detailed description:
    # TODO: Change the description accordingly
    This module provides functionality to fine-tune language models using the 
    unsloth library. The dataset should be downloaded before running this script.

    ```python
    from datasets import load_dataset
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    dataset.save_to_disk("FineTome-100k-gemma-3-4b-it-train")
    ```

Usage example:
    >>> qsub jobscript.sh

Author:
    Binod Karunanayake

Created:
    2026-01-28
"""

from unsloth import FastModel
import torch
import sys
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only, get_chat_template, standardize_data_formats
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
import wandb


################################################################################
# Set initial configs
################################################################################
job_id = sys.argv[1].split(".")[0]
model_name = sys.argv[2]
dataset_name = sys.argv[3]
# TODO: Get from args
chat_template_name = "gemma-3"
dataset_type = "population"
sample_size = 800 if dataset_type.startswith("sample") else 8000
# FIXME: Set correct paths to datasets
if dataset_name == "all":
    dataset_df1 = pd.read_csv(f"/scratch/wd04/bk2508/datasets/x-iiotid/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df2 = pd.read_csv(f"/scratch/wd04/bk2508/datasets/cic-iot-2023/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df3 = pd.read_csv(f"/scratch/wd04/bk2508/datasets/wustl-iiot/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df1 = dataset_df1.sample(n=sample_size, random_state=42)
    dataset_df2 = dataset_df2.sample(n=sample_size, random_state=42)
    dataset_df3 = dataset_df3.sample(n=sample_size, random_state=42)
    dataset_df = pd.concat([dataset_df1, dataset_df2, dataset_df3], ignore_index=True).sample(frac=1).reset_index(drop=True)
else:
    dataset_df = pd.read_csv(f"/scratch/wd04/bk2508/datasets/{dataset_name}/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df = dataset_df.sample(n=sample_size, random_state=42)
train_dataset = Dataset.from_pandas(dataset_df)
output_dir = f"/scratch/wd04/bk2508/repositories/iot-llm-pcp-ids/ft-models/{dataset_name}-{model_name.split('/')[1].lower()}-{dataset_type}-normal"


################################################################################
# Initialize wandb
################################################################################
wandb.init(
    project="pcp-ids-fine-tune-normal",
    name=f"run-{job_id}",
    config={
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "output_dir": output_dir,
        # TODO: Add more configs as needed
    }
)


################################################################################
# Load model and tokenizer
################################################################################
model, tokenizer = FastModel.from_pretrained(
    model_name = f"/scratch/wd04/bk2508/models/{model_name}",
    local_files_only=True,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

################################################################################
# Prepare model for PEFT fine-tuning
################################################################################
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = chat_template_name,
)

################################################################################
# Load and preprocess dataset
################################################################################
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = load_from_disk(f"/scratch/wd04/bk2508/repositories/iot-llm-grpo/data/{dataset_name}-{model_name.split('/')[1].lower()}-train")
dataset = standardize_data_formats(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)


################################################################################
# Set up SFTTrainer
################################################################################
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

################################################################################
# Train the model
################################################################################
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
trainer_stats = trainer.train()
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
print(f"Fine-tuned model saved to {output_dir}")

wandb.finish()
