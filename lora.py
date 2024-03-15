import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
import wandb, os
from tqdm import tqdm
from contextlib import nullcontext
from trl import SFTTrainer
from datetime import datetime

dataset = datasets.load_dataset("scott-persona/emotion_test_set",split="train")
train_test_split = dataset.train_test_split(test_size=0.05)
train_set = train_test_split["train"]
test_set = train_test_split["test"]

model_name_or_path = "mistralai/Mistral-7B-v0.1"
tokenizer_name_or_path = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16, load_in_4bit=True)

emotions = set()
for _, example in enumerate(dataset):
  emotions.add(example['emotion'])

num_added_tokens = tokenizer.add_tokens(list(emotions))
model.resize_token_embeddings(len(tokenizer))

def format_data(sample):
    instruction = "Classify the paragraph into one emotion."
    context = f"Paragraph: {sample['situation']}. Emotion: {sample['emotion']}"
    # join all the parts together
    prompt = "".join([instruction, context])
    return prompt

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_data(sample)}{tokenizer.eos_token}"
    return sample

def create_peft_config(model):

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules =["q_proj","k_proj","v_proj","o_proj"],
        modules_to_save=["embed_tokens"], #set embed_tokens as trainable too
    )

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

device = "cuda"
model, lora_config = create_peft_config(model)
train_set = train_set.map(template_dataset, remove_columns=list(train_set.features))

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-5,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 2,
}

project_name = "persona"
profiler = nullcontext()

wandb.login()

wandb_project = project_name
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

run_name = f"{project_name}-{datetime.now().strftime('%d-%H-%M')}"
training_params = TrainingArguments(
    output_dir="./" + run_name,
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=config['learning_rate'],
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    run_name=run_name
)
tokenizer.pad_token = tokenizer.eos_token
with profiler:
    # Create Trainer instance
    trainer = SFTTrainer(
    model = model,
    train_dataset = train_set,
    peft_config = lora_config,
    dataset_text_field = "text",
    max_seq_length = None,
    tokenizer = tokenizer,
    args = training_params,
    packing = False,
)
    # Start training
    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    trainer.model.save_pretrained("../finetuned")
    tokenizer.save_pretrained("../finetuned")
    
