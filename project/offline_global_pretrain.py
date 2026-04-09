import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def main():
    print("Loading Base Model and Tokenizer...")
    model_id = "bigscience/bloomz-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    # 1. UPGRADE: Expanded Global LoRA Capacity
    print("Initializing High-Capacity Global LoRA...")
    lora_config = LoraConfig(
        r=32,               # Increased from 8 to act as a strong foundation
        lora_alpha=64,      # Standard rule of thumb: alpha = 2 * r
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 2. UPGRADE: Reasoning-Heavy Dataset (OpenOrca)
    print("Loading OpenOrca Dataset...")
    # We load 100k high-quality examples instead of 50k simple ones
    dataset = load_dataset("Open-Orca/OpenOrca", split="train[:100000]")

    # 3. UPGRADE: Universal Prompt Formatting
    def format_orca(row):
        # We strip out complex formatting and just use Question/Answer.
        # MUST use this EXACT format in main_pre_exp.py!
        prompt = f"Question: {row['question']}\nAnswer: {row['response']}"
        return {"text": prompt}

    print("Formatting dataset...")
    dataset = dataset.map(format_orca, remove_columns=dataset.column_names)

    # 4. UPGRADE: Optimized Training Parameters
    training_args = SFTConfig(
        output_dir="./pretrained_global_lora",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,      # Effective batch size = 32
        learning_rate=2e-4,
        lr_scheduler_type="cosine",         # Smooth convergence
        warmup_ratio=0.05,                  # 5% of training spent warming up
        logging_steps=50,
        max_steps=2000,                     # Increased from 500 for deeper learning
        save_steps=500,
        fp16=True,                          # Use bf16=True if you have an Ampere GPU (RTX 3090/4090/A100)
        dataset_text_field="text",  
        max_length=512,                     # Increased to accommodate math/code reasoning
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting Optimized Offline Pre-training...")
    trainer.train()
    
    print("Saving highly capable Global LoRA...")
    trainer.model.save_pretrained("./pretrained_global_lora")
    print("Success! You can now load this into your Federated Simulation.")

if __name__ == "__main__":
    main()