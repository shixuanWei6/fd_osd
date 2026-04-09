from datasets import load_dataset

def get_math_data(start_offset=0, sample_size=2000):
    ds_math = load_dataset("gsm8k", "main", split="train", streaming=True)
    prompts = []
    for idx, item in enumerate(ds_math):
        if idx < start_offset: continue
        if len(prompts) >= sample_size: break
        prompts.append(f"Question: {item['question']}\nAnswer:")
    return prompts

def get_finance_data(start_offset=0, sample_size=2000):
    try:
        ds_fin = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    except Exception:
        ds_fin = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    prompts = []
    for idx, item in enumerate(ds_fin):
        if idx < start_offset: continue
        if len(prompts) >= sample_size: break
        instruction = item.get("instruction", "")
        context = item.get("input", "")
        combined = f"{instruction}\n{context}" if context else instruction
        prompts.append(f"Question: {combined.strip()}\nAnswer:")
    return prompts

def get_general_data(start_offset=0, sample_size=2000):
    ds_gen = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    prompts = []
    for idx, item in enumerate(ds_gen):
        if idx < start_offset: continue
        if len(prompts) >= sample_size: break
        prompts.append(f"Question: {item.get('instruction', '').strip()}\nAnswer:")
    return prompts

# ==========================================
# 🔌 THE EXTENSIBLE DOMAIN REGISTRY
# To add a new domain, simply add it to this dictionary!
# ==========================================
DOMAIN_REGISTRY = {
    "Math": get_math_data,
    "Finance": get_finance_data,
    "General": get_general_data
}

import torch
import os
import argparse
from tqdm import tqdm

from models.model_loader import load_models
from models.lora_adapter import setup_dual_lora
from engine.speculator import speculative_inference_with_feedback
from training.trainer import OnlineTrainer, set_trainable_adapters
from training.buffer import ExperienceReplayBuffer
from simulation.network import NetworkSimulator, RLHorizonController

def pretrain_domain(domain_name, data_loader_func, args, target_model, tokenizer, device):
    print(f"\n{'='*50}")
    print(f"🚀 Starting Offline Pre-training for Domain: [{domain_name}]")
    print(f"{'='*50}")
    
    # 1. Load Data
    prompts = data_loader_func(start_offset=2000, sample_size=args.warmup_steps)
    
    # 2. Setup Draft Model (Fresh initialization for this domain)
    draft_model, _, _ = load_models(args.draft_path, args.target_path, device)
    # We setup the dual architecture but ONLY train the local adapter
    draft_model = setup_dual_lora(draft_model, r=args.r, lora_alpha=args.lora_alpha)
    set_trainable_adapters(draft_model, trainable_names=["local_adapter"])
    
    # 3. Setup Optimizer & Trainer
    local_params = [p for n, p in draft_model.named_parameters() if "local_adapter" in n and p.requires_grad]
    optimizer = torch.optim.AdamW(local_params, lr=args.lr)
    buffer = ExperienceReplayBuffer(capacity=256)
    trainer = OnlineTrainer(model=draft_model, optimizer=optimizer, buffer=buffer, temp=2.0, alpha_kd=0.6)
    
    # 4. Dummy components for Speculator API
    network_sim = NetworkSimulator()
    rl_controller = RLHorizonController()
    
    # 5. Training Loop
    pbar = tqdm(prompts, desc=f"Pre-training {domain_name}")
    for step, prompt in enumerate(pbar):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        except Exception:
            continue
            
        _, alpha, feedback_buffer, _ = speculative_inference_with_feedback(
            draft_model, target_model, inputs.input_ids, 
            gamma=4, temperature=0.6, top_p=0.9,
            training_mode=True, rl_controller=rl_controller, network_sim=network_sim      
        )
        
        if len(feedback_buffer) > 0:
            buffer.add(feedback_buffer)
            if buffer.is_ready(batch_size=4):
                loss = trainer.train_step(batch_size=4)
                pbar.set_description(f"Pre-training {domain_name} | Loss: {loss:.4f} | Alpha: {alpha:.2f}")

        # =========================================================
        # [NEW] Checkpoint Saving Logic: Save every 100 data points
        # =========================================================
        if (step + 1) % 100 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"domain_{domain_name}", f"step_{step+1}")
            os.makedirs(ckpt_path, exist_ok=True)
            draft_model.save_pretrained(ckpt_path, selected_adapters=["local_adapter"])
            # Print a message without breaking the tqdm progress bar
            pbar.write(f"💾 Checkpoint saved: {ckpt_path}")

    # 6. Save ONLY the Local Adapter for this Domain (Final Save)
    save_path = os.path.join(args.save_dir, f"domain_{domain_name}")
    os.makedirs(save_path, exist_ok=True)
    draft_model.save_pretrained(save_path, selected_adapters=["local_adapter"])
    print(f"✅ Pre-trained Local LoRA for [{domain_name}] saved to {save_path}")
    
    # Free up memory before next domain
    del draft_model, optimizer, trainer, buffer
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_path", type=str, default="bigscience/bloomz-560m")
    parser.add_argument("--target_path", type=str, default="bigscience/bloomz-7b1")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default="./pretrained_local_loras")
    
    # [NEW] Add a specific directory for intermediate checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Cloud Oracle (Target Model) for Pre-training...")
    _, target_model, tokenizer = load_models(args.draft_path, args.target_path, device)
    
    # Iterate through our extensible registry
    for domain_name, data_func in DOMAIN_REGISTRY.items():
        pretrain_domain(domain_name, data_func, args, target_model, tokenizer, device)

if __name__ == "__main__":
    main()