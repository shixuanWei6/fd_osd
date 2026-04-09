import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import random
import argparse
import os 

from models.model_loader import load_models
from models.lora_adapter import setup_single_lora, setup_dual_lora 
from engine.speculator import speculative_inference_with_feedback
from training.trainer import OnlineTrainer, set_trainable_adapters
from training.buffer import ExperienceReplayBuffer 

from simulation.network import NetworkSimulator, RLHorizonController
from utils.data_registry import DOMAIN_REGISTRY
class ClientNode:
    def __init__(self, client_id, domain, draft_model, optimizer, buffer, trainer):
        self.client_id = client_id
        self.domain = domain
        self.model = draft_model
        self.optimizer = optimizer
        self.buffer = buffer
        self.trainer = trainer
        
        # Tracking metrics per client
        self.history_alpha = []
        self.history_loss = []
        self.history_loss_steps = [] # [Added] Track the step for X-axis alignment
        # [ADDED] Track RL specific metrics
        self.history_k = []
        self.history_rtt = []

def get_multi_client_streams(tokenizer, sample_size=2000):
    """Generates distinct data distributions for 3 different clients."""
    print(f"Loading datasets for distinct clients (Sample Size: {sample_size})...")
    
    # Client 0: Math (GSM8k)
    ds_math = load_dataset("gsm8k", "main", split="train", streaming=True)
    math_prompts = []
    for item in ds_math:
        if len(math_prompts) >= sample_size: break
        # MATCHES PRE-TRAINING
        math_prompts.append(f"Question: {item['question']}\nAnswer:")
        
    # Client 1: Finance (Alpaca-Finance)
    try:
        ds_fin = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    except:
        ds_fin = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    fin_prompts = []
    for item in ds_fin:
        if len(fin_prompts) >= sample_size: break
        
        # Combine instruction and input, but format as Question/Answer
        instruction = item.get('instruction', '')
        context = item.get('input', '')
        combined_text = f"{instruction}\n{context}" if context else instruction
        
        # MATCHES PRE-TRAINING
        fin_prompts.append(f"Question: {combined_text.strip()}\nAnswer:")

    # Client 2: General/Code (Alpaca-Cleaned)
    ds_gen = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    gen_prompts = []
    for item in ds_gen:
        if len(gen_prompts) >= sample_size: break
        
        instruction = item.get('instruction', '')
        
        # MATCHES PRE-TRAINING
        gen_prompts.append(f"Question: {instruction.strip()}\nAnswer:")

    return {
        "Math": math_prompts,
        "Finance": fin_prompts,
        "General": gen_prompts
    }



def get_multi_client_streams_after(tokenizer, start_offset=0, sample_size=2000):
    """
    Generates distinct data distributions for 3 different clients, but starts
    collecting only AFTER skipping `start_offset` samples from each streaming
    dataset. This ensures warmup data does not overlap with the online data.
    """
    print(
        f"Loading datasets for warmup clients (Skip: {start_offset}, Take: {sample_size})..."
    )

    # Client 0: Math (GSM8k)
    ds_math = load_dataset("gsm8k", "main", split="train", streaming=True)
    math_prompts = []
    idx = 0
    for item in ds_math:
        if idx < start_offset:
            idx += 1
            continue
        if len(math_prompts) >= sample_size:
            break
        math_prompts.append(f"Question: {item['question']}\nAnswer:")
        idx += 1

    # Client 1: Finance (Alpaca-Finance)
    try:
        ds_fin = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    except Exception:
        ds_fin = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    fin_prompts = []
    idx = 0
    for item in ds_fin:
        if idx < start_offset:
            idx += 1
            continue
        if len(fin_prompts) >= sample_size:
            break

        instruction = item.get("instruction", "")
        context = item.get("input", "")
        combined_text = f"{instruction}\n{context}" if context else instruction
        fin_prompts.append(f"Question: {combined_text.strip()}\nAnswer:")
        idx += 1

    # Client 2: General/Code (Alpaca-Cleaned)
    ds_gen = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    gen_prompts = []
    idx = 0
    for item in ds_gen:
        if idx < start_offset:
            idx += 1
            continue
        if len(gen_prompts) >= sample_size:
            break

        instruction = item.get("instruction", "")
        gen_prompts.append(f"Question: {instruction.strip()}\nAnswer:")
        idx += 1

    return {
        "Math": math_prompts,
        "Finance": fin_prompts,
        "General": gen_prompts,
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(data, window_size=20):
    if data is None or (isinstance(data, np.ndarray) and data.size == 0) or len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ==========================================
# 实时画图工具函数
# ==========================================
def save_plots(clients, sync_interval, exp_suffix):
    """
    Plots Alpha and Loss curves for multiple federated clients.
    """
    smooth_window = 15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # ==========================================
    # 1. Alpha Plot (Acceptance Rate)
    # ==========================================
    plt.figure(figsize=(12, 6))
    max_steps = 0
    
    for i, client in enumerate(clients):
        alpha_data = client.history_alpha
        if not alpha_data: continue
        
        max_steps = max(max_steps, len(alpha_data))
        smoothed_alpha = moving_average(alpha_data, window_size=smooth_window)
        
        plt.plot(
            range(len(smoothed_alpha)), 
            smoothed_alpha, 
            color=colors[i % len(colors)], 
            linewidth=2, 
            label=f'Client {client.client_id} ({client.domain})'
        )
        
    # Draw vertical lines for Federated Sync events
    if max_steps > 0 and sync_interval > 0:
        for sync_pt in range(sync_interval, max_steps, sync_interval):
            plt.axvline(x=sync_pt, color='gray', linestyle='--', alpha=0.5)
        # Add a single dummy line for the legend
        plt.axvline(x=-100, color='gray', linestyle='--', alpha=0.5, label='FedAvg Sync')

    plt.xlim(0, max_steps)
    plt.xlabel('Inference Steps (Requests)', fontsize=12)
    plt.ylabel('Mean Acceptance Rate (Alpha)', fontsize=12)
    plt.title('Multi-Client Speculative Decoding: Federated Online Learning', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"result_alpha_multiclient_{exp_suffix}.png", dpi=300)
    plt.close()

    # ==========================================
    # 2. Loss Plot (Training Dynamics)
    # ==========================================
    plt.figure(figsize=(12, 6))
    has_loss_data = False
    
    for i, client in enumerate(clients):
        loss_data = client.history_loss
        loss_steps = client.history_loss_steps
        
        if not loss_data or not loss_steps: continue
        has_loss_data = True
        
        smooth_loss_window = max(5, len(loss_data) // 20)
        smoothed_loss = moving_average(loss_data, window_size=smooth_loss_window)
        
        # Ensure x and y dimensions match after smoothing
        loss_x_axis = loss_steps[:len(smoothed_loss)]
        
        plt.plot(
            loss_x_axis, 
            smoothed_loss, 
            color=colors[i % len(colors)], 
            linewidth=2.5, 
            alpha=0.8,
            label=f'Client {client.client_id} ({client.domain})'
        )

    if has_loss_data:
        plt.xlabel('Inference Steps', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Federated Online Adaptation Training Dynamics', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"result_loss_multiclient_{exp_suffix}.png", dpi=300)
    # 3. RL Dynamics Plot (K vs RTT)
    # ==========================================
    # We will plot the dynamics for Client 0 as a representative example
    client0 = clients[0]
    if client0.history_k and client0.history_rtt:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot Network RTT on the Left Y-Axis (Red)
        color1 = 'tab:red'
        ax1.set_xlabel('Inference Steps', fontsize=12)
        ax1.set_ylabel('Network RTT (ms)', color=color1, fontsize=12)
        
        # Smooth RTT to see the network "weather" patterns clearly
        smooth_rtt = moving_average(client0.history_rtt, window_size=10)
        ax1.plot(range(len(smooth_rtt)), smooth_rtt, color=color1, alpha=0.5, label='Network RTT')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Plot Chosen K on the Right Y-Axis (Blue)
        ax2 = ax1.twinx()  
        color2 = 'tab:blue'
        ax2.set_ylabel('Chosen Step Size (K)', color=color2, fontsize=12)
        
        # Smooth K to observe the RL agent's policy trend
        smooth_k = moving_average(client0.history_k, window_size=10)
        ax2.plot(range(len(smooth_k)), smooth_k, color=color2, linewidth=2.5, label='RL Chosen K')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f'RL Agent Adaptation: Step Size (K) vs Network Latency (RTT) - {client0.domain}', fontsize=14)
        fig.tight_layout()
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.savefig(f"result_rl_dynamics_{exp_suffix}.png", dpi=300)
        plt.close()


from peft import set_peft_model_state_dict

class DomainRouter:
    """
    Dynamically routes clients to their domain-specific pre-trained Local LoRA.
    Highly extensible: Automatically works for any new domain saved by offline_pretrain.py.
    """
    def __init__(self, base_dir="./pretrained_local_loras"):
        self.base_dir = base_dir
        
    def route_and_load(self, client_id, domain, model):
        model_path = os.path.join(self.base_dir, f"domain_{domain}")
        safe_path = os.path.join(model_path, "adapter_model.safetensors")
        bin_path = os.path.join(model_path, "adapter_model.bin")
        
        if os.path.exists(safe_path) or os.path.exists(bin_path):
            print(f"[Router] 🧭 Routing Client {client_id} (Data: {domain}) -> {model_path}")
            if os.path.exists(safe_path):
                from safetensors.torch import load_file
                state_dict = load_file(safe_path)
            else:
                state_dict = torch.load(bin_path, map_location="cpu")
            
            # Safely inject weights specifically into the local_adapter branch
            set_peft_model_state_dict(model, state_dict, adapter_name="local_adapter")
            print(f"[Router] ✅ Domain '{domain}' weights successfully mounted!")
        else:
            print(f"[Router] ⚠️ Warning: No pre-trained weights found for '{domain}'. Client {client_id} will start from scratch!")

# ==========================================
# 运行实验
# ==========================================

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--r", type=int, default=32, help="LoRA rank r") 
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha ") 
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only 50 samples per domain)")
    args = parser.parse_args() 
    
    r_value = args.r
    lora_alpha = args.lora_alpha
    exp_suffix = os.environ.get("EXP_NAME", f"r{r_value}_lr{args.lr}_multiclient")
    if args.debug:
        exp_suffix += "_debug"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    draft_path = "bigscience/bloomz-560m" 
    target_path = "bigscience/bloomz-7b1"
    
    SAMPLE_PER_DOMAIN = 50 if args.debug else 2000
    set_seed(42)
    
    # ========================================================
    # FIX 2 & 3: Load Target Model and Tokenizer ONCE globally
    # ========================================================
    print(f"\n=== Loading Cloud Oracle (Target Model) ===")
    tokenizer = AutoTokenizer.from_pretrained(draft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path,
        torch_dtype=torch.bfloat16, # Keeping our previous bfloat16 fix!
        trust_remote_code=True,
        device_map="auto"
    )
    for param in target_model.parameters():
        param.requires_grad = False
    target_model.eval()

    # Dynamically Load Client Streams
    client_streams = {}
    for domain_name, data_func in DOMAIN_REGISTRY.items():
        client_streams[domain_name] = data_func(start_offset=500, sample_size=SAMPLE_PER_DOMAIN)
    
    clients = []
    router = DomainRouter(base_dir="./pretrained_local_loras")
    
    print(f"\n=== Setting up {len(client_streams)} Clients ===")

    for client_id, (domain, prompts) in enumerate(client_streams.items()):
        print(f"\n--- Initializing Client {client_id} ({domain}) ---")
        
        # Load only the small draft model for each client to prevent OOM
        c_draft = AutoModelForCausalLM.from_pretrained(
            draft_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        c_draft = setup_dual_lora(c_draft, r=r_value, lora_alpha=lora_alpha)
        
        # Route and load weights before making them trainable
        router.route_and_load(client_id, domain, c_draft)
        
        set_trainable_adapters(c_draft, trainable_names=["local_adapter", "global_adapter"])
        
        # ========================================================
        # FIX 1: Provide the missing Optimizer, Buffer, and Trainer
        # ========================================================
        local_params = [p for n, p in c_draft.named_parameters() if p.requires_grad]
        c_optimizer = torch.optim.AdamW(local_params, lr=args.lr)
        c_buffer = ExperienceReplayBuffer(capacity=256)
        c_trainer = OnlineTrainer(model=c_draft, optimizer=c_optimizer, buffer=c_buffer, temp=2.0, alpha_kd=0.6)
        
        clients.append(ClientNode(client_id, domain, c_draft, c_optimizer, c_buffer, c_trainer))

    # Initialize Server
    try:
        from simulation.federated import MultiClientFederatedServer
        fed_server = MultiClientFederatedServer(template_model=clients[0].model)
    except ImportError:
        print("Error: simulation/federated.py not found. Please ensure the MultiClientFederatedServer is created.")
        return
    
    network_sim = NetworkSimulator()
    rl_controller = RLHorizonController()

    SYNC_INTERVAL = 200
    UPDATE_INTERVAL = 8
    BATCH_SIZE = 4
    
    # FIX 4: The Phase 1 Warmup block has been deleted because you already pre-trained offline!
    
    # ----------------------------------------------------
    # Main Parallel Multi-Client Simulation (Online Learning)
    # ----------------------------------------------------
    print("\n>>> Starting Parallel Multi-Client Simulation")
    
    zipped_prompts = zip(
        client_streams["Math"],
        client_streams["Finance"],
        client_streams["General"],
    )
    
    pbar = tqdm(zipped_prompts, total=SAMPLE_PER_DOMAIN)
    
    for step, prompt_tuple in enumerate(pbar):
        step_alphas = []
        step_losses = []
        
        # A. Edge Inference & Local Training
        for client, prompt in zip(clients, prompt_tuple):
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            except Exception:
                continue
                
            input_ids = inputs.input_ids
            
            _, alpha, feedback_buffer, rl_stats = speculative_inference_with_feedback(
                client.model, target_model, input_ids, 
                gamma=4, 
                temperature=0.6,   
                top_p=0.9,         
                training_mode=True,
                rl_controller=rl_controller, 
                network_sim=network_sim      
            )
            
            client.history_alpha.append(alpha)
            step_alphas.append(alpha) 
            
            client.history_k.append(rl_stats["avg_k"])
            client.history_rtt.append(rl_stats["avg_rtt"])
            
            if len(feedback_buffer) > 0:
                client.buffer.add(feedback_buffer)
                
                if step > 0 and step % UPDATE_INTERVAL == 0 and client.buffer.is_ready(BATCH_SIZE):
                    loss = client.trainer.train_step(batch_size=BATCH_SIZE)
                    client.history_loss.append(loss)
                    client.history_loss_steps.append(step) 
                    step_losses.append(loss) 
        
        desc = f"Step {step}"
        if step_alphas:
            desc += f" | Avg Alpha: {sum(step_alphas)/len(step_alphas):.2f}"
        if step_losses:
            desc += f" | Avg Loss: {sum(step_losses)/len(step_losses):.4f}"
        pbar.set_description(desc)
        
        # B. Cloud Federated Sync
        if step > 0 and step % SYNC_INTERVAL == 0:
            pbar.write(f"\n--- Step {step}: Initiating Federated Sync with Cloud ---")
            fed_server.federated_averaging(clients)
            fed_server.broadcast(clients)
            
        # C. Visualization & Logging
        if step > 0 and step % 50 == 0:
            save_plots(clients, sync_interval=SYNC_INTERVAL, exp_suffix=exp_suffix)

    print(f"Experiment finished. Logs and images saved with suffix '{exp_suffix}'.")

    # ==========================================
    # Quantitative Results
    # ==========================================
    print("\n" + "="*70)
    print("📊 MULTI-CLIENT QUANTITATIVE RESULTS SUMMARY")
    print("="*70)
    
    def calc_speedup(alpha, k=4, c=0.13):
        if alpha == 1.0: return 1 / c
        return (1 - alpha**(k+1)) / ((1 - alpha) * (c*k + 1))

    print(f"{'Client / Domain':<20} | {'Mean Alpha':<15} | {'Est. Speedup (k=4)':<20}")
    print("-" * 70)
    
    for client in clients:
        alpha_data = client.history_alpha
        if not alpha_data:
            print(f"Client {client.client_id} ({client.domain:<10}) | No data")
            continue
            
        mean_alpha = sum(alpha_data) / len(alpha_data)
        speedup = calc_speedup(mean_alpha)
        
        print(f"Client {client.client_id} ({client.domain:<10}) | {mean_alpha:.4f}          | {speedup:.2f}x")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()