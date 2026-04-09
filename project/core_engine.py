# fileName: ours/core_engine.py
import time
import torch
import logging
from tqdm import tqdm
from .engine.speculator import speculative_inference_with_feedback
from .training.buffer import ExperienceReplayBuffer
from .training.trainer import OnlineTrainer
from .simulation.network import NetworkSimulator, RLHorizonController

logger = logging.getLogger(__name__)

class EdgeCloudFedSDEngine:
    """
    Encapsulated Engine for Edge-Cloud Federated Speculative Decoding.
    Handles the initialization of Dual-LoRA, RL Horizon, and Online Distillation.
    """
    def __init__(self, target_model, draft_model, tokenizer, **kwargs):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        
        # Hyperparameters
        self.k_default = kwargs.get("K", 4)
        self.update_interval = kwargs.get("I", 8)
        self.batch_size = kwargs.get("batch_size", 4)
        self.lr = kwargs.get("lr", 1e-5)
        self.simulated_rtt = kwargs.get("simulated_rtt_ms", 50) / 1000.0
        
        # 1. Setup Online Learning Infrastructure
        local_params = [p for n, p in self.draft_model.named_parameters() if p.requires_grad]
        if not local_params:
            logger.warning("No trainable parameters found in draft model! Ensure LoRA is applied.")
            
        self.optimizer = torch.optim.AdamW(local_params, lr=self.lr)
        self.buffer = ExperienceReplayBuffer(capacity=256)
        
        # Your custom trainer uses temp=2.0, alpha_kd=0.6 based on main_pre_exp.py
        self.trainer = OnlineTrainer(
            model=self.draft_model, 
            optimizer=self.optimizer, 
            buffer=self.buffer, 
            temp=2.0, 
            alpha_kd=0.6
        )
        
        # 2. Setup RL and Network Simulation
        self.network_sim = NetworkSimulator()
        self.rl_controller = RLHorizonController()

    def extract_prompts(self, dataset, num_samples):
        prompts = []
        for item in dataset.select(range(min(num_samples, len(dataset)))):
            text = item.get("question", item.get("query", item.get("instruction", "")))
            prompts.append(text)
        return prompts

    def run_simulation(self, dataset, num_samples=100, max_new_tokens=128):
        """
        Executes the evaluation loop, returning standard metrics for MethodManager.
        """
        prompts = self.extract_prompts(dataset, num_samples)
        
        total_tokens = 0
        total_time = 0.0
        network_time = 0.0
        total_alpha = 0.0
        valid_alpha_count = 0
        
        self.target_model.eval()
        # The draft model MUST be in train mode for the trainer to update gradients properly
        self.draft_model.train() 

        logger.info(f"Starting Fed-SD Evaluation (Samples: {len(prompts)}, Base K: {self.k_default})")

        try:
            import wandb
            has_wandb = wandb.run is not None
        except ImportError:
            has_wandb = False


        for step, prompt in enumerate(tqdm(prompts, desc="Fed-SD Generating & Learning")):
            # Temporarily set to eval to avoid dropout stochasticity during generation
            self.draft_model.eval() 
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.target_model.device)
            input_length = input_ids.shape[1]
            
            start_time = time.time()
            
            # --- CORE GENERATION ---
            output_ids, alpha, feedback_buffer, rl_stats = speculative_inference_with_feedback(
                draft_model=self.draft_model, 
                target_model=self.target_model, 
                input_ids=input_ids, 
                gamma=self.k_default, 
                temperature=0.6,   
                top_p=0.9,         
                training_mode=True, # Enables feedback collection
                rl_controller=self.rl_controller, 
                network_sim=self.network_sim
            )
            
            generation_time = time.time() - start_time
            generated_tokens = output_ids.shape[1] - input_length
            
            # --- METRICS CALCULATION ---
            # Simulate actual network trips based on the RL agent's chosen K
            avg_k = rl_stats.get("avg_k", self.k_default)
            avg_accepted_per_step = alpha * avg_k
            network_trips = generated_tokens / (avg_accepted_per_step + 1.0)
            
            network_time += network_trips * self.simulated_rtt
            total_time += generation_time
            total_tokens += generated_tokens
            total_alpha += alpha
            valid_alpha_count += 1
            
            if has_wandb:
                wandb.log({
                    "live/alpha": alpha,
                    "live/rl_chosen_k": avg_k,
                    "live/network_rtt_ms": self.network_sim.get_rtt() if self.network_sim else self.simulated_rtt * 1000,
                    "step": step
                }, commit=False) # commit=False means wait for the rest of the step's logs


            # --- ONLINE LEARNING UPDATE ---
            if len(feedback_buffer) > 0:
                self.buffer.add(feedback_buffer)
                
            # Execute training step every `update_interval`
            if step > 0 and step % self.update_interval == 0 and self.buffer.is_ready(self.batch_size):
                self.draft_model.train() 
                loss = self.trainer.train_step(batch_size=self.batch_size)
                
                # 2. LIVE LOGGING: Training Dynamics (Loss)
                if has_wandb:
                    wandb.log({
                        "live/train_loss": loss,
                        "step": step
                    }, commit=True) # commit=True pushes this step's data to the dashboard
                
                logger.debug(f"Fed-SD Update Step {step} | Loss: {loss:.4f} | Avg K: {avg_k:.1f}")

        # Compute final standard metrics
        ms_per_token = ((total_time + network_time) / total_tokens) * 1000 if total_tokens > 0 else 0
        mean_alpha = total_alpha / valid_alpha_count if valid_alpha_count > 0 else 0.0

        return {
            "alpha": mean_alpha, 
            "total_tokens_generated": total_tokens,
            "ms_per_token": ms_per_token,
            "compute_time_s": total_time,
            "network_time_s": network_time
        }