import logging
import time
import torch
from tqdm import tqdm
from typing import Callable, Dict, Any, List

# Import the generation strategies and processors from the reference codebase
# Note: Adjust the import paths based on where you place the reference code in your project
from sampling.base_decoding import autoregressive_generate, beam_search_generate
from sampling.speculative_decoding import speculative_generate
from sampling.utils.logits_processor import GreedyProcessor

# Try to import your custom engines for OSD and Fed-SD
try:
    from project.engine.speculator import speculative_inference_with_feedback
    from project.training.buffer import ExperienceReplayBuffer
    from project.training.trainer import OnlineTrainer
except ImportError:
    logging.warning("Could not import 'ours' modules. Ensure 'ours' is in your PYTHONPATH.")
# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MethodManager:
    """
    A registry pattern class for managing different decoding/serving methods.
    Provides a standardized interface to run experiments across baselines.
    """
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a new method under a specific name."""
        def wrapper(func: Callable) -> Callable:
            if name in cls._registry:
                logger.warning(f"Method '{name}' is already registered. Overwriting.")
            cls._registry[name] = func
            return func
        return wrapper

    @classmethod
    def run_method(cls, name: str, target_model, draft_model, tokenizer, dataset, **kwargs) -> Dict[str, Any]:
        """
        Factory method to execute a specific decoding/serving baseline.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Method '{name}' not found. Available methods: {available}")
        
        logger.info(f"Running Experiment Method: {name} with params {kwargs}")
        return cls._registry[name](target_model, draft_model, tokenizer, dataset, **kwargs)

# ==========================================
# Helper Function for Evaluation
# ==========================================
def extract_prompts(dataset, num_samples):
    """Helper to extract prompts uniformly from the loaded dataset."""
    prompts = []
    for item in dataset.select(range(min(num_samples, len(dataset)))):
        # Adjust 'question' or 'query' based on specific dataset schema
        text = item.get("question", item.get("query", item.get("text", "")))
        prompts.append(text)
    return prompts

# ==========================================
# Registered Baseline Methods
# ==========================================

@MethodManager.register("target_only")
def run_target_only(target_model, draft_model, tokenizer, dataset, **kwargs):
    """
    Baseline 1: Standard Autoregressive Decoding on the Cloud (Upper bound on quality).
    Simulates high network latency, no speculative drafting.
    """
    num_samples = kwargs.get("num_samples", 100)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    simulated_rtt = kwargs.get("simulated_rtt_ms", 50) / 1000.0  # e.g., 50ms per network trip
    
    prompts = extract_prompts(dataset, num_samples)
    total_tokens, total_time, network_time = 0, 0.0, 0.0

    logger.info("Executing Target-Only (Cloud) Baseline...")
    target_model.eval()

    processor = GreedyProcessor()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Target-Only Generating"):
            # The reference codebase expects inputs as a List[int]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
            input_length = len(input_ids)
            
            start_time = time.time()
            
            output_ids = autoregressive_generate(
                inputs=input_ids,
                model=target_model,
                max_gen_len=max_new_tokens,
                logits_processor=processor,
                eos_tokens_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True
            )
            
            generation_time = time.time() - start_time
            generated_tokens = len(output_ids) - input_length
            
            # Penalize cloud requests: 1 network trip per generated token
            penalty = generated_tokens * simulated_rtt 
            
            total_tokens += generated_tokens
            total_time += generation_time
            network_time += penalty

    total_latency = total_time + network_time
    ms_per_token = (total_latency / total_tokens) * 1000 if total_tokens > 0 else 0

    return {
        "alpha": 1.0, # 100% acceptance since it's the oracle
        "total_tokens_generated": total_tokens,
        "ms_per_token": ms_per_token,
        "compute_time_s": total_time,
        "network_time_s": network_time,
        "speedup": 1.0 # Baseline for speedup calculation
    }

@MethodManager.register("draft_only")
def run_draft_only(target_model, draft_model, tokenizer, dataset, **kwargs):
    """
    Baseline 2: Standard Autoregressive Decoding on the Edge (Upper bound on speed, lower on quality).
    No network latency, but highly prone to hallucination/errors.
    """
    num_samples = kwargs.get("num_samples", 100)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    
    prompts = extract_prompts(dataset, num_samples)
    total_tokens, total_time = 0, 0.0

    logger.info("Executing Draft-Only (Edge) Baseline...")
    draft_model.eval()

    processor = GreedyProcessor()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Draft-Only Generating"):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
            input_length = len(input_ids)
            
            start_time = time.time()
            
            output_ids = autoregressive_generate(
                inputs=input_ids,
                model=draft_model,
                max_gen_len=max_new_tokens,
                logits_processor=processor,
                eos_tokens_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True
            )
            
            generation_time = time.time() - start_time
            generated_tokens = len(output_ids) - input_length
            
            total_tokens += generated_tokens
            total_time += generation_time

    # Network time is explicitly 0.0 here, highlighting the edge advantage.
    ms_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0

    return {
        "alpha": "N/A (No Verification)", 
        "total_tokens_generated": total_tokens,
        "ms_per_token": ms_per_token,
        "compute_time_s": total_time,
        "network_time_s": 0.0,
        "speedup": "MAX (Edge-bound)"
    }

@MethodManager.register("vanilla_sd")
def run_vanilla_sd(target_model, draft_model, tokenizer, dataset, **kwargs):
    """
    Baseline 3: Vanilla Speculative Decoding.
    Fixed speculation length K, no online learning, no buffer updates.
    """
    num_samples = kwargs.get("num_samples", 100)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    k = kwargs.get("K", 5) # Default K=5 per OSD paper
    simulated_rtt = kwargs.get("simulated_rtt_ms", 50) / 1000.0
    
    prompts = extract_prompts(dataset, num_samples)
    total_tokens, total_time, network_time = 0, 0.0, 0.0
    total_alpha = 0.0
    valid_alpha_count = 0

    logger.info(f"Executing Vanilla Speculative Decoding (K={k})...")
    target_model.eval()
    draft_model.eval()

    processor = GreedyProcessor()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Vanilla SD Generating"):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
            input_length = len(input_ids)
            
            start_time = time.time()
            
            # Using the custom speculative_generate from the reference codebase
            output_ids, accept_rate = speculative_generate(
                inputs=input_ids,
                drafter=draft_model,
                target=target_model,
                tokenizer=tokenizer,
                gamma=k,
                logits_processor=processor,
                max_gen_len=max_new_tokens,
                eos_tokens_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
                debug=False
            )
            
            generation_time = time.time() - start_time
            generated_tokens = len(output_ids) - input_length
            
            # Approximate network trips: generated_tokens / (average accepted drafts + 1 target token)
            # This mathematically models sending batches of drafts to the cloud
            avg_accepted_per_step = accept_rate * k
            network_trips = generated_tokens / (avg_accepted_per_step + 1.0)
            penalty = network_trips * simulated_rtt 
            
            total_tokens += generated_tokens
            total_time += generation_time
            network_time += penalty
            
            # Record Alpha
            total_alpha += accept_rate
            valid_alpha_count += 1

    total_latency = total_time + network_time
    ms_per_token = (total_latency / total_tokens) * 1000 if total_tokens > 0 else 0
    mean_alpha = total_alpha / valid_alpha_count if valid_alpha_count > 0 else 0.0

    return {
        "alpha": mean_alpha, 
        "total_tokens_generated": total_tokens,
        "ms_per_token": ms_per_token,
        "compute_time_s": total_time,
        "network_time_s": network_time,
        "speedup": "Calculated relative to target_only ms_per_token"
    }

@MethodManager.register("osd")
def run_osd(target_model, draft_model, tokenizer, dataset, **kwargs):
    """
    Baseline 4: Online Speculative Decoding (ICML 2024).
    Directly uses the official OSD Generator and Distillation Step logic.
    """
    num_samples = kwargs.get("num_samples", 100)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    k = kwargs.get("K", 5)
    update_interval = kwargs.get("I", 8) 
    lr = kwargs.get("lr", 1e-5)
    simulated_rtt = kwargs.get("simulated_rtt_ms", 50) / 1000.0

    prompts = extract_prompts(dataset, num_samples)
    total_tokens, total_time, network_time = 0, 0.0, 0.0
    total_alpha, valid_alpha_count = 0.0, 0

    logger.info(f"Executing OSD (K={k}, I={update_interval}, LR={lr}) using official implementation...")

    # --- IMPORT OSD CODEBASE ---
    try:
        from OSD.distill.specInfer.generator import Generator
        from OSD.distill.specInfer.common import pad_to_2d
    except ImportError:
        logger.error("Could not import OSD modules. Ensure the OSD repository is in your PYTHONPATH.")
        return {"status": "Failed to import official OSD codebase."}

    # 1. Setup Models
    target_model.eval()
    draft_model.train() 
    for param in draft_model.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(draft_model.parameters(), lr=lr)

    # 2. Initialize OSD's Official Generator
    osd_generator = Generator(
        small_model=draft_model,
        large_model=target_model,
        tokenizer=tokenizer,
        max_propose_num=k,
        is_encoder_decoder=target_model.config.is_encoder_decoder, 
        use_cache=True
    )

    buffer = []

    # Helper replicating exactly how DistillTrainer computes Soft Cross Entropy
    def soft_cross_entropy(predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        return entropy.sum() / (~padding_mask).sum()

    for step, prompt in enumerate(tqdm(prompts, desc="OSD Generating & Adapting")):
        # Must be in eval mode to prevent dropout randomness during generation
        draft_model.eval() 
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(target_model.device)
        attention_mask = inputs.attention_mask.to(target_model.device)
        input_length = input_ids.shape[1]
        
        start_time = time.time()
        
        # 3. Generate using official OSD mechanism
        # OSD uses temperature=0.01 by default to simulate greedy decoding inside speculative loop
        output = osd_generator.generate(
            input_ids=input_ids,
            max_tokens=max_new_tokens,
            temperature=0.01, 
            attention_mask=attention_mask
        )
        
        generation_time = time.time() - start_time
        generated_tokens = output.generated_ids.shape[1] - input_length
        
        # Calculate Alpha (accepted / sampled steps)
        alpha = output.alpha_sum / output.sample_steps if output.sample_steps > 0 else 0
        
        # Calculate Latency 
        avg_accepted_per_step = alpha * k
        network_trips = generated_tokens / (avg_accepted_per_step + 1.0)
        network_time += network_trips * simulated_rtt
        total_time += generation_time
        total_tokens += generated_tokens
        total_alpha += alpha
        valid_alpha_count += 1
        
        # 4. OSD Online Update Routine (Knowledge Distillation)
        # Replicates logic from distill_trainer.py: online_training_step
        token_ids = torch.cat([input_ids, output.generated_ids.to(target_model.device)], dim=-1)
        wrong_token_ids = [input_length + t for t in output.wrong_token_ids]
        buffer.append((token_ids, wrong_token_ids))
            
        if len(buffer) >= update_interval:
            draft_model.train() # Switch back to train mode for backprop
            
            # Pad sequences in the buffer
            padded_input_ids = pad_to_2d([x[0].squeeze(0) for x in buffer], tokenizer.pad_token_id).to(target_model.device)
            
            # Get logits
            student_logits = draft_model(input_ids=padded_input_ids, attention_mask=torch.ones_like(padded_input_ids)).logits.float()
            with torch.no_grad():
                teacher_logits = target_model(input_ids=padded_input_ids, attention_mask=torch.ones_like(padded_input_ids)).logits.float()

            # Mask correct tokens (OSD trains ONLY on wrong predictions)
            mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
            for i, data in enumerate(buffer):
                cur_wrong_token_ids = data[1]
                mask[i, cur_wrong_token_ids] = False

            # Compute Distillation loss
            loss = soft_cross_entropy(student_logits, teacher_logits, mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            buffer = []
            logger.debug(f"OSD Update Step | KD Loss: {loss.item():.4f}")

    ms_per_token = ((total_time + network_time) / total_tokens) * 1000 if total_tokens > 0 else 0

    return {
        "alpha": total_alpha / valid_alpha_count if valid_alpha_count > 0 else 0.0, 
        "total_tokens_generated": total_tokens,
        "ms_per_token": ms_per_token,
        "compute_time_s": total_time,
        "network_time_s": network_time
    }

@MethodManager.register("ours_fed_sd")
def run_ours_fed_sd(target_model, draft_model, tokenizer, dataset, **kwargs):
    """
    Our Proposed Method: Edge-Cloud Federated Speculative Decoding.
    Dual-LoRA, RL-driven dynamic horizon, Sequence-Level Mixed Feedback.
    """
    logger.info("Executing Our Edge-Cloud Fed-SD...")
    
    # Import the newly created engine
    try:
        from project.core_engine import EdgeCloudFedSDEngine
    except ImportError:
        logger.error("Could not import EdgeCloudFedSDEngine. Check your python path.")
        return {"status": "Failed"}
        
    engine = EdgeCloudFedSDEngine(target_model, draft_model, tokenizer, **kwargs)
    
    # Run the encapsulated evaluation
    results = engine.run_simulation(dataset, num_samples=kwargs.get("num_samples", 100))
    
    return results

if __name__ == "__main__":
    # --- Quick Test to verify the Registry ---
    try:
        dummy_results = MethodManager.run_method("vanilla_sd", None, None, None, None, K=8)
        print(f"Vanilla SD initialized successfully: {dummy_results}")
    except Exception as e:
        print(f"Error: {e}")