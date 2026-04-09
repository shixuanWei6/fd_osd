import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """
    A registry and factory class for managing HuggingFace models and tokenizers.
    Easily extensible for new draft and target models.
    """
    
    # Pre-defined mapping of short names to HuggingFace repository IDs
    # Aligned with OSD (ICML 2024) settings + your future mobile LLM needs
    MODEL_PATHS = {
        # --- OSD Pair A (Fast Iteration / Ablation) ---
        "llama-160m": "JackFram/llama-160m",
        "vicuna-7b": "lmsys/vicuna-7b-v1.5",
        
        # --- OSD Pair B (Main Performance Table) ---
        "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "vicuna-33b": "lmsys/vicuna-33b-v1.3",
        
        # --- Future / Alternative Extensibility ---
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
    }

    @classmethod
    def register_model_path(cls, name: str, hf_repo_id: str):
        """Register a new model path dynamically."""
        if name in cls.MODEL_PATHS:
            logger.warning(f"Model '{name}' is already registered. Overwriting with {hf_repo_id}.")
        cls.MODEL_PATHS[name] = hf_repo_id
        logger.info(f"Registered new model: {name} -> {hf_repo_id}")

    @classmethod
    def load_tokenizer(cls, model_name: str) -> AutoTokenizer:
        """Loads and returns the tokenizer for a given model."""
        if model_name not in cls.MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(cls.MODEL_PATHS.keys())}")
            
        repo_id = cls.MODEL_PATHS[model_name]
        logger.info(f"Loading tokenizer for {model_name} ({repo_id})...")
        
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer

    @classmethod
    def load_target_model(cls, model_name: str, use_4bit: bool = False, **kwargs) -> AutoModelForCausalLM:
        """
        Loads the massive Cloud Target Model (Oracle).
        Automatically freezes parameters and applies device_map="auto".
        """
        if model_name not in cls.MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(cls.MODEL_PATHS.keys())}")

        repo_id = cls.MODEL_PATHS[model_name]
        logger.info(f"Loading TARGET model {model_name} ({repo_id})...")

        # Handle optional 4-bit quantization for massive models (e.g., 70B)
        quantization_config = None
        if use_4bit:
            logger.info("Applying 4-bit BitsAndBytes quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Base loading configuration
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "trust_remote_code": True,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            
        # Merge any user overrides
        load_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
        
        # Oracle target model should always be completely frozen
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        return model

    @classmethod
    def load_draft_model(cls, model_name: str, apply_dual_lora: bool = False, **kwargs) -> AutoModelForCausalLM:
        """
        Loads the small Edge Draft Model (Student).
        Optional flag to automatically wrap it in your Dual-LoRA architecture.
        """
        if model_name not in cls.MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(cls.MODEL_PATHS.keys())}")

        repo_id = cls.MODEL_PATHS[model_name]
        logger.info(f"Loading DRAFT model {model_name} ({repo_id})...")

        load_kwargs = {
            # Typically mapped to a specific GPU/device downstream, so we don't force 'auto' here 
            # unless not specified, so federated clients can be placed manually
            "device_map": kwargs.pop("device_map", "auto"), 
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "trust_remote_code": True,
        }
        load_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)

        # Hook for your specific Dual-LoRA architecture from models.lora_adapter
        if apply_dual_lora:
            logger.info("Applying Dual-LoRA architecture to Draft Model...")
            try:
                from project.models.lora_adapter import setup_dual_lora
                # Pass LoRA specific kwargs like 'r' and 'lora_alpha' if they exist
                r = kwargs.get("r", 32)
                lora_alpha = kwargs.get("lora_alpha", 64)
                model = setup_dual_lora(model, r=r, lora_alpha=lora_alpha)
            except ImportError:
                logger.error("Could not import setup_dual_lora from models.lora_adapter. Returning base model.")

        return model

if __name__ == "__main__":
    # --- Quick Test ---
    # Registering a custom model path on the fly
    ModelManager.register_model_path("custom-draft", "Qwen/Qwen1.5-0.5B")
    
    # Example: Loading Tokenizer and Draft Model
    tokenizer = ModelManager.load_tokenizer("llama-160m")
    draft = ModelManager.load_draft_model("llama-160m", apply_dual_lora=False)
    print(f"Draft model loaded on: {draft.device}")