import torch
from peft import LoraConfig, get_peft_model, TaskType
import re

def _get_target_modules(model):
    """
    根据模型架构自动匹配合适的 LoRA target_modules。
    支持 Llama, Bloom, OPT, Qwen 等常见架构。
    """
    # 获取模型的类名，例如 "BloomForCausalLM" -> "Bloom"
    model_class = model.__class__.__name__
    
    print(f"[LoRA Setup] Detected model architecture: {model_class}")

    if "Llama" in model_class or "Mistral" in model_class or "TinyLlama" in model_class:
        # Llama 家族的标准命名
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    elif "Bloom" in model_class:
        # Bloom 的命名风格完全不同
        # query_key_value 是 Q, K, V 的融合层
        # dense, dense_h_to_4h, dense_4h_to_h 是 FFN 层
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    elif "Baichuan" in model_class:
        # Baichuan 通常沿用 Llama 风格，或者是 W_pack
        return ["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"]
    
    elif "ChatGLM" in model_class:
         return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    elif "GPT2" in model_class:
        return ["c_attn", "c_proj", "c_fc"]
    
    else:
        # 默认回退：尝试打印模型结构并让用户检查，这里先给一个通用猜测
        print(f"[Warning] Unknown model architecture {model_class}. Defaulting to simple linear layers.")
        # 你可以通过 print(model) 查看层名称来手动填入
        return ["q_proj", "v_proj"] 

def setup_single_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """
    【预实验 Baseline】: 仅挂载一个标准的 LoRA 适配器。
    """
    print("Setting up Single LoRA Adapter...")
    
    # 自动获取目标模块
    target_modules = _get_target_modules(model)
    print(f"[LoRA Setup] Target modules set to: {target_modules}")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
from peft import LoraConfig, TaskType, PeftModel

def setup_dual_lora(model, r=32, lora_alpha=64, pretrained_global_path="/private/wsx/federated/project/pretrained_global_lora"):
    """
    Sets up the Dual-LoRA architecture by loading a pre-trained global adapter 
    and injecting a fresh local adapter on top.
    """
    print("Setting up Heterogeneity-Aware Dual-LoRA Architecture...")
    
    target_modules = _get_target_modules(model)
    
    # 1. Load the Pre-trained Global Adapter
    # Instead of creating a fresh config, we load the weights saved from Step 1
    try:
        model = PeftModel.from_pretrained(
            model, 
            pretrained_global_path, 
            adapter_name="global_adapter",
            is_trainable=True # We still want to update it during Federated Learning
        )
        print(f"Successfully loaded pre-trained Global LoRA from {pretrained_global_path}")
    except Exception as e:
        print(f"Warning: Could not load pre-trained LoRA. Falling back to fresh initialization. Error: {e}")
        # Fallback to fresh init if path doesn't exist
        global_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules
        )
        model = get_peft_model(model, global_config, adapter_name="global_adapter")

    # 2. Define and inject the fresh Local Adapter
    local_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules
    )
    model.add_adapter("local_adapter", local_config)
    
    # 3. Activate both adapters simultaneously
    model.base_model.set_adapter(["global_adapter", "local_adapter"])
    
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    print(f"Dual-LoRA setup complete. Active adapters: {model.active_adapters}")
    model.print_trainable_parameters()
    
    return model