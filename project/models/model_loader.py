import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 从你的 main.py 中提取的常用模型路径，作为默认备选
# 你可以根据实际情况修改这里的路径
DEFAULT_MODEL_PATHS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-step-50K-105b", 
    "bloom-560m": "bigscience/bloom-560m",
    "bloom7b": "bigscience/bloomz-7b1",
}


def load_models(draft_model_name_or_path, target_model_name_or_path, device="cuda"):
    """
    加载 Draft Model 和 Target Model。
    
    Args:
        draft_model_name_or_path: 草稿模型路径或名称
        target_model_name_or_path: 目标大模型路径或名称
        device: 指定设备，默认为 cuda
    
    Returns:
        draft_model, target_model, tokenizer
    """
    logging.info(f"Loading Draft Model: {draft_model_name_or_path}")
    logging.info(f"Loading Target Model: {target_model_name_or_path}")

    # 1. Load Tokenizer
    # 注意：Speculative Decoding 要求两个模型使用相同的 Tokenizer
    # 通常加载 Draft 模型的 Tokenizer 即可，因为它们通常是同系列的
    tokenizer = AutoTokenizer.from_pretrained(draft_model_name_or_path, trust_remote_code=True)
    
    # 确保 pad_token 存在，否则训练时 padding 会报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Draft Model (Small)
    # 边缘端模型，我们需要对它进行训练，所以加载后保留梯度
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" # 自动分配到显卡
    )
    
    # 3. Load Target Model (Large)
    # 云端模型，仅用于验证（Inference only），可以加载为 int4/8 节省显存，或者 fp16
    # 你的环境有 48G 显存，跑 Llama-2-70B 可能需要多卡或量化
    # 预实验建议先用 13B 或 7B 作为 Target，跑通流程
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 冻结 Target Model，它只充当 Oracle
    for param in target_model.parameters():
        param.requires_grad = False
    target_model.eval()

    logging.info("Models loaded successfully.")
    
    return draft_model, target_model, tokenizer