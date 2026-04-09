# fileName: data_registry.py
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