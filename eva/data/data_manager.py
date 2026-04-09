import logging
from datasets import load_dataset
from typing import Callable, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    A registry pattern class for managing datasets.
    Provides a highly extensible way to add and retrieve datasets without modifying core logic.
    """
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a new dataset loader under a specific name."""
        def wrapper(func: Callable) -> Callable:
            if name in cls._registry:
                logger.warning(f"Dataset '{name}' is already registered. Overwriting.")
            cls._registry[name] = func
            return func
        return wrapper

    @classmethod
    def get_dataset(cls, name: str, **kwargs) -> Any:
        """
        Factory method to instantiate and return a dataset.
        
        Args:
            name (str): The registered name of the dataset.
            **kwargs: Additional arguments to pass to the loader (e.g., split, sample_size).
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
        
        logger.info(f"Loading dataset: {name} with params {kwargs}")
        return cls._registry[name](**kwargs)

# ==========================================
# Registered Dataset Loaders
# ==========================================

@DatasetManager.register("gsm8k")
def load_gsm8k(**kwargs):
    split = kwargs.get("split", "train")
    # For testing, you can pass sample_size to slice the dataset
    sample_size = kwargs.get("sample_size", None)
    
    ds = load_dataset("gsm8k", "main", split=split)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
    return ds

@DatasetManager.register("spider")
def load_spider(**kwargs):
    split = kwargs.get("split", "train")
    sample_size = kwargs.get("sample_size", None)
    
    ds = load_dataset("spider", split=split)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
    return ds

@DatasetManager.register("code_search_python")
def load_code_search_python(**kwargs):
    split = kwargs.get("split", "train")
    sample_size = kwargs.get("sample_size", None)
    
    ds = load_dataset("code_search_net", "python", split=split)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
    return ds

@DatasetManager.register("alpaca_finance")
def load_alpaca_finance(**kwargs):
    split = kwargs.get("split", "train")
    sample_size = kwargs.get("sample_size", None)
    
    ds = load_dataset("gbharti/finance-alpaca", split=split)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
    return ds

# ==========================================
# Example: Adding a local or custom dataset later
# ==========================================
# @DatasetManager.register("custom_local_data")
# def load_custom_data(**kwargs):
#     file_path = kwargs.get("file_path", "data/custom.json")
#     return load_dataset("json", data_files=file_path)

if __name__ == "__main__":
    # Quick test to verify the registry works
    try:
        math_data = DatasetManager.get_dataset("gsm8k", split="train", sample_size=10)
        print(f"Successfully loaded GSM8K: {len(math_data)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")