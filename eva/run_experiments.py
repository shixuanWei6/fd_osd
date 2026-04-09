import argparse
import logging
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Adjust these imports based on your exact directory structure
from model.model_manager import ModelManager
from data.data_manager import DatasetManager
from method.method_manager import MethodManager

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Component 1: Metrics Management (Now with W&B)
# ==========================================
class MetricsRegistry:
    """
    Highly extensible metrics tracker with Weights & Biases integration
    for real-time dashboard monitoring.
    """
    def __init__(self, exp_name: str, global_config: dict = None, use_wandb: bool = False, save_dir: str = "./results"):
        self.exp_name = exp_name
        self.save_dir = save_dir
        self.records = []
        self.use_wandb = use_wandb
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize W&B if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="NeurIPS-Fed-SD", 
                    name=self.exp_name,
                    config=global_config
                )
                logger.info("Weights & Biases initialized successfully.")
            except ImportError:
                logger.error("wandb is not installed. Please run `pip install wandb`. Falling back to local logging.")
                self.use_wandb = False

    def add_record(self, method_name: str, dataset_name: str, config: dict, results: dict):
        """Merges metadata, config, and final metrics into a single record."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "dataset": dataset_name,
            **config,  
            **results  
        }
        self.records.append(record)
        logger.info(f"Recorded final metrics for {method_name} on {dataset_name}: {results}")
        
        # Log the final summary metrics to W&B
        if self.use_wandb:
            import wandb
            # Prefix keys to group them nicely in the W&B dashboard
            wandb_summary = {f"summary/{method_name}/{dataset_name}/{k}": v for k, v in results.items()}
            wandb.log(wandb_summary)

    def save_all(self):
        """Saves records to CSV/JSON and closes the W&B run."""
        if not self.records:
            return None

        df = pd.DataFrame(self.records)
        
        # Determine baseline speed for speedup calculations (target_only)
        if "target_only" in df["method"].values and "ms_per_token" in df.columns:
            baseline_rows = df[df["method"] == "target_only"]
            for dataset in df["dataset"].unique():
                baseline_ms = baseline_rows[baseline_rows["dataset"] == dataset]["ms_per_token"].values
                if len(baseline_ms) > 0:
                    dataset_mask = df["dataset"] == dataset
                    df.loc[dataset_mask, "calculated_speedup"] = baseline_ms[0] / df.loc[dataset_mask, "ms_per_token"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.save_dir, f"{self.exp_name}_{timestamp}")
        
        df.to_csv(f"{base_filename}.csv", index=False)
        with open(f"{base_filename}.json", "w") as f:
            json.dump(self.records, f, indent=4)
            
        logger.info(f"Saved local metrics to {base_filename}.csv")

        # Finish the W&B run cleanly
        if self.use_wandb:
            import wandb
            wandb.finish()
            
        return f"{base_filename}.csv"

# ==========================================
# Component 2: Decoupled Plotting
# ==========================================
class ExperimentPlotter:
    """
    Isolated plotting class. Reads from CSVs to generate graphs.
    Easy to extend with new plotting functions for new metrics.
    """
    def __init__(self, save_dir: str = "./results/plots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid") # Clean, academic style

    def plot_speedup_comparison(self, csv_path: str):
        """Plots a bar chart comparing the speedup of different methods."""
        df = pd.read_csv(csv_path)
        if "calculated_speedup" not in df.columns:
            logger.warning("calculated_speedup metric not found. Skipping plot.")
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="dataset", y="calculated_speedup", hue="method", palette="viridis")
        
        plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (Target Only)')
        plt.title("End-to-End Speedup Comparison Across Datasets", fontsize=14, fontweight='bold')
        plt.ylabel("Speedup Factor (x)", fontsize=12)
        plt.xlabel("Dataset", fontsize=12)
        plt.legend(title="Decoding Method")
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, f"speedup_comparison_{os.path.basename(csv_path)}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved speedup plot to {plot_path}")

    def plot_acceptance_rate(self, csv_path: str):
        """Plots the token acceptance rate (alpha) across methods."""
        df = pd.read_csv(csv_path)
        if "alpha" not in df.columns:
            return

        # Filter out methods where alpha isn't applicable
        df_filtered = df[pd.to_numeric(df['alpha'], errors='coerce').notnull()]
        df_filtered["alpha"] = df_filtered["alpha"].astype(float)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_filtered, x="dataset", y="alpha", hue="method", palette="mako")
        
        plt.title("Token Acceptance Rate (α) Comparison", fontsize=14, fontweight='bold')
        plt.ylabel("Acceptance Rate", fontsize=12)
        plt.xlabel("Dataset", fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(title="Decoding Method")
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, f"alpha_comparison_{os.path.basename(csv_path)}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved alpha plot to {plot_path}")


# ==========================================
# Component 3: Main Execution Engine
# ==========================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="NeurIPS Federated Speculative Decoding Evaluation Pipeline")
    parser.add_argument("--exp_name", type=str, default="baseline_eval")
    parser.add_argument("--target_model", type=str, default="vicuna-7b")
    parser.add_argument("--draft_model", type=str, default="llama-160m")
    parser.add_argument("--datasets", nargs="+", default=["spider", "gsm8k"])
    parser.add_argument("--methods", nargs="+", default=["target_only", "vanilla_sd", "osd", "ours_fed_sd"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--simulated_rtt_ms", type=float, default=50.0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--update_interval", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    # NEW W&B FLAG
    parser.add_argument("--use_wandb", action="store_true", help="Enable real-time tracking with Weights & Biases")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger.info(f"Starting Experiment: {args.exp_name}")
    
    base_config = vars(args)
    
    # Initialize Registry with W&B flag and config
    metrics_registry = MetricsRegistry(
        exp_name=args.exp_name, 
        global_config=base_config, 
        use_wandb=args.use_wandb
    )
    
    # 1. Load Models & Tokenizer Once
    logger.info("Initializing models...")
    tokenizer = ModelManager.load_tokenizer(args.target_model)
    target_model = ModelManager.load_target_model(args.target_model)
    
    # Apply Dual-LoRA only if testing your method, else load base draft
    needs_lora = "ours_fed_sd" in args.methods
    draft_model = ModelManager.load_draft_model(args.draft_model, apply_dual_lora=needs_lora)

    # Base configuration dictionary to attach to every recorded metric
    base_config = {
        "num_samples": args.num_samples,
        "K": args.k,
        "I": args.update_interval,
        "lr": args.lr,
        "simulated_rtt_ms": args.simulated_rtt_ms
    }
    
    # Separate dict for metrics recording (includes model names for reference)
    metrics_config = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        **base_config
    }

    # 2. Main Evaluation Loop (Dataset -> Method)
    for dataset_name in args.datasets:
        logger.info(f"\n{'='*40}\nEvaluating Dataset: {dataset_name}\n{'='*40}")
        dataset = DatasetManager.get_dataset(dataset_name, split="train") # Ensure OSD alignment
        
        for method_name in args.methods:
            logger.info(f"\n--- Running Method: {method_name} ---")
            
            # Execute the method dynamically through the registry
            results = MethodManager.run_method(
                name=method_name,
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                dataset=dataset,
                **base_config
            )
            
            # Log results
            metrics_registry.add_record(method_name, dataset_name, metrics_config, results)

    # 3. Save Data and Generate Plots
    logger.info("\nEvaluation complete. Processing metrics...")
    saved_csv_path = metrics_registry.save_all()
    
    if saved_csv_path:
        plotter = ExperimentPlotter()
        plotter.plot_speedup_comparison(saved_csv_path)
        plotter.plot_acceptance_rate(saved_csv_path)

if __name__ == "__main__":
    main()