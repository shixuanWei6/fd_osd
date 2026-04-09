import os
import time
import argparse
import random

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset

from models.model_loader import load_models
from engine.speculator import speculative_inference_with_feedback


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(data, window_size=20):
    if data is None or (isinstance(data, np.ndarray) and data.size == 0) or len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def get_multi_client_streams(tokenizer, sample_size=2000):
    """
    Mirrors `main_pre_exp.py` prompt formatting + dataset streaming order.
    Returns 3 ordered lists: Math, Finance, General.
    """
    print(f"Loading datasets for distinct clients (Sample Size: {sample_size})...")

    # Client 0: Math (GSM8k)
    ds_math = load_dataset("gsm8k", "main", split="train", streaming=True)
    math_prompts = []
    for item in ds_math:
        if len(math_prompts) >= sample_size:
            break
        # MATCHES PRE-TRAINING
        math_prompts.append(f"Question: {item['question']}\nAnswer:")

    # Client 1: Finance (Alpaca-Finance)
    try:
        ds_fin = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    except Exception:
        ds_fin = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)

    fin_prompts = []
    for item in ds_fin:
        if len(fin_prompts) >= sample_size:
            break

        # Combine instruction and input, but format as Question/Answer
        instruction = item.get("instruction", "")
        context = item.get("input", "")
        combined_text = f"{instruction}\n{context}" if context else instruction

        # MATCHES PRE-TRAINING
        fin_prompts.append(f"Question: {combined_text.strip()}\nAnswer:")

    # Client 2: General/Code (Alpaca-Cleaned)
    ds_gen = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    gen_prompts = []
    for item in ds_gen:
        if len(gen_prompts) >= sample_size:
            break

        instruction = item.get("instruction", "")
        gen_prompts.append(f"Question: {instruction.strip()}\nAnswer:")

    return {"Math": math_prompts, "Finance": fin_prompts, "General": gen_prompts}


def _sync_if_cuda():
    # For accurate wall-clock latency, ensure pending CUDA kernels complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_static_speculative_for_prompt(
    draft_model,
    target_model,
    tokenizer,
    prompt,
    *,
    gamma,
    temperature,
    top_p,
    top_k=0,
):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids

    _sync_if_cuda()
    start = time.perf_counter()

    generated_ids, alpha, _feedback_buffer, _rl_stats = speculative_inference_with_feedback(
        draft_model,
        target_model,
        input_ids,
        gamma=gamma,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        training_mode=False,  # critical: no online learning / no feedback buffer
        rl_controller=None,
        network_sim=None,
    )

    _sync_if_cuda()
    elapsed_s = time.perf_counter() - start

    # Number of newly generated tokens for this prompt.
    gen_tokens = int(generated_ids.shape[1] - input_ids.shape[1])
    del generated_ids

    return float(alpha), elapsed_s, gen_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=int, default=4, help="Speculative decoding gamma")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--debug", action="store_true", help="Run with 50 prompts per domain")
    parser.add_argument("--draft_path", type=str, default="bigscience/bloomz-560m")
    parser.add_argument("--target_path", type=str, default="bigscience/bloomz-7b1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_per_domain = 50 if args.debug else 2000

    exp_suffix = os.environ.get("EXP_NAME", "no_lora_static_baseline_multiclient")
    if args.debug:
        exp_suffix += "_debug"
    exp_suffix += f"_gamma{args.gamma}_t{args.temperature}_p{args.top_p}"
    exp_suffix += "_no_lora"

    set_seed(42)

    print("\n=== Loading Target Model (Oracle) + Tokenizer ===")
    _, target_model, tokenizer = load_models(args.draft_path, args.target_path, device)

    client_streams = get_multi_client_streams(tokenizer, sample_size=sample_per_domain)
    domains = ["Math", "Finance", "General"]

    print(f"\n=== Setting up {len(domains)} Client Draft Models (static, NO LoRA) ===")
    clients = []
    for client_id, domain in enumerate(domains):
        print(f"\n--- Initializing Client {client_id} ({domain}) ---")
        c_draft, _, _ = load_models(args.draft_path, args.target_path, device)
        c_draft.eval()
        clients.append({"client_id": client_id, "domain": domain, "model": c_draft})

    # Per-client histories aligned to online learning's step ordering.
    alpha_hist = {d: [] for d in domains}
    latency_hist_s = {d: [] for d in domains}
    gen_tokens_hist = {d: [] for d in domains}

    step_latency_total_s = []

    zipped_prompts = zip(
        client_streams["Math"],
        client_streams["Finance"],
        client_streams["General"],
    )

    smooth_window = 15
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    print("\n=== Running static speculative inference (no online training, no LoRA) ===")
    pbar = tqdm(zipped_prompts, total=sample_per_domain)
    for step, prompt_tuple in enumerate(pbar):
        # Total step time (matches online "per-step" scheduling of clients).
        step_start = time.perf_counter()

        step_alphas = []
        for client, prompt in zip(clients, prompt_tuple):
            domain = client["domain"]
            try:
                alpha, latency_s, gen_tokens = run_static_speculative_for_prompt(
                    client["model"],
                    target_model,
                    tokenizer,
                    prompt,
                    gamma=args.gamma,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception:
                continue

            alpha_hist[domain].append(alpha)
            latency_hist_s[domain].append(latency_s)
            gen_tokens_hist[domain].append(gen_tokens)
            step_alphas.append(alpha)

        step_latency_total_s.append(time.perf_counter() - step_start)

        desc = f"Step {step} | Avg Alpha: {np.mean(step_alphas):.4f}"
        pbar.set_description(desc)

    # -----------------------
    # Summary statistics
    # -----------------------
    print("\n" + "=" * 70)
    print("STATIC BASELINE (NO-LORA) QUANTITATIVE RESULTS SUMMARY")
    print("=" * 70)
    for domain in domains:
        a = np.array(alpha_hist[domain], dtype=np.float64)
        lat_s = np.array(latency_hist_s[domain], dtype=np.float64)
        gen_toks = np.array(gen_tokens_hist[domain], dtype=np.int64)

        mean_alpha = float(a.mean()) if a.size else 0.0
        mean_lat_ms = float((lat_s.mean() * 1000.0)) if lat_s.size else 0.0
        p90_lat_ms = float(np.percentile(lat_s, 90) * 1000.0) if lat_s.size else 0.0
        mean_lat_per_token_ms = float(((lat_s / np.maximum(gen_toks, 1)) * 1000.0).mean()) if lat_s.size else 0.0

        print(
            f"{domain:<8} | Mean Alpha: {mean_alpha:.4f} | "
            f"Mean Latency: {mean_lat_ms:.2f} ms | p90: {p90_lat_ms:.2f} ms"
        )
        print(f"{domain:<8} | Mean Latency/GeneratedToken: {mean_lat_per_token_ms:.4f} ms")

    # -----------------------
    # Plots
    # -----------------------
    os.makedirs(".", exist_ok=True)

    # 1) Acceptance rate plot
    plt.figure(figsize=(12, 6))
    for i, domain in enumerate(domains):
        a = np.array(alpha_hist[domain], dtype=np.float64)
        if a.size == 0:
            continue
        a_smooth = moving_average(a, window_size=smooth_window)
        x = np.arange(len(a_smooth))
        plt.plot(x, a_smooth, color=colors[i], linewidth=2.0, label=f"{domain}")
    plt.xlabel("Inference Steps (Requests)")
    plt.ylabel("Mean Acceptance Rate (Alpha)")
    plt.title("Static Speculative Decoding Baseline (No LoRA): Alpha vs Step")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"baseline_alpha_multiclient_{exp_suffix}.png", dpi=300)
    plt.close()

    # 2) Latency plot
    plt.figure(figsize=(12, 6))
    for i, domain in enumerate(domains):
        lat_s = np.array(latency_hist_s[domain], dtype=np.float64)
        if lat_s.size == 0:
            continue
        lat_ms = lat_s * 1000.0
        lat_smooth = moving_average(lat_ms, window_size=smooth_window)
        x = np.arange(len(lat_smooth))
        plt.plot(x, lat_smooth, color=colors[i], linewidth=2.0, label=f"{domain}")
    plt.xlabel("Inference Steps (Requests)")
    plt.ylabel("Inference Latency (ms, wall-clock)")
    plt.title("Static Speculative Decoding Baseline (No LoRA): Latency vs Step")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"baseline_latency_multiclient_{exp_suffix}.png", dpi=300)
    plt.close()

    # 3) Combined step latency (Math+Finance+General sequential calls)
    plt.figure(figsize=(12, 6))
    step_lat_ms = np.array(step_latency_total_s, dtype=np.float64) * 1000.0
    step_lat_smooth = moving_average(step_lat_ms, window_size=smooth_window)
    x = np.arange(len(step_lat_smooth))
    plt.plot(x, step_lat_smooth, color="#9467bd", linewidth=2.0, label="Total (3 clients sequential)")
    plt.xlabel("Inference Steps (Requests)")
    plt.ylabel("Total Step Wall-Clock Latency (ms)")
    plt.title("Static Speculative Decoding Baseline (No LoRA): Total Step Latency")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"baseline_step_latency_multiclient_{exp_suffix}.png", dpi=300)
    plt.close()

    # Save raw arrays for later comparison.
    np.savez_compressed(
        f"baseline_metrics_multiclient_{exp_suffix}.npz",
        step_indices=np.arange(sample_per_domain, dtype=np.int64),
        step_latency_total_s=np.array(step_latency_total_s, dtype=np.float64),
        alpha_hist_math=np.array(alpha_hist["Math"], dtype=np.float64),
        alpha_hist_finance=np.array(alpha_hist["Finance"], dtype=np.float64),
        alpha_hist_general=np.array(alpha_hist["General"], dtype=np.float64),
        latency_hist_s_math=np.array(latency_hist_s["Math"], dtype=np.float64),
        latency_hist_s_finance=np.array(latency_hist_s["Finance"], dtype=np.float64),
        latency_hist_s_general=np.array(latency_hist_s["General"], dtype=np.float64),
        gen_tokens_hist_math=np.array(gen_tokens_hist["Math"], dtype=np.int64),
        gen_tokens_hist_finance=np.array(gen_tokens_hist["Finance"], dtype=np.int64),
        gen_tokens_hist_general=np.array(gen_tokens_hist["General"], dtype=np.int64),
    )

    print(f"\nBaseline finished. Plots + metrics saved with suffix: '{exp_suffix}'")


if __name__ == "__main__":
    main()

