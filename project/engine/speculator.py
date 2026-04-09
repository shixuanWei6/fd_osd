# fileName: engine/speculator.py
import torch
from torch.nn import functional as F
from .kv_cache import KVCacheModel

def norm_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
    # Keep the same logic, but make the math numerically safe.
    # Under fp16 autocast, logits/probs can become non-finite and later break multinomial.
    temperature = float(temperature) if temperature is not None else 1.0
    temperature = max(temperature, 1e-8)
    logits = logits.float() / temperature
    # If logits have non-finite values, replace them with finite extremes so that:
    # - NaNs become effectively masked out
    # - +Inf becomes dominating (softmax ~= one-hot)
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    if top_k > 0:
        filter_val = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter_val[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter_mask = cumulative_probs > top_p
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = 0
        indices_to_remove = filter_mask.scatter(1, sorted_indices, filter_mask)
        logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)

    # If masking ever produced an all -inf row (rare), softmax can yield all-zeros.
    # Fall back to one-hot at argmax(logits) to avoid breaking training.
    probs_sum = probs.sum(dim=-1, keepdim=True)
    fallback_idx = torch.argmax(logits, dim=-1, keepdim=True)
    fallback = torch.zeros_like(probs).scatter(-1, fallback_idx, 1.0)
    probs = torch.where(probs_sum > 0, probs / probs_sum, fallback)

    return probs

def sample(probs):
    # Ensure `probs` is a valid multinomial distribution without changing sampling intent.
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    if torch.any(probs_sum <= 0):
        fallback_idx = torch.argmax(probs, dim=-1, keepdim=True)
        fallback = torch.zeros_like(probs).scatter(-1, fallback_idx, 1.0)
        probs = torch.where(probs_sum > 0, probs / probs_sum, fallback)
    else:
        probs = probs / probs_sum
    return torch.multinomial(probs, num_samples=1)

def speculative_inference_with_feedback(
    draft_model, target_model, input_ids, 
    gamma=4, top_k=0, top_p=0.0, temperature=1.0, 
    training_mode=False,
    rl_controller=None, network_sim=None
):
    input_ids = input_ids.to(draft_model.device)
    previous_mode = draft_model.training 
    draft_model.eval() 
    if hasattr(draft_model, "enable_adapter_layers"):
        draft_model.enable_adapter_layers()

    draft_cache = KVCacheModel(draft_model)
    target_cache = KVCacheModel(target_model)
    
    max_new_tokens = 64 
    curr_input_ids = input_ids
    total_accepted = 0
    total_drafted = 0
    feedback_buffer = [] 
    feedback_points = []
    
    # [ADDED] Lists to track RL decisions for this sequence
    rl_gammas = []
    rl_rtts = []

    device_type = draft_model.device.type if hasattr(draft_model, "device") else "cuda"
    compute_dtype = torch.bfloat16
    
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=compute_dtype):
        
        # 初始前向传播现在受到 autocast 保护
        draft_cache.forward(curr_input_ids)
        target_cache.forward(curr_input_ids)

        while curr_input_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
            prefix_len = curr_input_ids.shape[1]
            
            current_gamma = gamma
            action_idx = None
            rtt = 0.0
            state = None
            
            if rl_controller is not None and network_sim is not None:
                rtt = network_sim.get_rtt()
                conf = 0.8
                if draft_cache._prob_history is not None:
                    last_probs = norm_logits(draft_cache._prob_history[:, -1, :], temperature, top_k, top_p)
                    conf = torch.max(last_probs).item()
                
                alpha_hist = total_accepted / max(1, total_drafted)
                state = [min(rtt / 500.0, 1.0), conf, alpha_hist]
                
                current_gamma, action_idx = rl_controller.select_action(state, explore=training_mode)
                
                # [ADDED] Record the choices
                rl_gammas.append(current_gamma)
                rl_rtts.append(rtt)
            # -------------------------------------------------------

            # --- 第一阶段：草稿生成 ---
            x = curr_input_ids
            for step in range(current_gamma): # [MODIFIED] Use current_gamma
                if step == 0:
                    logits = draft_cache._prob_history[:, -1, :]
                else:
                    logits = draft_cache.forward(x)
                
                probs = norm_logits(logits, temperature, top_k, top_p)
                next_token = sample(probs)
                x = torch.cat((x, next_token), dim=1)
            
            draft_tokens = x[:, prefix_len:] 
            
            # --- 第二阶段：目标验证 ---
            _ = target_cache.forward(x) 
            check_start_idx = prefix_len - 1
            
            # --- 第三阶段：拒绝采样循环 ---
            n = 0 
            corrected_token = None
            reject_occurred = False
            
            for i in range(current_gamma):
                token_id = draft_tokens[0, i] 
                step_idx = check_start_idx + i
                
                draft_probs = norm_logits(draft_cache._prob_history[:, step_idx, :], temperature, top_k, top_p)
                target_probs = norm_logits(target_cache._prob_history[:, step_idx, :], temperature, top_k, top_p)
                # IMPORTANT: acceptance probability should not be forced to 0 just
                # because `token_id` falls outside top-p/top-k in the target.
                # We keep sampling behavior the same, but compute accept/reject ratio
                # from *untruncated* model probabilities.
                # Compute acceptance ratio in log-space to avoid probability underflow to 0.
                # Acceptance criterion: r <= p_target / p_draft  <=>  log(r) <= log_p_target - log_p_draft
                draft_logits_full = draft_cache._prob_history[:, step_idx, :].float() / max(float(temperature), 1e-8)
                target_logits_full = target_cache._prob_history[:, step_idx, :].float() / max(float(temperature), 1e-8)
                draft_logits_full = torch.nan_to_num(draft_logits_full, nan=-1e9, posinf=1e9, neginf=-1e9)
                target_logits_full = torch.nan_to_num(target_logits_full, nan=-1e9, posinf=1e9, neginf=-1e9)
                draft_log_probs = F.log_softmax(draft_logits_full, dim=-1)
                target_log_probs = F.log_softmax(target_logits_full, dim=-1)
                log_p_draft = draft_log_probs[0, token_id]
                log_p_target = target_log_probs[0, token_id]
                
                r = torch.rand(1, device=curr_input_ids.device)
                
                log_r = torch.log(r)
                log_ratio = log_p_target - log_p_draft
                log_ratio = torch.nan_to_num(log_ratio, nan=-1e9, posinf=1e9, neginf=-1e9)
                if log_r <= log_ratio:
                    n += 1
                    if training_mode:
                        # Logits at index step_idx predict the token at position step_idx + 1
                        # (which corresponds to the drafted token we just accepted).
                        feedback_points.append({
                            "step_idx": int(step_idx),
                            "label": torch.tensor(int(token_id), dtype=torch.long).cpu(),
                            "teacher_logits": target_cache._prob_history[:, step_idx, :].detach().cpu()
                        })
                else:
                    reject_occurred = True
                    # Differential sampling for the corrected token
                    diff_probs = torch.relu(target_probs - draft_probs)
                    if diff_probs.sum() > 1e-6:
                        diff_probs = diff_probs / diff_probs.sum()
                        corrected_token = sample(diff_probs)
                    else:
                        corrected_token = sample(target_probs)
                    if training_mode and corrected_token is not None:
                        # Teacher-corrected token for the rejected position.
                        feedback_points.append({
                            "step_idx": int(step_idx),
                            "label": corrected_token.view(-1).to(dtype=torch.long).cpu(),
                            "teacher_logits": target_cache._prob_history[:, step_idx, :].detach().cpu()
                        })
                    break
            
            # --- 第四和第五阶段：回滚与同步 ---
            # 1. Append accepted tokens
            valid_tokens = draft_tokens[:, :n]
            curr_input_ids = torch.cat((curr_input_ids, valid_tokens), dim=1)
            
            # 2. Handle Rejection / Rollback
            if reject_occurred:
                if corrected_token is not None:
                    curr_input_ids = torch.cat((curr_input_ids, corrected_token), dim=1)
                
                # Rollback caches to match the new sequence length
                draft_cache.rollback(curr_input_ids.shape[1])
                target_cache.rollback(curr_input_ids.shape[1])
            else:
                # Optional: If all drafted tokens are accepted, sample one extra from the target
                extra_token = sample(target_probs)
                curr_input_ids = torch.cat((curr_input_ids, extra_token), dim=1)
            
            total_drafted += current_gamma 
            total_accepted += n

            # Re-sync standard caches for the next step
            draft_cache.forward(curr_input_ids)
            target_cache.forward(curr_input_ids)
            
            # --- [ADDED] RL Reward Calculation & Update ---
            if rl_controller is not None and network_sim is not None and training_mode:
                t_edge = current_gamma * 5.0 
                reward = n / (t_edge + rtt) 
                next_rtt = network_sim.get_rtt()
                next_alpha = total_accepted / max(1, total_drafted)
                next_state = [min(next_rtt / 500.0, 1.0), conf, next_alpha]
                
                # We already wrapped this in torch.enable_grad() in the previous fix
                rl_controller.update(state, action_idx, reward, next_state)

    if training_mode and len(feedback_points) > 0:
        feedback_buffer.append({
            "input_ids": curr_input_ids.clone().detach().cpu(),
            "feedback_points": feedback_points
        })

    alpha = total_accepted / total_drafted if total_drafted > 0 else 0.0
    if previous_mode:
        draft_model.train()

    # [MODIFIED] Calculate averages and return as a 4th dictionary output
    avg_k = sum(rl_gammas) / len(rl_gammas) if rl_gammas else gamma
    avg_rtt = sum(rl_rtts) / len(rl_rtts) if rl_rtts else 0.0
    rl_stats = {"avg_k": avg_k, "avg_rtt": avg_rtt}

    return curr_input_ids, alpha, feedback_buffer, rl_stats
