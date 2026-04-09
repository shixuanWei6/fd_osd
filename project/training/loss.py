# fileName: training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    def __init__(self, alpha_kd=0.5, lambda_ortho=0.1, temp=2.0):
        super().__init__()
        self.alpha_kd = alpha_kd
        self.lambda_ortho = lambda_ortho
        self.temp = temp

    def forward(self, student_logits, teacher_logits, labels, model):
        # [Fix] Cast logits to float32 for numerical stability
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()
        
        # 1. SFT Loss
        loss_sft = F.cross_entropy(student_logits, labels)
        
        # 2. KD Loss
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        s_dist = F.log_softmax(student_logits[:, :min_vocab] / self.temp, dim=-1)
        t_dist = F.softmax(teacher_logits[:, :min_vocab] / self.temp, dim=-1)
        loss_kd = F.kl_div(s_dist, t_dist, reduction='batchmean') # * (self.temp ** 2)
        
        # 3. Orthogonal Loss (Mutual Decoupling)
        loss_ortho = 0.0
        adapter_pairs = 0
        
        # Iterate through parameters to find Local and Global pairs
        params_dict = dict(model.named_parameters())
        for name, param in params_dict.items():
            if "lora_A" in name and "local_adapter" in name:
                global_name = name.replace("local_adapter", "global_adapter")
                if global_name in params_dict:
                    global_param = params_dict[global_name]
                    
                    # Force both vectors to float32 and align their device first
                    local_flat = param.view(-1).float()
                    global_flat = global_param.detach().view(-1).float().to(local_flat.device)
                    
                    min_len = min(local_flat.numel(), global_flat.numel())
                    if min_len == 0:
                        continue
                    local_flat = local_flat[:min_len]
                    global_flat = global_flat[:min_len]
                    
                    # Numerical stability is now guaranteed in FP32
                    sim = F.cosine_similarity(local_flat, global_flat, dim=0, eps=1e-8)
                    loss_ortho += torch.abs(sim)
                    adapter_pairs += 1
                    
        if adapter_pairs > 0:
            loss_ortho = loss_ortho / adapter_pairs
        
        total_loss = (1 - self.alpha_kd) * loss_sft + self.alpha_kd * loss_kd + self.lambda_ortho * loss_ortho
        return total_loss