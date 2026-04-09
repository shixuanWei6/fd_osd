import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from .loss import CompositeLoss
class OnlineTrainer:
    def __init__(self, model, optimizer, buffer, temp=1.0, alpha_kd=0.8):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.device = next(model.parameters()).device
        self.temp = temp
        self.alpha_kd = alpha_kd 
        
        self.task_centroid = None 
        self.momentum = 0.99
        self.use_amp = (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.criterion = CompositeLoss(alpha_kd=alpha_kd, lambda_ortho=0.1, temp=temp)
        
    def _get_task_embedding(self, input_ids):
        with torch.no_grad():
            prompt_ids = input_ids[:, :min(32, input_ids.shape[1])]
            embeds = self.model.get_input_embeddings()(prompt_ids)
            return embeds.mean(dim=1)
        
    def train_step(self, batch_size=4):
        samples = self.buffer.sample(batch_size)
        if not samples: return 0.0
        
        self.model.train()
        self.optimizer.zero_grad() 
        
        batch_loss = 0
        valid_samples = 0
        
        for sample in samples:
            input_ids = sample['input_ids'].to(self.device)
            feedback_points = sample['feedback_points']
            
            if not feedback_points:
                continue
                
            # --- Task Weighting ---
            current_embed = self._get_task_embedding(input_ids)
            if self.task_centroid is None:
                self.task_centroid = current_embed
                weight = 1.0
            else:
                # Numerical stability: avoid NaNs when either vector has ~0 norm.
                sim = F.cosine_similarity(current_embed, self.task_centroid, eps=1e-8)
                weight = torch.clamp(sim, min=0.2).mean().item() 
                self.task_centroid = self.momentum * self.task_centroid + (1 - self.momentum) * current_embed

            # --- Forward Pass (Sequence Level) ---
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_amp
            ):
                outputs = self.model(input_ids)
                
                step_indices = [fp['step_idx'] for fp in feedback_points]
                student_logits = outputs.logits[0, step_indices, :] 
                
                target_labels = torch.cat([fp['label'].view(-1) for fp in feedback_points]).to(self.device)
                teacher_logits = torch.cat([fp['teacher_logits'] for fp in feedback_points]).to(self.device)

                # --- Loss Calculation ---
                current_sample_loss = self.criterion(student_logits, teacher_logits, target_labels, self.model)
                
                # 🛑 SAFETY CHECK: Catch NaNs before they poison the batch
                if torch.isnan(current_sample_loss) or torch.isinf(current_sample_loss):
                    print(f"⚠️ Warning: NaN/Inf loss detected for a sample. Skipping...")
                    continue # Skip this sample entirely
                    
                batch_loss += current_sample_loss * weight
                valid_samples += 1 # Only increment if the loss was valid and finite
                
        if valid_samples == 0:
            print("⚠️ Warning: No valid samples in batch after NaN filtering.")
            return 0.0
            
        avg_loss = batch_loss / valid_samples
        
        # 🛑 SECOND SAFETY CHECK: Ensure avg_loss is completely safe before backward
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print("⚠️ Warning: Average loss is NaN/Inf. Aborting backward pass.")
            self.optimizer.zero_grad()
            return 0.0
        
        # --- Backpropagation ---
        self.scaler.scale(avg_loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return avg_loss.item()

def set_trainable_adapters(model, trainable_names=["local_adapter"]):
    """
    Enables gradients only for specified Adapters, freezing others.
    Casts trainable parameters to float32 for GradScaler compatibility.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = False # Turn off all initially
            
    for adapter_name in trainable_names:
        for name, param in model.named_parameters():
            if adapter_name in name:
                param.requires_grad = True
                
    # [Fix] Cast all trainable parameters to float32
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params restricted to: {len(trainable_params)} layers in {trainable_names} (Cast to FP32)")