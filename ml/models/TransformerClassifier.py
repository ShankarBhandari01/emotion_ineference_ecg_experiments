import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RelativePositionalEncoding(nn.Module):
    """More effective positional encoding for sequences"""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_embedding[:seq_len].unsqueeze(0)


class MultiScaleAttention(nn.Module):
    """Enhanced attention with multiple scales"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0

        self.nhead = nhead
        self.d_model = d_model
        self.d_k = d_model // nhead

        # Multi-scale projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Local attention for fine-grained patterns
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Standard multi-head attention
        q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        global_attn = torch.matmul(attn_weights, v)
        global_attn = global_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        global_attn = self.w_o(global_attn)

        # Local convolution attention
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        local_attn = self.local_conv(x_conv).transpose(1, 2)  # Back to (batch, seq_len, d_model)

        # Combine global and local attention
        combined = global_attn + 0.3 * local_attn  # Weight local attention less
        return self.layer_norm(x + self.dropout(combined))


class EnhancedTransformerLayer(nn.Module):
    """Improved transformer layer with better regularization"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, drop_path=0.0):
        super().__init__()

        self.attention = MultiScaleAttention(d_model, nhead, dropout)

        # Enhanced feed-forward with gating
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim_feedforward, d_model)
        )

        # Gating mechanism for better information flow
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        # Attention with residual
        attn_out = self.attention(x, mask)
        x = x + self.drop_path(attn_out)
        x = self.norm1(x)

        # Gated feed-forward with residual
        ffn_out = self.ffn(x)
        gate_weights = self.gate(x)
        ffn_out = ffn_out * gate_weights  # Gating
        x = x + self.drop_path(self.dropout(ffn_out))
        x = self.norm2(x)

        return x


class DropPath(nn.Module):
    """Stochastic Depth for regularization"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class AttentionPooling(nn.Module):
    """Learnable attention pooling for better feature aggregation"""

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, attn_weights = self.multihead_attn(query, x, x)
        return self.layer_norm(attn_output.squeeze(1)), attn_weights


class EnhancedClassificationHead(nn.Module):
    """Advanced classification head with uncertainty estimation"""

    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(d_model // 4, num_classes)
        )

        # Uncertainty estimation branch
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )

    def forward(self, x):
        logits = self.head(x)
        uncertainty = self.uncertainty_head(x)
        return logits, uncertainty


class OptimizedECGEmotionTransformer(nn.Module):
    """
    State-of-the-art transformer for ECG emotion classification
    """

    def __init__(self,
                 input_dim=128,
                 d_model=256,  # Increased from 128
                 nhead=8,
                 num_layers=6,  # Increased from 2
                 num_tasks=3,
                 num_classes=3,
                 max_len=500,
                 dropout=0.15,
                 drop_path_rate=0.1):
        super().__init__()

        assert d_model % nhead == 0

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.num_classes = num_classes

        # Enhanced input projection with residual
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model)
        )
        self.input_norm = nn.LayerNorm(d_model)

        # Better positional encoding
        self.pos_encoder = RelativePositionalEncoding(d_model, max_len)

        # Progressive dropout rates (higher for later layers)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Enhanced transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 6,  # Increased from 4x
                dropout=dropout,
                drop_path=drop_path_rates[i]
            ) for i in range(num_layers)
        ])

        # Multiple pooling strategies
        self.attention_pooling = AttentionPooling(d_model, num_heads=4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Task-specific enhanced heads
        self.task_heads = nn.ModuleList([
            EnhancedClassificationHead(d_model * 2, num_classes, dropout)  # *2 for concatenated features
            for _ in range(num_tasks)
        ])

        # Task importance weighting (learnable)
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            logits: (batch_size, num_tasks, num_classes)
            uncertainties: (batch_size, num_tasks) if training
        """
        batch_size = x.size(0)

        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        elif x.dim() != 3:
            raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")

        # Enhanced input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Dual pooling strategy
        cls_output = x[:, 0]  # CLS token
        pooled_output, attention_weights = self.attention_pooling(x[:, 1:])  # Skip CLS for attention pooling

        # Combine features
        combined_features = torch.cat([cls_output, pooled_output], dim=-1)
        fused_features = self.fusion_layer(combined_features)

        # Apply task-specific heads with uncertainty
        task_logits = []
        task_uncertainties = []

        for i, head in enumerate(self.task_heads):
            logits, uncertainty = head(combined_features)
            task_logits.append(logits)
            task_uncertainties.append(uncertainty.squeeze(-1))

        # Stack outputs
        logits = torch.stack(task_logits, dim=1)  # (batch_size, num_tasks, num_classes)
        uncertainties = torch.stack(task_uncertainties, dim=1)  # (batch_size, num_tasks)

        if return_attention:
            return logits, uncertainties, attention_weights
        return logits, uncertainties


# Advanced loss function with uncertainty weighting
class UncertaintyWeightedLoss(nn.Module):
    """Advanced loss with uncertainty weighting and label smoothing"""

    def __init__(self, num_tasks=3, label_smoothing=0.1, uncertainty_weight=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.label_smoothing = label_smoothing
        self.uncertainty_weight = uncertainty_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, uncertainties, targets, task_weights=None):
        """
        Args:
            logits: (batch_size, num_tasks, num_classes)
            uncertainties: (batch_size, num_tasks)
            targets: (batch_size, num_tasks)
            task_weights: (num_tasks,) learnable task importance
        """
        total_loss = 0.0
        task_losses = []

        for task_idx in range(self.num_tasks):
            task_logits = logits[:, task_idx]  # (batch_size, num_classes)
            task_targets = targets[:, task_idx]  # (batch_size,)
            task_uncertainty = uncertainties[:, task_idx]  # (batch_size,)

            # Standard cross-entropy loss
            ce_loss = self.ce_loss(task_logits, task_targets)

            # Uncertainty-weighted loss (lower uncertainty = higher confidence = higher weight)
            uncertainty_weights = 1.0 / (1.0 + task_uncertainty.mean())
            weighted_loss = ce_loss * uncertainty_weights

            # Add uncertainty regularization (encourage confident predictions)
            uncertainty_reg = task_uncertainty.mean() * self.uncertainty_weight

            task_loss = weighted_loss + uncertainty_reg

            # Apply learnable task importance weights
            if task_weights is not None:
                task_loss = task_loss * torch.softmax(task_weights, dim=0)[task_idx]

            task_losses.append(task_loss)
            total_loss += task_loss

        return total_loss, task_losses


# Complete training configuration
class TrainingConfig:
    """Optimal training configuration"""

    # Model hyperparameters
    d_model = 256
    num_layers = 6
    nhead = 8
    dropout = 0.15
    drop_path_rate = 0.1

    # Training hyperparameters
    learning_rate = 5e-5
    weight_decay = 0.01
    batch_size = 32
    warmup_steps = 1000
    max_epochs = 100
    gradient_clip = 1.0

    # Loss configuration
    label_smoothing = 0.1
    uncertainty_weight = 0.1

    # Data augmentation
    noise_std = 0.01
    mixup_alpha = 0.2

    @staticmethod
    def get_optimizer(model):
        """Get optimized optimizer with layer-wise learning rates"""
        # Different learning rates for different parts
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'input_proj' in n],
             'lr': TrainingConfig.learning_rate * 0.5},
            {'params': [p for n, p in model.named_parameters() if 'layers' in n], 'lr': TrainingConfig.learning_rate},
            {'params': [p for n, p in model.named_parameters() if 'task_heads' in n],
             'lr': TrainingConfig.learning_rate * 2.0},
            {'params': [p for n, p in model.named_parameters() if
                        not any(x in n for x in ['input_proj', 'layers', 'task_heads'])],
             'lr': TrainingConfig.learning_rate}
        ]
        return torch.optim.AdamW(param_groups, weight_decay=TrainingConfig.weight_decay)

    @staticmethod
    def get_scheduler(optimizer, num_training_steps):
        """Get learning rate scheduler with warmup"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=TrainingConfig.warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - TrainingConfig.warmup_steps,
            eta_min=1e-7
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[TrainingConfig.warmup_steps]
        )
