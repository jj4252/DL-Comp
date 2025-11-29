from functools import partial
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOHead(nn.Module):
    """
    Projection head for DINO
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # For DINO, we only use the CLS token (first token)
        # x: [B, N, D] where N = 1 + num_patches -> extract x: [B, D]
        if x.dim() == 3:
            x = x[:, 0]  # Extract CLS token
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Wrapper to handle multiple crops for self-supervised learning
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, return_backbone_feat=False, masked_indices=None):
        # Handle multi-crop input
        if not isinstance(x, list):
            x = [x]

        # Group crops by resolution for efficient processing
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True
            )[1], 0
        )

        start_idx = 0
        output = []
        for end_idx in idx_crops:
            # Get the crops for this resolution group
            crops_in_group = x[start_idx:end_idx]
            num_crops_in_group = len(crops_in_group)

            # Concatenate crops in this group
            _out = self.backbone(torch.cat(crops_in_group))

            if return_backbone_feat:
                # Split output back into individual crops
                # _out shape: [batch_size * num_crops, num_tokens, embed_dim]
                batch_size = crops_in_group[0].shape[0]
                _out_split = _out.reshape(num_crops_in_group, batch_size, *_out.shape[1:])
                for i in range(num_crops_in_group):
                    output.append(_out_split[i])
            else:
                # Pass through head
                _out = self.head(_out)

                # Split output back into individual crops
                # _out is a tensor [batch_size * num_crops, out_dim]
                batch_size = crops_in_group[0].shape[0]
                _out_split = _out.reshape(num_crops_in_group, batch_size, -1)
                for i in range(num_crops_in_group):
                    output.append(_out_split[i])

            start_idx = end_idx

        return output


def create_vision_transformer(cfg):
    """
    Create Vision Transformer backbone using timm
    """
    vit_cfg = cfg.model.vit

    # Create model using timm
    model = timm.create_model(
        cfg.model.architecture,
        pretrained=False,
        num_classes=0,  # Remove classification head
        img_size=vit_cfg.image_size,
        patch_size=vit_cfg.patch_size,
        embed_dim=vit_cfg.embed_dim,
        depth=vit_cfg.depth,
        num_heads=vit_cfg.num_heads,
        mlp_ratio=vit_cfg.mlp_ratio,
        drop_path_rate=vit_cfg.drop_path_rate,
    )

    # Assert the model is a timm Vision Transformer
    assert hasattr(model, "blocks"), (
        "Expected a timm transformer model with `.blocks`, but the model has no `.blocks` attribute. "
        "Are you sure it comes from timm.create_model(...) ?"
    )

    # Enable Flash Attention (fused attention) for all attention blocks
    # This uses PyTorch's scaled_dot_product_attention which can leverage Flash Attention
    flash_attn_enabled = False
    for block in model.blocks:
        if hasattr(block, 'attn') and hasattr(block.attn, 'fused_attn'):
            block.attn.fused_attn = True
            flash_attn_enabled = True

    if flash_attn_enabled:
        print(f"✓ Flash Attention (fused_attn) enabled for {len(model.blocks)} transformer blocks")

    # Modify to return all tokens (CLS + patches)
    original_forward = model.forward_features

    def new_forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x  # Return all tokens

    model.forward_features = partial(new_forward, model)
    model.forward = model.forward_features

    return model


def create_dino_model(cfg: DictConfig):
    """
    Create DINOv2 model (student and teacher)
    """
    # Create backbone
    backbone_s = create_vision_transformer(cfg)
    backbone_t = create_vision_transformer(cfg)

    # Create head (only for CLS token in pure DINO)
    head_s = DINOHead(
        in_dim=cfg.model.vit.embed_dim,
        out_dim=cfg.model.dino.out_dim,
        norm_last_layer=cfg.model.dino.norm_last_layer,
        bottleneck_dim=cfg.model.dino.bottleneck_dim,
    )
    head_t = DINOHead(
        in_dim=cfg.model.vit.embed_dim,
        out_dim=cfg.model.dino.out_dim,
        norm_last_layer=cfg.model.dino.norm_last_layer,
        bottleneck_dim=cfg.model.dino.bottleneck_dim,
    )

    # Wrap with multi-crop wrapper
    student = MultiCropWrapper(backbone_s, head_s)
    teacher = MultiCropWrapper(backbone_t, head_t)

    # Teacher starts with same weights as student
    teacher.load_state_dict(student.state_dict())

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    """
    Update teacher with exponential moving average of student weights
    """
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)


class DINOLoss(nn.Module):
    """
    DINO loss with temperature scaling and centering
    """
    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Temperature schedule
        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ])

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_output and teacher_output are lists of tensors (one per crop)

        # Get teacher temperature for this epoch
        temp = self.teacher_temp_schedule[epoch]

        # Process student outputs
        student_out = [s / self.student_temp for s in student_output]

        # Process teacher outputs with centering and sharpening
        teacher_out = [F.softmax((t - self.center) / temp, dim=-1).detach() for t in teacher_output]

        # Ensure we have outputs to compute loss
        if len(student_out) == 0 or len(teacher_out) == 0:
            raise ValueError(f"Empty outputs: student_out={len(student_out)}, teacher_out={len(teacher_out)}")

        # For multi-crop, we need at least 2 crops to compute meaningful loss
        if len(student_out) < 2:
            raise ValueError(f"Need at least 2 crops for multi-crop training, got {len(student_out)} student crops")

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip same crop
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class ProjectionHead(nn.Module):
    """
    Projection head for MoCo V3
    MLP: embed_dim -> hidden_dim -> proj_dim
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        # For ViT, extract CLS token if needed
        if x.dim() == 3:
            x = x[:, 0]  # Extract CLS token
        return self.mlp(x)


class MoCoModel(nn.Module):
    """
    MoCo V3 model with momentum encoder and negative queue
    """
    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        embed_dim: int,
        proj_dim: int,
        proj_hidden_dim: int,
        queue_size: int = 65536,
        momentum: float = 0.99,
        temperature: float = 0.2,
        use_queue: bool = True,
    ):
        super().__init__()
        
        # Query encoder (trainable)
        self.encoder_q = timm.create_model(
            backbone_name,
            img_size=image_size,
            num_classes=0,
            pretrained=False,
            global_pool="",  # Return all tokens for ViT
        )
        
        # Key encoder (momentum encoder)
        self.encoder_k = timm.create_model(
            backbone_name,
            img_size=image_size,
            num_classes=0,
            pretrained=False,
            global_pool="",  # Return all tokens for ViT
        )
        
        # Projection heads
        if proj_hidden_dim <= 0:
            proj_hidden_dim = embed_dim
        
        self.proj_q = ProjectionHead(embed_dim, proj_hidden_dim, proj_dim)
        self.proj_k = ProjectionHead(embed_dim, proj_hidden_dim, proj_dim)
        
        # Initialize encoder_k with encoder_q weights
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Initialize projection heads
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # MoCo queue: [proj_dim, queue_size]
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        self.proj_dim = proj_dim
        self.use_queue = use_queue
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """EMA update: param_k = m * param_k + (1-m) * param_q"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        keys = F.normalize(keys, dim=-1)
        
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :end - self.queue_size] = keys[first:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def _extract_features(self, encoder, images):
        """Extract CLS token features from ViT"""
        features = encoder.forward_features(images)
        if isinstance(features, torch.Tensor):
            feat = features[:, 0]  # CLS token
        else:
            feat = features.get('x', features.get('tokens', None))[:, 0]
        return feat
    
    def forward_moco(self, im_q: torch.Tensor, im_k: torch.Tensor, 
                     im_local1: torch.Tensor = None, im_local2: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for MoCo V3 with optional multi-crop support
        
        Args:
            im_q: Query images (global crop 1) [B, C, H, W]
            im_k: Key images (global crop 2) [B, C, H, W]
            im_local1: Optional local crop 1 [B, C, H, W]
            im_local2: Optional local crop 2 [B, C, H, W]
        
        Returns:
            Contrastive loss (InfoNCE)
        """
        batch_size = im_q.shape[0]
        
        # Key branch (momentum encoder)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_features = self._extract_features(self.encoder_k, im_k)
            k = self.proj_k(k_features)
            k = F.normalize(k, dim=-1)
        
        # Count crops
        num_crops = 1
        if im_local1 is not None:
            num_crops += 1
        if im_local2 is not None:
            num_crops += 1
        
        # Global crop 1 (query) vs Global crop 2 (key)
        q_features = self._extract_features(self.encoder_q, im_q)
        q = self.proj_q(q_features)
        q = F.normalize(q, dim=-1)
        
        loss_global = self._compute_contrastive_loss(q, k, batch_size)
        total_loss = loss_global / num_crops
        
        del q_features, q
        
        # Local crops (if provided)
        if im_local1 is not None:
            q_local1_features = self._extract_features(self.encoder_q, im_local1)
            q_local1 = self.proj_q(q_local1_features)
            q_local1 = F.normalize(q_local1, dim=-1)
            loss_local1 = self._compute_contrastive_loss(q_local1, k, batch_size)
            total_loss = total_loss + loss_local1 / num_crops
            del q_local1_features, q_local1, loss_local1
        
        if im_local2 is not None:
            q_local2_features = self._extract_features(self.encoder_q, im_local2)
            q_local2 = self.proj_q(q_local2_features)
            q_local2 = F.normalize(q_local2, dim=-1)
            loss_local2 = self._compute_contrastive_loss(q_local2, k, batch_size)
            total_loss = total_loss + loss_local2 / num_crops
            del q_local2_features, q_local2, loss_local2
        
        # Update queue
        if self.use_queue:
            with torch.no_grad():
                self._dequeue_and_enqueue(k)
        
        return total_loss
    
    def _compute_contrastive_loss(self, q: torch.Tensor, k: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute InfoNCE contrastive loss"""
        # Positive: q · k (same index)
        logits_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(-1)  # [B, 1]
        
        if self.use_queue:
            # Negative: q · queue
            logits_neg = torch.einsum("bd,dk->bk", [q, self.queue])  # [B, queue_size]
            logits = torch.cat([logits_pos, logits_neg], dim=1)  # [B, 1 + queue_size]
            targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        else:
            # Batch-only contrastive learning
            logits_all = torch.einsum("bd,cd->bc", [q, k])  # [B, B]
            logits = logits_all
            targets = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        
        # Apply temperature
        logits = logits / self.temperature
        
        # InfoNCE loss
        loss = F.cross_entropy(logits, targets)
        return loss


def create_moco_model(cfg: DictConfig):
    """
    Create MoCo V3 model (query encoder and key encoder)
    """
    # Create backbone using same function as DINO
    backbone_q = create_vision_transformer(cfg)
    backbone_k = create_vision_transformer(cfg)
    
    # Get embed_dim from config
    embed_dim = cfg.model.vit.embed_dim
    
    # MoCo V3 configuration
    moco_cfg = cfg.model.moco
    proj_dim = moco_cfg.proj_dim
    proj_hidden_dim = moco_cfg.get("proj_hidden_dim", 0)
    if proj_hidden_dim <= 0:
        proj_hidden_dim = embed_dim
    
    # Create projection heads
    proj_q = ProjectionHead(embed_dim, proj_hidden_dim, proj_dim)
    proj_k = ProjectionHead(embed_dim, proj_hidden_dim, proj_dim)
    
    # Create MoCo model
    model = MoCoModel(
        backbone_name=cfg.model.architecture,
        image_size=cfg.model.vit.image_size,
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        proj_hidden_dim=proj_hidden_dim,
        queue_size=moco_cfg.queue_size,
        momentum=moco_cfg.momentum,
        temperature=moco_cfg.temperature,
        use_queue=moco_cfg.get("use_queue", True),
    )
    
    # Replace the encoders with our created backbones
    model.encoder_q = backbone_q
    model.encoder_k = backbone_k
    model.proj_q = proj_q
    model.proj_k = proj_k
    
    # Initialize encoder_k with encoder_q weights
    for param_q, param_k in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
    
    # Initialize projection heads
    for param_q, param_k in zip(model.proj_q.parameters(), model.proj_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
    
    return model

