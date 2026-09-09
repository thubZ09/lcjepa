from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from core.config import Config
from core.data.masking import sample_block_masks
from core.models.embeddings import ObservationEmbedding
from core.models.encoder import TransformerEncoder
from core.models.predictor import JEPAPredictor, LatentRollout

@dataclass
class LCJEPAOutput:
    loss: Tensor
    pred_loss: Tensor
    rollout_loss: Tensor
    vic_var: Tensor
    vic_cov: Tensor

def vicreg_terms(z: Tensor, var_target: float, eps: float = 1e-4) -> tuple[Tensor, Tensor]:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(var_target - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag**2).sum() / d
    return var_loss, cov_loss

class LCJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.embed = ObservationEmbedding(m.d_model, m.time_embed, m.n_fourier)
        self.context_encoder = TransformerEncoder(
            m.d_model, m.n_heads, m.enc_depth, m.mlp_ratio, m.dropout
        )
        self.predictor = JEPAPredictor(m.d_model, m.n_heads, m.pred_depth, m.mlp_ratio, m.dropout)
        self.rollout = LatentRollout(m.d_model, m.n_heads, m.rollout_depth, m.mlp_ratio, m.dropout)
        self.recon_head = nn.Linear(m.d_model, 1)
        self.target_embed = copy.deepcopy(self.embed)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_embed.parameters():
            p.requires_grad_(False)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for tp, sp in zip(self.target_embed.parameters(), self.embed.parameters()):
            tp.mul_(momentum).add_(sp, alpha=1 - momentum)
        for tp, sp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.mul_(momentum).add_(sp, alpha=1 - momentum)
    def _embed_full(self, batch: dict[str, Tensor], target: bool) -> Tensor:
        embed = self.target_embed if target else self.embed
        return embed(batch["dt"], batch["mag"], batch["mag_err"])
    def forward(self, batch: dict[str, Tensor]) -> LCJEPAOutput:
        cfg = self.cfg
        device = batch["dt"].device
        b, seq_len = batch["dt"].shape
        ctx_tokens_full = self._embed_full(batch, target=False) 
        if cfg.loss.objective == "recon":
            tgt_latents_full = None 
        else:
            with torch.no_grad():
                tgt_tokens_full = self._embed_full(batch, target=True)
                tgt_latents_full = self.target_encoder(tgt_tokens_full)  
        context_idx, target_blocks = sample_block_masks(
            seq_len,
            cfg.mask.n_target_blocks,
            cfg.mask.target_block_frac,
            cfg.mask.context_keep_frac,
        )
        context_idx = context_idx.to(device)
        ctx_tokens = ctx_tokens_full[:, context_idx, :]
        ctx_latents = self.context_encoder(ctx_tokens)    
        if cfg.loss.objective == "recon":
            recon_loss = ctx_latents.new_zeros(())
            for block in target_blocks:
                block = block.to(device)
                t_embed = self.embed.time(batch["dt"][:, block])
                pred_latent = self.predictor(ctx_latents, t_embed)   
                pred_mag = self.recon_head(pred_latent).squeeze(-1)   
                recon_loss = recon_loss + F.smooth_l1_loss(pred_mag, batch["mag"][:, block])
            recon_loss = recon_loss / max(len(target_blocks), 1)
            zero = ctx_latents.new_zeros(())
            return LCJEPAOutput(recon_loss, recon_loss, zero, zero, zero)
        pred_loss = ctx_latents.new_zeros(())
        for block in target_blocks:
            block = block.to(device)
            t_embed = self.embed.time(batch["dt"][:, block])  
            pred = self.predictor(ctx_latents, t_embed)        
            tgt = tgt_latents_full[:, block, :]              
            pred_loss = pred_loss + F.smooth_l1_loss(pred, tgt)
        pred_loss = pred_loss / max(len(target_blocks), 1)
        if cfg.loss.rollout_weight > 0:
            rolled = self.rollout(tgt_latents_full[:, :-1, :].contiguous())
            rollout_loss = F.smooth_l1_loss(
                rolled.contiguous(), tgt_latents_full[:, 1:, :].contiguous()
            )
        else:
            rollout_loss = ctx_latents.new_zeros(())
        var_loss, cov_loss = vicreg_terms(
            ctx_latents.reshape(-1, ctx_latents.size(-1)),
            cfg.loss.vic_var_target,
        )
        loss = (
            cfg.loss.pred_weight * pred_loss
            + cfg.loss.rollout_weight * rollout_loss
            + cfg.loss.vic_var_weight * var_loss
            + cfg.loss.vic_cov_weight * cov_loss
        )
        return LCJEPAOutput(loss, pred_loss, rollout_loss, var_loss, cov_loss)

    @torch.no_grad()
    def embed_sequence(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self.embed(batch["dt"], batch["mag"], batch["mag_err"])
        latents = self.context_encoder(tokens)          
        mean = latents.mean(dim=1)                        
        std = latents.std(dim=1)                         
        return torch.cat([mean, std], dim=-1)              

    @torch.no_grad()
    def rollout_surprise(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self.target_embed(batch["dt"], batch["mag"], batch["mag_err"])
        latents = self.target_encoder(tokens)
        pred = self.rollout(latents[:, :-1, :])
        err = F.smooth_l1_loss(pred, latents[:, 1:, :], reduction="none")
        return err.mean(dim=(1, 2))  