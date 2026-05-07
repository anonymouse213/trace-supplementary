import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalContrastiveLoss(nn.Module):
    """

      z1, z2: (B, T, H)

    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5):
        super().__init__()
        self.temp  = temperature
        self.alpha = alpha

    def temporal_cl_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B, T, H = z1.shape
        z1_flat = F.normalize(z1.reshape(B * T, H), dim=-1)
        z2_flat = F.normalize(z2.reshape(B * T, H), dim=-1)
        sim    = torch.mm(z1_flat, z2_flat.T) / self.temp
        labels = torch.arange(B * T, device=z1.device)
        loss   = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2.0

    def instance_cl_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        v1 = F.normalize(z1.max(dim=1).values, dim=-1)
        v2 = F.normalize(z2.max(dim=1).values, dim=-1)
        B  = v1.shape[0]
        z_cat = torch.cat([v1, v2], dim=0)
        sim   = torch.mm(z_cat, z_cat.T) / self.temp
        mask  = torch.eye(2 * B, device=z1.device).bool()
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(0, B,     device=z1.device),
        ])
        return F.cross_entropy(sim, labels)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        l_temp = self.temporal_cl_loss(z1, z2)
        l_inst = self.instance_cl_loss(z1, z2)
        return self.alpha * l_temp + (1 - self.alpha) * l_inst


class SoftSupConLoss(nn.Module):
    """

    Positive weight:
      w_ij = exp(-|sofa_i - sofa_j|^2 / (2 * sigma^2))
    """

    def __init__(self, temperature: float = 0.2, sofa_sigma: float = 0.3):
        super().__init__()
        self.temp       = temperature
        self.sofa_sigma = sofa_sigma

    def forward(
        self,
        embeddings: torch.Tensor,
        labels:     torch.Tensor,
        sofa:       torch.Tensor,
    ) -> torch.Tensor:
        B      = embeddings.shape[0]
        device = embeddings.device

        sim = torch.mm(embeddings, embeddings.T) / self.temp
        eye = torch.eye(B, device=device).bool()
        sim = sim.masked_fill(eye, -1e9)

        sofa_diff = (sofa.unsqueeze(0) - sofa.unsqueeze(1)) ** 2
        w         = torch.exp(-sofa_diff / (2 * self.sofa_sigma ** 2))
        w         = w.masked_fill(eye, 0.0)
        w_sum     = w.sum(dim=1, keepdim=True).clamp(min=1e-8)

        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob  = sim - log_denom
        loss      = -(w * log_prob).sum(dim=1, keepdim=True) / w_sum
        return loss.mean()


class IntraSOFAOutcomeLoss(nn.Module):
    """


    Args:
    """

    def __init__(
        self,
        temperature: float = 0.1,
        n_bins:      int   = 5,
        min_pairs:   int   = 2,
    ):
        super().__init__()
        self.temp      = temperature
        self.n_bins    = n_bins
        self.min_pairs = min_pairs

    def forward(
        self,
        embeddings: torch.Tensor,
        labels:     torch.Tensor,
        sofa:       torch.Tensor,
    ) -> torch.Tensor:
        B      = embeddings.shape[0]
        device = embeddings.device

        sofa_np     = sofa.detach().cpu()
        percentiles = torch.linspace(0, 100, self.n_bins + 1).tolist()
        boundaries  = torch.tensor(
            [torch.quantile(sofa_np, p / 100.0).item() for p in percentiles],
            device=device, dtype=sofa.dtype
        )
        bins = torch.bucketize(sofa, boundaries[1:-1])

        same_bin     = (bins.unsqueeze(0) == bins.unsqueeze(1))
        same_outcome = (labels.unsqueeze(0) == labels.unsqueeze(1))
        eye          = torch.eye(B, device=device).bool()

        pos_mask = same_bin & same_outcome & ~eye
        denom_mask = same_bin & ~eye

        sim = torch.mm(embeddings, embeddings.T) / self.temp
        sim_masked = sim.masked_fill(~denom_mask, -1e9)

        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)
        log_prob  = sim - log_denom

        n_pos = pos_mask.float().sum(dim=1)
        valid = (n_pos >= 1) & (denom_mask.float().sum(dim=1) >= self.min_pairs)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1)
        loss_per_row = -pos_log_prob / n_pos.clamp(min=1)
        loss         = loss_per_row[valid].mean()

        return loss


class UniformityLoss(nn.Module):
    """

    L_uniform = log E[exp(-t * ||z_i - z_j||²)]
              = log mean(exp(-t * 2 * (1 - cosine_sim(z_i, z_j))))


    Args:
        t : Gaussian kernel bandwidth (default: 2.0)
    """

    def __init__(self, t: float = 2.0, n_sample: int = None):
        super().__init__()
        self.t        = t
        self.n_sample = n_sample

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) L2-normalized
        """
        B      = embeddings.shape[0]
        device = embeddings.device

        sim = torch.mm(embeddings, embeddings.T)

        triu_i, triu_j = torch.triu_indices(B, B, offset=1, device=device)

        if self.n_sample is not None and len(triu_i) > self.n_sample:
            sel    = torch.randperm(len(triu_i), device=device)[:self.n_sample]
            triu_i = triu_i[sel]
            triu_j = triu_j[sel]

        sim_pairs = sim[triu_i, triu_j]

        sq_dist = 2.0 * (1.0 - sim_pairs)

        loss = torch.logsumexp(-self.t * sq_dist, dim=0) \
               - torch.log(torch.tensor(len(sq_dist), dtype=torch.float32,
                                        device=device))
        return loss


class MortalityClassificationLoss(nn.Module):
    """
    embedding → 2-class BCE.
    """

    def __init__(self, embed_dim: int = 256, pos_weight_val: float = 9.0):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, 1)
        pos_w = torch.tensor([pos_weight_val])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        logits = self.classifier(embeddings).squeeze(-1)
        return self.bce(logits, labels.float())