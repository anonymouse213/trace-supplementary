import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """Residual dilated causal conv block."""

    def __init__(self, in_channels: int, hidden: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3,
                      dilation=dilation, padding=dilation),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden, in_channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.norm((out + residual).transpose(1, 2)).transpose(1, 2)
        return out


class TemporalEncoder(nn.Module):
    """
    (B, T, D_in) → per-timestep embedding (B, T, hidden_dim)
    → [CLS] pooling → (B, hidden_dim)

    Architecture:
        Linear projection → N × DilatedConvBlock (dilation 1,2,4,8,...) → mean pool
    """

    def __init__(
        self,
        input_dim: int   = 183,
        hidden_dim: int  = 256,
        n_layers: int    = 6,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        dilations = [2 ** i for i in range(n_layers)]
        self.conv_blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, hidden_dim * 2, d)
            for d in dilations
        ])
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, return_seq: bool = False):
        """
        x: (B, T, D_in)
        """
        h = self.input_proj(x)
        h = self.dropout(h)
        h = h.transpose(1, 2)

        for block in self.conv_blocks:
            h = block(h)

        h = h.transpose(1, 2)

        if return_seq:
            return h
        else:
            return h.mean(dim=1)


class StaticEncoder(nn.Module):
    """
    static_numeric (B, 33) + static_categoric (B, 2) → (B, static_dim)
    """

    def __init__(
        self,
        numeric_dim: int   = 33,
        gender_emb: int    = 8,
        icu_type_emb: int  = 16,
        out_dim: int       = 64,
    ):
        super().__init__()
        self.gender_emb   = nn.Embedding(2, gender_emb)
        self.icu_type_emb = nn.Embedding(6, icu_type_emb)

        in_dim = numeric_dim + gender_emb + icu_type_emb
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, num: torch.Tensor, cat: torch.Tensor):
        """
        num: (B, 33)  float
        cat: (B, 2)   long  [gender, icu_type]
        """
        g   = self.gender_emb(cat[:, 0].clamp(0, 1))
        icu = self.icu_type_emb(cat[:, 1].clamp(0, 5))
        h   = torch.cat([num, g, icu], dim=-1)
        return self.mlp(h)


class PatientEncoder(nn.Module):
    """


      Stage 1:  ts_encoder(183→H) → projection(H→embed_dim) → L2 norm
      Stage 2:  ts_encoder(183→H) → projection(H→embed_dim)
                                  ↓ concat static_encoder(33→S)
                              final_proj(embed_dim+S → embed_dim) → L2 norm


    """

    def __init__(
        self,
        ts_input_dim: int       = 183,
        ts_hidden_dim: int      = 256,
        ts_n_layers: int        = 6,
        static_numeric_dim: int = 33,
        static_out_dim: int     = 64,
        embed_dim: int          = 256,
        use_static: bool        = True,
    ):
        super().__init__()
        self.use_static = use_static

        self.ts_encoder = TemporalEncoder(
            input_dim  = ts_input_dim,
            hidden_dim = ts_hidden_dim,
            n_layers   = ts_n_layers,
        )

        self.projection = nn.Sequential(
            nn.Linear(ts_hidden_dim, ts_hidden_dim),
            nn.GELU(),
            nn.Linear(ts_hidden_dim, embed_dim),
        )

        if use_static:
            self.static_encoder = StaticEncoder(
                numeric_dim = static_numeric_dim,
                out_dim     = static_out_dim,
            )
            self.final_proj = nn.Sequential(
                nn.Linear(embed_dim + static_out_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        self.embed_dim = embed_dim

    def forward(
        self,
        ts: torch.Tensor,
        static_num: torch.Tensor = None,
        static_cat: torch.Tensor = None,
        return_seq: bool = False,
    ):
        """
        ts         : (B, T, D)
        static_num : (B, 33)
        static_cat : (B, 2)
        """
        if return_seq:
            return self.ts_encoder(ts, return_seq=True)

        ts_vec = self.ts_encoder(ts, return_seq=False)
        ts_emb = self.projection(ts_vec)

        if self.use_static and static_num is not None:
            stat_vec = self.static_encoder(static_num, static_cat)
            h = torch.cat([ts_emb, stat_vec], dim=-1)
            emb = self.final_proj(h)
        else:
            emb = ts_emb

        return F.normalize(emb, dim=-1)