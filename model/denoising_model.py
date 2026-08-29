import torch
from torch import nn

from .transformer_model import *
from .layers import MLP, PositionalEncoding, ConcatSquashLinear


class CondTraj(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 context_dim=256,
                 num_sample=20,
                 tf_layer=2,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.num_sample = num_sample
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.output_dim = input_feats * self.num_sample

        # Input Embedding
        self.global_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )

        # Output Module
        self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())
        self.sample_decoder = MLP(
            latent_dim + 32,
            self.output_dim,
            hid_feat=(1024, 1024),
            activation=nn.ReLU(),
        )
        self.mean_decoder = MLP(
            latent_dim,
            input_feats,
            hid_feat=(512, 256, 128),
            activation=nn.ReLU(),
        )
        self.var_decoder = MLP(
            latent_dim, 1, hid_feat=(512, 256, 128), activation=nn.ReLU()
        )

        # define denoise sampling model
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(
            input_feats, 2 * context_dim, context_dim + 3
        )
        self.context_embedding = nn.Linear(self.input_feats, context_dim)
        self.context_pos_emb = PositionalEncoding(
            d_model=context_dim, dropout=dropout, max_len=self.num_frames
        )
        context_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            dim_feedforward=context_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=tf_layer
        )

        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear = ConcatSquashLinear(
            context_dim // 2, input_feats, context_dim + 3
        )

        denoising_layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim,
            nhead=num_heads,
            dim_feedforward=2 * context_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.denoising_transformer = nn.TransformerEncoder(
            denoising_layer, num_layers=tf_layer
        )

    def _decode_temporal(self, h, emb):
        residuals = []
        for index, module in enumerate(self.temporal_decoder_blocks):
            if index < self.num_layers // 2:
                residuals.append(h)
                h = module(h, emb)
            else:
                h = module(h, emb)
                h = h + residuals.pop()
        return h

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        h = self.global_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        h = self._decode_temporal(h, emb)

        noised_mean = self.mean_decoder(h)
        noised_var = self.var_decoder(h)
        noised_scale = self.scale_encoder(noised_var)
        total_var = torch.cat((h, noised_scale), dim=-1)
        noised_traj = self.sample_decoder(total_var).view(B, self.num_sample, T, -1).contiguous()

        return noised_traj, noised_mean, noised_var

    def generate_accelerate(self, x, beta, context):
        beta = beta.view(beta.size(0), 1, 1)

        context = self.context_embedding(context)
        context = self.context_pos_emb(context.permute(1, 0, 2)).permute(1, 0, 2)
        context = self.context_encoder(context).mean(dim=1, keepdim=True)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)

        x = self.concat1.batch_generate(ctx_emb, x)

        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb).contiguous().permute(1, 0, 2)

        trans = self.denoising_transformer(final_emb)

        trans = self.concat3.batch_generate(ctx_emb, trans)
        trans = self.concat4.batch_generate(ctx_emb, trans)

        return self.linear.batch_generate(ctx_emb, trans)
