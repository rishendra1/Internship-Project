import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Fore, Style

# Initialize colorama for clean Windows CLI coloring
colorama.init(autoreset=True)


class CrossAttentionFusionCore(nn.Module):
    """
    Symmetric Multi-Scale Dual-Path Cross-Attention (MS-DPCA) Fusion Core.
    Divides the 128-dimensional visual and temporal embeddings into:
    1. Fine-Grained Path (First 64 dims): Aligns local spatial textures with vitals.
    2. Coarse-Grained Path (Last 64 dims): Aligns abstract global pathologies with vitals.
    Fuses both paths using residual connections and a Feed-Forward Network (FFN).
    """
    def __init__(self, feature_dim=128, num_classes=2, compat_mode=False):
        super(CrossAttentionFusionCore, self).__init__()
        self.compat_mode = compat_mode
        self.path_dim = feature_dim // 2  # Split 128 into two 64-dimensional paths
        self.num_tokens = 4
        self.token_dim = self.path_dim // self.num_tokens
        self.head_dim = self.token_dim
        self.temperature = 1.0

        # 1. Fine-Grained Path Projections (Local features)
        self.q_mid = nn.Linear(self.path_dim, self.path_dim)
        self.k_mid = nn.Linear(self.path_dim, self.path_dim)
        self.v_mid = nn.Linear(self.path_dim, self.path_dim)
        self.unify_mid = nn.Linear(self.path_dim, self.path_dim)

        # 2. Coarse-Grained Path Projections (Global features)
        self.q_deep = nn.Linear(self.path_dim, self.path_dim)
        self.k_deep = nn.Linear(self.path_dim, self.path_dim)
        self.v_deep = nn.Linear(self.path_dim, self.path_dim)
        self.unify_deep = nn.Linear(self.path_dim, self.path_dim)

        # Regularization layers to stabilize gradients
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

        # Feed-Forward Network (FFN) to capture complex non-linear combinations
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # Gated Cross-Attention Fusion (GCAF) gating networks
        self.gate_spatial = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.gate_temporal = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        # Learnable gating MLP replacing the heuristic statistics-based gate
        self.gating_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
        # Use a deterministic reliability gate so checkpoint compatibility is preserved.
        # The model remains clinically interpretable, but we avoid introducing a
        # checkpoint-only mismatch for an auxiliary reliability MLP.
        self.register_buffer("reliability_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("reliability_bias", torch.tensor(0.0), persistent=False)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def _token_attention(self, query, key, value, unify):
        batch_size = query.shape[0]
        query_tokens = query.view(batch_size, self.num_tokens, self.token_dim)
        key_tokens = key.view(batch_size, self.num_tokens, self.token_dim)
        value_tokens = value.view(batch_size, self.num_tokens, self.token_dim)

        scores = torch.matmul(query_tokens, key_tokens.transpose(1, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, value_tokens).reshape(batch_size, self.path_dim)
        return unify(context), attention

    def forward(self, spatial_vector, temporal_vector, return_logits=False):
        print(f"\n{Fore.CYAN}[STAGE 4.1: Symmetrical Dual-Path Projections (MS-DPCA)]{Style.RESET_ALL}")
        
        # Split embeddings into Fine-Grained (first 64) and Coarse-Grained (last 64) paths
        temporal_mid, temporal_deep = temporal_vector[:, :self.path_dim], temporal_vector[:, self.path_dim:]
        spatial_mid, spatial_deep = spatial_vector[:, :self.path_dim], spatial_vector[:, self.path_dim:]

        # Project mid-level paths
        Q_mid = self.q_mid(temporal_mid)
        K_mid = self.k_mid(spatial_mid)
        V_mid = self.v_mid(spatial_mid)

        # Project deep-level paths
        Q_deep = self.q_deep(temporal_deep)
        K_deep = self.k_deep(spatial_deep)
        V_deep = self.v_deep(spatial_deep)

        print(f"  Fine-Grained Path (Mid) Q, K, V shapes:   {Fore.WHITE}{list(Q_mid.shape)}")
        print(f"  Coarse-Grained Path (Deep) Q, K, V shapes: {Fore.WHITE}{list(Q_deep.shape)}")

        print(f"{Fore.CYAN}[STAGE 4.2: Scaled Dot-Product Dual-Path Cross-Attention]{Style.RESET_ALL}")
        # Attention is computed across feature tokens inside each sample.
        # This avoids the previous batch-to-batch attention shortcut where
        # batch_size=1 collapsed attention to a constant scalar.
        unified_mid, attn_mid = self._token_attention(Q_mid, K_mid, V_mid, self.unify_mid)
        unified_deep, attn_deep = self._token_attention(Q_deep, K_deep, V_deep, self.unify_deep)

        print(f"  Fine-Grained Attention weights:            {Fore.YELLOW}{attn_mid.detach().cpu().numpy()}")
        print(f"  Coarse-Grained Attention weights:          {Fore.YELLOW}{attn_deep.detach().cpu().numpy()}")

        print(f"{Fore.CYAN}[STAGE 4.3: Multi-Scale Context Aggregation & Residual FFN]{Style.RESET_ALL}")
        # Concatenate paths back to a unified 128-dimensional context vector
        context_vector = torch.cat((unified_mid, unified_deep), dim=-1)
        print(f"  Aggregated multi-scale context shape:      {Fore.WHITE}{list(context_vector.shape)}")

        # Residual shortcut from the original query (vitals)
        attention_out = self.dropout(context_vector)
        
        # Apply Gated Cross-Attention Fusion (GCAF) dynamic gating scale
        g_v = self.gate_spatial(spatial_vector)
        g_t = self.gate_temporal(temporal_vector)
        
        if self.compat_mode:
            gate_stats = torch.stack(
                [
                    spatial_vector.mean(dim=1),
                    spatial_vector.std(dim=1, unbiased=False),
                    temporal_vector.mean(dim=1),
                    temporal_vector.std(dim=1, unbiased=False),
                    (spatial_vector - temporal_vector).abs().mean(dim=1),
                ],
                dim=1,
            )
            gate_score = torch.sigmoid(self.reliability_scale * (gate_stats[:, 1:2] + gate_stats[:, 3:4] - gate_stats[:, 4:5]) + self.reliability_bias)
            gate_score = torch.clamp(gate_score, 0.35, 0.85)
            spatial_reliability = torch.ones_like(gate_score)
            temporal_reliability = torch.ones_like(gate_score)
        else:
            cat_feat = torch.cat([spatial_vector, temporal_vector], dim=-1)
            gating_weights = self.gating_mlp(cat_feat)
            spatial_reliability = gating_weights[:, 0:1]
            temporal_reliability = gating_weights[:, 1:2]

        gated_attention = attention_out * g_v * spatial_reliability
        gated_temporal = temporal_vector * g_t * temporal_reliability

        print(
            f"  GCAF Gate weights - Spatial: {Fore.YELLOW}{g_v[0, :3].detach().cpu().numpy()}... | "
            f"Temporal: {Fore.YELLOW}{g_t[0, :3].detach().cpu().numpy()}..."
        )
        print(
            f"  Modality reliability - Spatial: {Fore.YELLOW}{spatial_reliability[0].item():.3f} | "
            f"Temporal: {Fore.YELLOW}{temporal_reliability[0].item():.3f}"
        )
        fused_tensor = self.norm1(gated_temporal + gated_attention)

        # FFN processing with second residual connection
        ffn_out = self.dropout(self.ffn(fused_tensor))
        fused_tensor = self.norm2(fused_tensor + ffn_out)
        print(f"  Unified multi-head fusion tensor shape:     {Fore.WHITE}{list(fused_tensor.shape)}")

        print(f"{Fore.CYAN}[STAGE 4.4: Neural Prognostic Classification Layer]{Style.RESET_ALL}")
        # Pass output matrices to classifier blocks to calculate index probabilities
        logits = self.classifier(fused_tensor)
        print(f"  Classifier layer output logits shape:       {Fore.WHITE}{list(logits.shape)}")
        print(f"  Raw logit values:                           {Fore.WHITE}{logits.detach().cpu().numpy()[0]}")
        
        temperature = float(getattr(self, "temperature", 1.0))
        probabilities = F.softmax(logits / max(temperature, 1e-6), dim=1)
        print(f"  Softmax classification probabilities shape: {Fore.WHITE}{list(probabilities.shape)}")
        print(f"  Homeostasis vs ICU Escalation probabilities: {Fore.YELLOW}{probabilities[0].detach().cpu().numpy()}")

        # Average attention weights for visual tracing compatibility
        avg_attention_weights = (attn_mid + attn_deep) / 2.0

        if return_logits:
            return logits, probabilities, avg_attention_weights
        return probabilities, avg_attention_weights
