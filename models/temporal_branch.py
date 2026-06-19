import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Fore, Style

# Initialize colorama for clean Windows CLI coloring
colorama.init(autoreset=True)


class TemporalVitalsBranch(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, nhead=4, compat_mode=False):
        super(TemporalVitalsBranch, self).__init__()
        self.compat_mode = compat_mode
        self.input_dim = input_dim
        self.register_buffer("feature_mean", torch.tensor([80.0, 95.0, 120.0, 37.0, 16.0], dtype=torch.float32), persistent=False)
        self.register_buffer("feature_std", torch.tensor([20.0, 5.0, 15.0, 1.0, 4.0], dtype=torch.float32), persistent=False)
        self.sequence_norm = nn.Identity()
        # 1. Capture local sequential dependencies and momentum via Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles the dimensions, so divide by 2
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 2. Add a Transformer Encoder Layer to compute global long-range relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Learnable sequence attention pooling layer to weight hourly vitals dynamically
        # Keep checkpoint key names stable for backward compatibility.
        # The old weights expect sequence_attention.0 and sequence_attention.2,
        # so we preserve a 3-layer parameterized structure here.
        self.sequence_attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Keep projection.* keys stable for checkpoint compatibility.
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def _prepare_vitals(self, vitals_matrix):
        if self.compat_mode:
            return vitals_matrix
        vitals_matrix = self.sequence_norm(vitals_matrix)
        return vitals_matrix

    def forward(self, csv_path, seed=None, add_noise=False, return_attention=False):
        print(f"\n{Fore.CYAN}[STAGE 3.1: Temporal Input Ingestion]{Style.RESET_ALL}")
        print(f"  Reading patient vitals timeline CSV from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"  Ingested CSV data. Row count: {Fore.WHITE}{len(df)}{Style.RESET_ALL} | Columns: {Fore.WHITE}{list(df.columns)}")
 
        # Extract continuous physiological metrics and apply clean interpolation passes
        vitals_matrix = df[["HeartRate", "SpO2", "BloodPressure", "Temperature", "RespirationRate"]].copy().ffill().bfill()
        
        # If seed is provided and noise is enabled, add realistic clinical noise to make every patient sequence unique
        if seed is not None and add_noise:
            # Deterministic generator based on the seed
            rng = np.random.default_rng(seed)
            hr_noise = rng.normal(0, 4.0, size=len(df))
            spo2_noise = rng.normal(0, 0.8, size=len(df))
            bp_noise = rng.normal(0, 5.0, size=len(df))
            temp_noise = rng.normal(0, 0.2, size=len(df))
            rr_noise = rng.normal(0, 1.5, size=len(df))
            
            # Add noise and clip to realistic physiological ranges
            vitals_matrix["HeartRate"] = (vitals_matrix["HeartRate"] + hr_noise).clip(40, 180)
            vitals_matrix["SpO2"] = (vitals_matrix["SpO2"] + spo2_noise).clip(50, 100)
            vitals_matrix["BloodPressure"] = (vitals_matrix["BloodPressure"] + bp_noise).clip(50, 200)
            vitals_matrix["Temperature"] = (vitals_matrix["Temperature"] + temp_noise).clip(34.0, 42.0)
            vitals_matrix["RespirationRate"] = (vitals_matrix["RespirationRate"] + rr_noise).clip(8.0, 45.0)
            print(f"  Injected patient-specific clinical noise (Seed: {seed})")

        print(f"  Extracted metrics: ['HeartRate', 'SpO2', 'BloodPressure', 'Temperature', 'RespirationRate']")

        # Clamp raw vitals to clinical ranges before normalization
        if not self.compat_mode:
            lower_limits = np.array([30.0, 50.0, 40.0, 32.0, 6.0], dtype=np.float32)
            upper_limits = np.array([220.0, 100.0, 220.0, 43.0, 50.0], dtype=np.float32)
            for idx, col in enumerate(["HeartRate", "SpO2", "BloodPressure", "Temperature", "RespirationRate"]):
                vitals_matrix[col] = vitals_matrix[col].clip(lower_limits[idx], upper_limits[idx])

        # Apply clinical standard scaling (z-score normalization) to stabilize gradients and prevent scale dominance
        mean_ref = np.array([80.0, 95.0, 120.0, 37.0, 16.0], dtype=np.float32)
        std_ref = np.array([20.0, 5.0, 15.0, 1.0, 4.0], dtype=np.float32)
        normalized_vitals = (vitals_matrix.values - mean_ref) / std_ref

        # Standardize matrix into a 3D sequential PyTorch tensor [Batch, Sequence_Length (24), Features (3)]
        vitals_tensor = torch.tensor(normalized_vitals, dtype=torch.float32).unsqueeze(0)
        vitals_tensor = self._prepare_vitals(vitals_tensor)
        print(f"  Standardized vitals tensor shape: {Fore.WHITE}{list(vitals_tensor.shape)}")

        print(f"{Fore.CYAN}[STAGE 3.2: Bidirectional GRU Sequence processing]{Style.RESET_ALL}")
        # Pass 1: Extract forward and backward contextual states via BiGRU
        gru_out, _ = self.gru(vitals_tensor)  # Shape: [1, 24, 128]
        print(f"  BiGRU output sequence tensor shape: {Fore.WHITE}{list(gru_out.shape)}")
        
        # Display BiGRU output values snippet for research visibility
        print(f"  BiGRU output values snippet (First 3 hours, first 5 hidden features):")
        for hour in range(3):
            features_slice = gru_out[0, hour, :5].detach().cpu().numpy()
            print(f"    Hour {hour+1:02d}: {Fore.YELLOW}{features_slice}{Style.RESET_ALL}")
        print(f"    ...")

        print(f"{Fore.CYAN}[STAGE 3.3: Self-Attention Transformer Encoder]{Style.RESET_ALL}")
        # Pass 2: Apply Multi-Head Self-Attention across the timeline matrix
        transformer_out = self.transformer_encoder(gru_out)  # Shape: [1, 24, 128]
        print(f"  Transformer Encoder attention output shape: {Fore.WHITE}{list(transformer_out.shape)}")

        print(f"{Fore.CYAN}[STAGE 3.4: Learnable Temporal Attention Pooling & Embedding Projection]{Style.RESET_ALL}")
        # Compute dynamic attention weights for each of the 24 hour sequence steps
        # transformer_out shape: [1, 24, 128]
        attn_logits = self.sequence_attention(transformer_out)  # Shape: [1, 24, 1]
        attn_weights = F.softmax(attn_logits, dim=1)            # Shape: [1, 24, 1]
        
        # Weighted sum across sequence dimension
        pooled_sequence = torch.sum(transformer_out * attn_weights, dim=1)  # Shape: [1, 128]
        
        # Log first 5 hours of weights for visual tracing in terminal
        weights_snippet = attn_weights[0, :5, 0].detach().cpu().numpy()
        print(f"  Computed temporal attention weights (First 5 hours): {Fore.YELLOW}{weights_snippet}")
        print(f"  Attention pooled sequence vector shape: {Fore.WHITE}{list(pooled_sequence.shape)}")
        
        temporal_embeddings = self.projection(pooled_sequence)
        print(f"  Projected temporal embedding vector (Query) shape: {Fore.WHITE}{list(temporal_embeddings.shape)}")
        print(f"  Sample Query values (first 5 elements): {Fore.YELLOW}{temporal_embeddings[0][:5].detach().cpu().numpy()}")

        if return_attention:
            return temporal_embeddings, attn_weights
        return temporal_embeddings  # Master Output Matrix Footprint: [1, 128] -> Core Query Vector
