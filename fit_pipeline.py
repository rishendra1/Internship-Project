import sys

_sys_argv_backup = list(sys.argv)
_sys_orig_argv_backup = list(sys.orig_argv) if hasattr(sys, "orig_argv") else None

# Ultralytics inspects argv on import in some environments. Keep imports quiet.
sys.argv = [sys.argv[0]]
if hasattr(sys, "orig_argv"):
    sys.orig_argv = [sys.orig_argv[0]]

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass

import colorama
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore, Style
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

colorama.init(autoreset=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion_core import CrossAttentionFusionCore
from models.spatial_branch import SpatialVisionBranch
from models.temporal_branch import TemporalVitalsBranch

sys.argv = _sys_argv_backup
if _sys_orig_argv_backup is not None:
    sys.orig_argv = _sys_orig_argv_backup


SEVERE_CLASSES = {1, 4, 10, 12}
VITAL_COLUMNS = ["HeartRate", "SpO2", "BloodPressure", "Temperature", "RespirationRate"]
DEFAULT_SPLIT_VITAL_RANGES = {
    "train": (1, 300),
    "valid": (301, 400),
    "val": (301, 400),
    "test": (401, 500),
}


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_int_hash(text):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def infer_split_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for split in ("train", "valid", "val", "test"):
        if split in parts:
            return "valid" if split == "val" else split
    return "train"


def read_image_label(label_path):
    is_critical = 0
    if not os.path.exists(label_path):
        return is_critical
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if class_id in SEVERE_CLASSES:
                is_critical = 1
                break
    return is_critical


def confusion_from_scores(labels, scores, threshold):
    tn = fp = fn = tp = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    return tn, fp, fn, tp


def classification_metrics(labels, scores, threshold):
    tn, fp, fn, tp = confusion_from_scores(labels, scores, threshold)
    total = max(tn + fp + fn + tp, 1)
    accuracy = (tp + tn) / total
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def roc_auc_score(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision_score(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    tp = 0
    precision_sum = 0.0
    for idx, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / idx
    return precision_sum / positives


def fbeta_score(precision, sensitivity, beta=2.0):
    beta_sq = beta * beta
    denom = beta_sq * precision + sensitivity
    if denom <= 1e-12:
        return 0.0
    return (1.0 + beta_sq) * precision * sensitivity / denom


def optimize_threshold(labels, scores, min_sensitivity=0.78, min_precision=0.70, min_specificity=0.50, beta=1.0):
    """
    Validation-only threshold selection.

    Preference order:
    1. Satisfy minimum sensitivity, precision, and specificity floors.
    2. Maximize F-beta.
    3. Maximize balanced accuracy.
    4. Prefer thresholds near 0.50 when score separation is weak.

    This prevents pathological low thresholds from turning the classifier into
    a near-always-positive triage alarm when probabilities are compressed.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    prevalence = float(labels.mean()) if len(labels) else 0.5
    prevalence_midpoint = float(np.clip(prevalence, 0.20, 0.80))
    target_threshold = 0.50
    thresholds = np.unique(np.concatenate([
        np.linspace(0.35, 0.75, 81),
        scores,
        np.array([0.40, 0.45, 0.50, 0.55, 0.60, prevalence_midpoint], dtype=np.float64),
    ]))
    best = None
    best_safe = None
    for threshold in thresholds:
        metrics = classification_metrics(labels, scores, float(threshold))
        metrics["balanced_accuracy"] = 0.5 * (metrics["sensitivity"] + metrics["specificity"])
        metrics["f_beta"] = fbeta_score(metrics["precision"], metrics["sensitivity"], beta=beta)
        proximity = -abs(float(threshold) - target_threshold)
        key = (metrics["f_beta"], metrics["balanced_accuracy"], metrics["precision"], metrics["specificity"], proximity)
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
        if metrics["sensitivity"] < min_sensitivity:
            continue
        if metrics["precision"] < min_precision:
            continue
        if metrics["specificity"] < min_specificity:
            continue
        proximity = -abs(float(threshold) - target_threshold)
        key = (metrics["f_beta"], metrics["balanced_accuracy"], metrics["precision"], metrics["specificity"], proximity)
        if best_safe is None or key > best_safe[0]:
            best_safe = (key, float(threshold), metrics)

    chosen = best_safe if best_safe is not None else best
    if chosen is None:
        fallback_threshold = float(np.clip(prevalence_midpoint, 0.30, 0.70))
        metrics = classification_metrics(labels, scores, fallback_threshold)
        metrics["balanced_accuracy"] = 0.5 * (metrics["sensitivity"] + metrics["specificity"])
        metrics["f_beta"] = fbeta_score(metrics["precision"], metrics["sensitivity"], beta=beta)
        return fallback_threshold, metrics

    final_threshold = float(np.clip(chosen[1], 0.40, 0.75))
    if final_threshold != chosen[1]:
        metrics = classification_metrics(labels, scores, final_threshold)
        metrics["balanced_accuracy"] = 0.5 * (metrics["sensitivity"] + metrics["specificity"])
        metrics["f_beta"] = fbeta_score(metrics["precision"], metrics["sensitivity"], beta=beta)
        return final_threshold, metrics
    return chosen[1], chosen[2]


def threshold_sweep(labels, scores, thresholds):
    rows = []
    for threshold in thresholds:
        m = classification_metrics(labels, scores, float(threshold))
        m["balanced_accuracy"] = 0.5 * (m["sensitivity"] + m["specificity"])
        m["threshold"] = float(threshold)
        m["f_beta"] = fbeta_score(m["precision"], m["sensitivity"], beta=1.0)
        rows.append(m)
    return rows


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return ((centre - margin) / denom, (centre + margin) / denom)


def attach_confidence_intervals(metrics):
    tp, fp, fn, tn = metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]
    intervals = {
        "accuracy_ci95": wilson_interval(tp + tn, tp + fp + fn + tn),
        "sensitivity_ci95": wilson_interval(tp, tp + fn),
        "specificity_ci95": wilson_interval(tn, tn + fp),
        "precision_ci95": wilson_interval(tp, tp + fp),
    }
    metrics.update(intervals)
    return metrics


def class_distribution(labels):
    labels = list(labels)
    critical = int(sum(labels))
    stable = int(len(labels) - critical)
    total = max(len(labels), 1)
    return {
        "stable": stable,
        "critical": critical,
        "total": len(labels),
        "critical_fraction": critical / total,
    }


def evaluation_reliability(labels, min_critical=30):
    distribution = class_distribution(labels)
    critical = distribution["critical"]
    if critical < min_critical:
        status = "unstable"
        note = f"Only {critical} critical cases; sensitivity has high binomial uncertainty."
    else:
        status = "acceptable"
        note = f"{critical} critical cases available for sensitivity estimation."
    distribution.update({"status": status, "note": note})
    return distribution


def nll_from_logits(logits, labels, temperature):
    scaled = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(torch.tensor(scaled, dtype=torch.float32), dim=1)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return float(nn.functional.nll_loss(torch.log(probs + 1e-8), label_tensor).item())


def calibrate_temperature(logits, labels):
    candidates = np.linspace(0.8, 2.0, 25)
    best_temp = 1.0
    best_nll = float("inf")
    for temp in candidates:
        loss = nll_from_logits(logits, labels, temp)
        if loss < best_nll:
            best_nll = loss
            best_temp = float(temp)
    return best_temp, best_nll


def probability_summary(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "over_0_24": float("nan"),
            "over_0_50": float("nan"),
        }
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
        "over_0_24": float(np.mean(scores >= 0.24)),
        "over_0_50": float(np.mean(scores >= 0.50)),
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.25, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha[targets] * loss
        return loss.mean()


class MultimodalClinicalDataset(Dataset):
    """
    Image-label dataset paired with split-disjoint vitals files.

    The model receives only image pixels and numeric vitals values. Vitals file
    pools are disjoint across train/valid/test, preventing the old overlap where
    the same CSV could appear in both training and evaluation.
    """
    def __init__(
        self,
        images_dir,
        labels_dir,
        vitals_dir,
        mismatch_rate=0.0,
        split=None,
        split_vital_ranges=None,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.vitals_dir = vitals_dir
        self.mismatch_rate = mismatch_rate
        self.split = split or infer_split_from_path(images_dir)
        self.split_vital_ranges = split_vital_ranges or DEFAULT_SPLIT_VITAL_RANGES
        self.image_files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.image_files)

    def _vital_index_for_patient(self, patient_id):
        start, end = self.split_vital_ranges.get(self.split, (1, 500))
        span = end - start + 1
        return start + (stable_int_hash(patient_id) % span)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        patient_id = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, f"{patient_id}.txt")
        is_critical = read_image_label(label_path)

        # Keep the image label and vitals label aligned. The training loop already
        # applies physiological noise augmentation, so we do not inject label flips
        # here because they destroy calibration and inflate false positives.
        prefix = "critical" if is_critical == 1 else "stable"
        index = self._vital_index_for_patient(patient_id)
        csv_path = os.path.join(self.vitals_dir, f"{prefix}_p{index:06d}.csv")

        if not os.path.exists(csv_path):
            legacy_file = "patient_crtitcal.csv" if is_critical == 1 else "patient_stable.csv"
            csv_path = os.path.join(self.vitals_dir, legacy_file)

        return {
            "image_path": img_path,
            "vitals_path": csv_path,
            "label": is_critical,
            "patient_id": patient_id,
            "index": idx,
        }


@dataclass
class EvalResult:
    labels: list
    scores: list
    logits: list
    loss: float
    indices: list


def build_datasets():
    root = os.getcwd()
    return {
        "train": MultimodalClinicalDataset(
            images_dir=os.path.join(root, "data", "images", "train", "images"),
            labels_dir=os.path.join(root, "data", "images", "train", "labels"),
            vitals_dir=os.path.join(root, "data", "vitals"),
            split="train",
            mismatch_rate=0.0,
        ),
        "valid": MultimodalClinicalDataset(
            images_dir=os.path.join(root, "data", "images", "valid", "images"),
            labels_dir=os.path.join(root, "data", "images", "valid", "labels"),
            vitals_dir=os.path.join(root, "data", "vitals"),
            split="valid",
            mismatch_rate=0.0,
        ),
        "test": MultimodalClinicalDataset(
            images_dir=os.path.join(root, "data", "images", "test", "images"),
            labels_dir=os.path.join(root, "data", "images", "test", "labels"),
            vitals_dir=os.path.join(root, "data", "vitals"),
            split="test",
            mismatch_rate=0.0,
        ),
    }


def make_loader(dataset, batch_size=1, sampler=None, shuffle=False, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        generator=generator,
    )


def collect_labels(dataset):
    return np.array([dataset[i]["label"] for i in range(len(dataset))], dtype=np.int64)


def label_indices(dataset):
    labels = collect_labels(dataset)
    positives = np.where(labels == 1)[0].tolist()
    negatives = np.where(labels == 0)[0].tolist()
    return positives, negatives


def stratified_indices(dataset, seed=42, neg_per_pos=1, max_positives=None):
    positives, negatives = label_indices(dataset)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if max_positives is not None:
        positives = positives[:max_positives]
    n_neg = min(len(negatives), max(len(positives) * neg_per_pos, 1))
    selected = positives + negatives[:n_neg]
    rng.shuffle(selected)
    return selected


def make_subset_loader(dataset, indices, batch_size=1, seed=42):
    return make_loader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, seed=seed)


def build_weighted_sampler(labels, class_weights, hard_weights=None, seed=42):
    sample_weights = class_weights[labels].astype(np.float64)
    if hard_weights is not None:
        sample_weights = sample_weights * hard_weights.astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate_model(vision_net, vitals_net, fusion_brain, loader, criterion=None, max_steps=None, verbose=False):
    vision_net.eval()
    vitals_net.eval()
    fusion_brain.eval()
    labels, scores, logits_out, indices = [], [], [], []
    total_loss = 0.0
    n_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            img_path = batch["image_path"][0]
            v_path = batch["vitals_path"][0]
            label = torch.tensor([int(batch["label"][0])], dtype=torch.long)
            item_index = int(batch["index"][0]) if "index" in batch else i

            with open(os.devnull, "w") as f_null, contextlib.redirect_stdout(f_null):
                spatial_embeddings = vision_net(img_path)
                seed = stable_int_hash(os.path.basename(img_path)) % (2**32)
                temporal_embeddings = vitals_net(v_path, seed=seed, add_noise=False)
                logits, probs, _ = fusion_brain(
                    spatial_embeddings,
                    temporal_embeddings,
                    return_logits=True,
                )

            if criterion is not None:
                total_loss += float(criterion(logits, label).item())
                n_loss += 1
            labels.append(int(label.item()))
            scores.append(float(probs[0, 1].item()))
            logits_out.append([float(x) for x in logits[0].detach().cpu().tolist()])
            indices.append(item_index)

    avg_loss = total_loss / max(n_loss, 1)
    return EvalResult(labels=labels, scores=scores, logits=logits_out, loss=avg_loss, indices=indices)


def save_probability_histogram(scores, output_path, title="ICU Probability Histogram"):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=30, color="#2b6cb0", edgecolor="white", alpha=0.9)
    ax.axvline(0.24, color="#d97706", linestyle="--", linewidth=2, label="0.24")
    ax.axvline(0.50, color="#dc2626", linestyle="--", linewidth=2, label="0.50")
    ax.set_xlabel("P(ICU Risk)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def mine_hard_examples(
    vision_net,
    vitals_net,
    fusion_brain,
    dataset,
    hard_weights,
    threshold,
    seed=42,
    max_positives=180,
    neg_per_pos=3,
    hard_positive_weight=1.8,
    hard_negative_weight=2.4,
):
    indices = stratified_indices(dataset, seed=seed, neg_per_pos=neg_per_pos, max_positives=max_positives)
    loader = make_subset_loader(dataset, indices, batch_size=1, seed=seed)
    result = evaluate_model(vision_net, vitals_net, fusion_brain, loader)
    for item_index, label, score in zip(result.indices, result.labels, result.scores):
        if label == 1 and score < threshold:
            hard_weights[item_index] = max(hard_weights[item_index], hard_positive_weight)
        elif label == 0 and score >= threshold:
            hard_weights[item_index] = max(hard_weights[item_index], hard_negative_weight)
    return hard_weights, result


def train_model(
    iterations=100,
    batch_size=1,
    lr=2e-4,
    seed=42,
    output="best_fusion_weights.pt",
    max_train_steps=600,
    max_val_steps=300,
    patience=12,
    focal_gamma=1.0,
    partial_finetune=False,
    min_val_sensitivity=0.70,
    min_val_precision=0.60,
    min_val_specificity=0.50,
    threshold_beta=1.0,
    critical_alpha_multiplier=1.05,
    hard_mining=True,
    hard_mining_interval=4,
    stratified_test_neg_per_pos=1,
):
    set_global_seed(seed)

    print("=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}STARTING LEAKAGE-SAFE MULTIMODAL TRAINING{Style.RESET_ALL}")
    print("=" * 80)
    print(f"Seed: {seed}")
    print("Protocol: train -> validation calibration/threshold -> independent test")

    datasets = build_datasets()
    train_dataset = datasets["train"]
    val_dataset = datasets["valid"]
    test_dataset = datasets["test"]

    train_labels = collect_labels(train_dataset)
    class_counts = np.bincount(train_labels, minlength=2)
    neg_to_pos_ratio = float(class_counts[0]) / max(float(class_counts[1]), 1.0)
    sampler_class_weights = np.array([
        1.0,
        min(3.5, math.sqrt(neg_to_pos_ratio) * critical_alpha_multiplier),
    ], dtype=np.float32)
    hard_weights = np.ones(len(train_labels), dtype=np.float32)
    sampler = build_weighted_sampler(train_labels, sampler_class_weights, hard_weights=hard_weights, seed=seed)

    train_loader = make_loader(train_dataset, batch_size=batch_size, sampler=sampler, seed=seed)
    val_loader = make_loader(val_dataset, batch_size=1, shuffle=False, seed=seed)
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False, seed=seed)

    val_labels = collect_labels(val_dataset)
    test_labels = collect_labels(test_dataset)
    print(f"Training images:   {len(train_dataset)} | {class_distribution(train_labels)}")
    print(f"Validation images: {len(val_dataset)} | {class_distribution(val_labels)}")
    print(f"Test images:       {len(test_dataset)} | {class_distribution(test_labels)}")
    print(f"Split vitals pools: train={DEFAULT_SPLIT_VITAL_RANGES['train']} valid={DEFAULT_SPLIT_VITAL_RANGES['valid']} test={DEFAULT_SPLIT_VITAL_RANGES['test']}")
    print(f"Sampler weights: stable={sampler_class_weights[0]:.3f} critical={sampler_class_weights[1]:.3f}")
    print(f"Threshold objective: validation F{threshold_beta:.1f}, min sensitivity={min_val_sensitivity:.2f}, min precision={min_val_precision:.2f}, min specificity={min_val_specificity:.2f}")

    vision_net = SpatialVisionBranch(
        compat_mode=False,
        use_label_rois=False,
        apply_clahe=True,
        bilateral_filter=True,
        preserve_aspect=True,
        roi_padding=0.08,
        fine_tune_backbone=partial_finetune,
    )
    vitals_net = TemporalVitalsBranch()
    fusion_brain = CrossAttentionFusionCore()

    for name, param in vision_net.named_parameters():
        if "mid_proj" in name or "deep_proj" in name or "spatial_attention" in name:
            param.requires_grad = True
        elif partial_finetune and name.startswith("features_deep"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if not partial_finetune:
        vision_net.eval()

    focal_alpha = torch.tensor([1.0, min(1.15, 1.0 + 0.04 * sampler_class_weights[1])], dtype=torch.float32)
    criterion = FocalLoss(alpha=focal_alpha, gamma=max(0.75, focal_gamma), label_smoothing=0.01)
    optimizer = optim.AdamW(
        list(filter(lambda p: p.requires_grad, vision_net.parameters()))
        + list(vitals_net.parameters())
        + list(fusion_brain.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(iterations, 1))

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    max_train_steps = min(max_train_steps, len(train_loader))
    max_val_steps = min(max_val_steps, len(val_loader))

    for epoch in range(iterations):
        if partial_finetune:
            vision_net.train()
        vitals_net.train()
        fusion_brain.train()
        train_loss = 0.0
        train_correct = 0

        print("-" * 80)
        print(f"{Fore.YELLOW}{Style.BRIGHT}Epoch {epoch + 1}/{iterations}{Style.RESET_ALL}")

        for i, batch in enumerate(train_loader):
            if i >= max_train_steps:
                break
            img_path = batch["image_path"][0]
            v_path = batch["vitals_path"][0]
            label = torch.tensor([int(batch["label"][0])], dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with open(os.devnull, "w") as f_null, contextlib.redirect_stdout(f_null):
                spatial_embeddings = vision_net(img_path)
                seed_i = stable_int_hash(os.path.basename(img_path)) % (2**32)
                temporal_embeddings = vitals_net(v_path, seed=seed_i, add_noise=True)
                
                # Apply Modality Dropout during training to prevent shortcut learning
                if fusion_brain.training:
                    dropout_rand = random.random()
                    if dropout_rand < 0.15:
                        spatial_embeddings = torch.zeros_like(spatial_embeddings)
                    elif dropout_rand < 0.30:
                        temporal_embeddings = torch.zeros_like(temporal_embeddings)

                logits, probs, _ = fusion_brain(
                    spatial_embeddings,
                    temporal_embeddings,
                    return_logits=True,
                )
                loss = criterion(logits, label)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(vitals_net.parameters()) + list(fusion_brain.parameters()),
                max_norm=3.0,
            )
            optimizer.step()

            train_loss += float(loss.item())
            pred = torch.argmax(probs, dim=1)
            train_correct += int((pred == label).sum().item())

            if (i + 1) % 50 == 0:
                print(f"  Batch {i + 1:03d}/{max_train_steps} | focal loss={loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(max_train_steps, 1)
        train_acc = train_correct / max(max_train_steps, 1)

        val_result = evaluate_model(
            vision_net,
            vitals_net,
            fusion_brain,
            val_loader,
            criterion=criterion,
            max_steps=max_val_steps,
        )
        val_metrics = classification_metrics(val_result.labels, val_result.scores, threshold=0.5)

        print(f"  Train loss={avg_train_loss:.4f} | train acc={train_acc * 100:.2f}%")
        print(f"  Val loss={val_result.loss:.4f} | val acc@0.5={val_metrics['accuracy'] * 100:.2f}% | val F1@0.5={val_metrics['f1'] * 100:.2f}%")

        if hard_mining and (epoch + 1) % max(hard_mining_interval, 1) == 0:
            hard_weights, hard_result = mine_hard_examples(
                vision_net,
                vitals_net,
                fusion_brain,
                train_dataset,
                hard_weights,
                threshold=0.5,
                seed=seed + epoch + 1,
            )
            hard_metrics = classification_metrics(hard_result.labels, hard_result.scores, threshold=0.5)
            sampler = build_weighted_sampler(train_labels, sampler_class_weights, hard_weights=hard_weights, seed=seed + epoch + 1)
            train_loader = make_loader(train_dataset, batch_size=batch_size, sampler=sampler, seed=seed + epoch + 1)
            print(
                "  Hard mining subset | "
                f"sens={hard_metrics['sensitivity'] * 100:.2f}% "
                f"prec={hard_metrics['precision'] * 100:.2f}% "
                f"boosted={int(np.sum(hard_weights > 1.0))}"
            )

        if val_result.loss < best_val_loss - 1e-5:
            best_val_loss = val_result.loss
            best_epoch = epoch + 1
            stale_epochs = 0
            checkpoint = {
                "vitals_net_state": vitals_net.state_dict(),
                "fusion_brain_state": fusion_brain.state_dict(),
                "vision_net_state": vision_net.state_dict(),
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "split_protocol": "split_disjoint_vitals_train_1_300_valid_301_400_test_401_500",
                "architecture": {
                    "roi_detector": "YOLOv8n",
                    "spatial_branch": "MobileNetV3-Small",
                    "temporal_branch": "BiGRU+Transformer",
                    "fusion": "Multi-scale dual-path cross-attention",
                },
            }
            torch.save(checkpoint, output)
            print(f"  {Fore.GREEN}Saved best validation checkpoint to {output}{Style.RESET_ALL}")
        else:
            stale_epochs += 1
            print(f"  No validation improvement ({stale_epochs}/{patience})")
            if stale_epochs >= patience:
                print(f"{Fore.YELLOW}Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}.{Style.RESET_ALL}")
                break

    print("=" * 80)
    print("VALIDATION-ONLY CALIBRATION AND THRESHOLD SELECTION")
    print("=" * 80)

    checkpoint = torch.load(output, map_location="cpu")
    vision_net.load_state_dict(checkpoint["vision_net_state"], strict=False)
    vitals_net.load_state_dict(checkpoint["vitals_net_state"], strict=False)
    fusion_brain.load_state_dict(checkpoint["fusion_brain_state"], strict=False)

    val_result = evaluate_model(
        vision_net,
        vitals_net,
        fusion_brain,
        val_loader,
        criterion=criterion,
        max_steps=None,
    )
    temperature, val_nll = calibrate_temperature(np.array(val_result.logits), val_result.labels)
    calibrated_val_probs = torch.softmax(
        torch.tensor(np.array(val_result.logits), dtype=torch.float32) / temperature,
        dim=1,
    )[:, 1].numpy().tolist()
    threshold, val_threshold_metrics = optimize_threshold(
        val_result.labels,
        calibrated_val_probs,
        min_sensitivity=min_val_sensitivity,
        min_precision=min_val_precision,
        min_specificity=min_val_specificity,
        beta=threshold_beta,
    )

    print(f"Validation temperature: {temperature:.3f} | validation NLL={val_nll:.4f}")
    print(f"Validation threshold:   {threshold:.3f} | val F1={val_threshold_metrics['f1'] * 100:.2f}% | val sens={val_threshold_metrics['sensitivity'] * 100:.2f}%")

    fusion_brain.temperature = temperature
    test_result = evaluate_model(
        vision_net,
        vitals_net,
        fusion_brain,
        test_loader,
        criterion=criterion,
        max_steps=None,
    )
    test_scores = test_result.scores
    test_metrics = classification_metrics(test_result.labels, test_scores, threshold=threshold)
    test_metrics["roc_auc"] = roc_auc_score(test_result.labels, test_scores)
    test_metrics["pr_auc"] = average_precision_score(test_result.labels, test_scores)
    test_metrics = attach_confidence_intervals(test_metrics)
    test_reliability = evaluation_reliability(test_result.labels)

    stratified_indices_test = stratified_indices(
        test_dataset,
        seed=seed,
        neg_per_pos=stratified_test_neg_per_pos,
        max_positives=None,
    )
    stratified_test_loader = make_subset_loader(test_dataset, stratified_indices_test, batch_size=1, seed=seed)
    stratified_result = evaluate_model(
        vision_net,
        vitals_net,
        fusion_brain,
        stratified_test_loader,
        criterion=criterion,
        max_steps=None,
    )
    stratified_test_metrics = classification_metrics(stratified_result.labels, stratified_result.scores, threshold=threshold)
    stratified_test_metrics["roc_auc"] = roc_auc_score(stratified_result.labels, stratified_result.scores)
    stratified_test_metrics["pr_auc"] = average_precision_score(stratified_result.labels, stratified_result.scores)
    stratified_test_metrics = attach_confidence_intervals(stratified_test_metrics)
    stratified_reliability = evaluation_reliability(stratified_result.labels)

    checkpoint.update({
        "calibrated_temperature": temperature,
        "optimal_threshold": threshold,
        "validation_metrics_at_threshold": val_threshold_metrics,
        "test_metrics": test_metrics,
        "test_reliability": test_reliability,
        "stratified_test_metrics": stratified_test_metrics,
        "stratified_test_reliability": stratified_reliability,
        "threshold_objective": {
            "metric": f"F{threshold_beta:.1f}",
            "min_sensitivity": min_val_sensitivity,
            "min_precision": min_val_precision,
            "min_specificity": min_val_specificity,
        },
    })
    torch.save(checkpoint, output)

    print("=" * 80)
    print(f"{Fore.GREEN}{Style.BRIGHT}INDEPENDENT TEST RESULTS - NO TEST TUNING{Style.RESET_ALL}")
    print("=" * 80)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) and math.isfinite(v) else v for k, v in test_metrics.items()}, indent=2))
    print("Test reliability:")
    print(json.dumps(test_reliability, indent=2))
    print("=" * 80)
    print(f"{Fore.GREEN}{Style.BRIGHT}STRATIFIED TEST ANALYSIS - REPORT AS SECONDARY{Style.RESET_ALL}")
    print("=" * 80)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) and math.isfinite(v) else v for k, v in stratified_test_metrics.items()}, indent=2))
    print("Stratified reliability:")
    print(json.dumps(stratified_reliability, indent=2))
    print("=" * 80)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-safe multimodal clinical training loop")
    parser.add_argument("--iterations", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--output", type=str, default="best_fusion_weights.pt", help="Checkpoint path")
    parser.add_argument("--max-train-steps", type=int, default=600, help="Training batches per epoch")
    parser.add_argument("--max-val-steps", type=int, default=300, help="Validation batches per epoch for checkpointing")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--focal-gamma", type=float, default=1.0, help="Focal loss gamma")
    parser.add_argument("--partial-finetune", action="store_true", help="Unfreeze final MobileNetV3 blocks")
    parser.add_argument("--min-val-sensitivity", type=float, default=0.70, help="Minimum validation sensitivity during threshold selection")
    parser.add_argument("--min-val-precision", type=float, default=0.60, help="Minimum validation precision during threshold selection")
    parser.add_argument("--min-val-specificity", type=float, default=0.50, help="Minimum validation specificity during threshold selection")
    parser.add_argument("--threshold-beta", type=float, default=1.0, help="F-beta value for validation threshold selection")
    parser.add_argument("--critical-alpha-multiplier", type=float, default=1.05, help="Extra sampler emphasis for critical cases")
    parser.add_argument("--disable-hard-mining", action="store_true", help="Disable hard positive/negative replay weighting")
    parser.add_argument("--hard-mining-interval", type=int, default=4, help="Epoch interval for hard-example mining")
    parser.add_argument("--stratified-test-neg-per-pos", type=int, default=1, help="Stable controls per critical case in secondary stratified test analysis")
    args = parser.parse_args(_sys_argv_backup[1:])

    train_model(
        iterations=args.iterations,
        lr=args.lr,
        seed=args.seed,
        output=args.output,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
        partial_finetune=args.partial_finetune,
        min_val_sensitivity=args.min_val_sensitivity,
        min_val_precision=args.min_val_precision,
        min_val_specificity=args.min_val_specificity,
        threshold_beta=args.threshold_beta,
        critical_alpha_multiplier=args.critical_alpha_multiplier,
        hard_mining=not args.disable_hard_mining,
        hard_mining_interval=args.hard_mining_interval,
        stratified_test_neg_per_pos=args.stratified_test_neg_per_pos,
    )
