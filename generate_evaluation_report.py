import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fit_pipeline import (
    MultimodalClinicalDataset,
    classification_metrics,
    roc_auc_score,
    average_precision_score,
    make_loader,
    evaluate_model,
)
from models.spatial_branch import SpatialVisionBranch
from models.temporal_branch import TemporalVitalsBranch
from models.fusion_core import CrossAttentionFusionCore

def bootstrap_metrics(labels, scores, threshold=0.555, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    bootstrapped_stats = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
    }
    
    indices = np.arange(len(labels))
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=len(labels), replace=True)
        boot_labels = labels[boot_idx]
        boot_scores = scores[boot_idx]
        
        # Check if we have at least one positive and one negative sample
        if len(np.unique(boot_labels)) < 2:
            continue
            
        m = classification_metrics(boot_labels, boot_scores, threshold)
        bootstrapped_stats["accuracy"].append(m["accuracy"])
        bootstrapped_stats["sensitivity"].append(m["sensitivity"])
        bootstrapped_stats["specificity"].append(m["specificity"])
        bootstrapped_stats["precision"].append(m["precision"])
        bootstrapped_stats["f1"].append(m["f1"])
        bootstrapped_stats["roc_auc"].append(roc_auc_score(boot_labels, boot_scores))
        bootstrapped_stats["pr_auc"].append(average_precision_score(boot_labels, boot_scores))
        
    ci_results = {}
    for metric, values in bootstrapped_stats.items():
        sorted_values = np.sort(values)
        if len(sorted_values) > 0:
            low = np.percentile(sorted_values, 2.5)
            high = np.percentile(sorted_values, 97.5)
            mean_val = np.mean(sorted_values)
            ci_results[metric] = (mean_val, low, high)
        else:
            ci_results[metric] = (float("nan"), float("nan"), float("nan"))
            
    return ci_results

def compute_calibration_curve(labels, scores, n_bins=10):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    true_proportions = []
    pred_probabilities = []
    bin_counts = []
    valid_centers = []
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        in_bin = (scores >= bin_lower) & (scores < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (scores == bin_upper)
            
        count = np.sum(in_bin)
        bin_counts.append(count)
        
        if count > 0:
            true_proportions.append(np.mean(labels[in_bin]))
            pred_probabilities.append(np.mean(scores[in_bin]))
            valid_centers.append(bin_centers[i])
            
    return np.array(valid_centers), np.array(true_proportions), np.array(pred_probabilities), bin_centers, np.array(bin_counts)

def compute_decision_curve(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n_samples = len(labels)
    prevalence = np.mean(labels)
    
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    for pt in thresholds:
        # Net Benefit (Model) = (TP - FP * (pt / (1 - pt))) / N
        tp = np.sum((scores >= pt) & (labels == 1))
        fp = np.sum((scores >= pt) & (labels == 0))
        nb_model = (tp - fp * (pt / (1.0 - pt))) / n_samples
        net_benefit_model.append(nb_model)
        
        # Net Benefit (Treat All) = (Positives - Negatives * (pt / (1 - pt))) / N
        positives = np.sum(labels == 1)
        negatives = np.sum(labels == 0)
        nb_all = (positives - negatives * (pt / (1.0 - pt))) / n_samples
        net_benefit_all.append(nb_all)
        
        # Net Benefit (Treat None) = 0
        net_benefit_none.append(0.0)
        
    return thresholds, np.array(net_benefit_model), np.array(net_benefit_all), np.array(net_benefit_none)

def main():
    print("=" * 70)
    print("  GENERATING EVALUATION REPORT AND PLOTS")
    print("=" * 70)

    checkpoint_path = os.path.join(PROJECT_ROOT, "best_fusion_weights.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Load model in compat_mode=True to reproduce target results
    vision_net = SpatialVisionBranch(compat_mode=False, use_label_rois=False)
    vitals_net = TemporalVitalsBranch(compat_mode=False)
    fusion_brain = CrossAttentionFusionCore(compat_mode=False)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vision_net.load_state_dict(checkpoint["vision_net_state"], strict=False)
    vitals_net.load_state_dict(checkpoint["vitals_net_state"], strict=False)
    fusion_brain.load_state_dict(checkpoint["fusion_brain_state"], strict=False)
    
    vision_net.eval()
    vitals_net.eval()
    fusion_brain.eval()
    
    temperature = float(checkpoint.get("calibrated_temperature", 0.800))
    threshold = float(checkpoint.get("optimal_threshold", 0.555))
    fusion_brain.temperature = temperature
    
    print(f"Model loaded: Temp={temperature:.3f}, Threshold={threshold:.3f}")
    
    # Load test dataset
    test_dataset = MultimodalClinicalDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data", "images", "test", "images"),
        labels_dir=os.path.join(PROJECT_ROOT, "data", "images", "test", "labels"),
        vitals_dir=os.path.join(PROJECT_ROOT, "data", "vitals"),
        split="test",
        mismatch_rate=0.0
    )
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate
    print("Running evaluation on full test set...")
    eval_result = evaluate_model(vision_net, vitals_net, fusion_brain, test_loader)
    labels = eval_result.labels
    scores = eval_result.scores
    
    print(f"Total test cases: {len(labels)} | Critical cases: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
    
    # Compute 95% CIs
    print("Computing 95% Confidence Intervals (1000 bootstraps)...")
    ci = bootstrap_metrics(labels, scores, threshold=threshold, n_bootstraps=1000)
    
    # Print metrics with 95% CI
    print("\nPROGNOSTIC PERFORMANCE BENCHMARKS (95% CI):")
    for metric, (mean_val, low, high) in ci.items():
        print(f"  - {metric.capitalize():<12}: {mean_val*100:6.2f}% ({low*100:.2f}% - {high*100:.2f}%)" if metric not in ["roc_auc", "pr_auc"]
              else f"  - {metric.upper():<12}: {mean_val:6.4f} ({low:.4f} - {high:.4f})")
              
    # ----------------------------------------------------
    # PLOT 1: ROC & PR Curves
    # ----------------------------------------------------
    print("\nGenerating ROC and PR Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve calculation
    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tps = 0
    fps = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    for score, label in sorted_pairs:
        if label == 1:
            tps += 1
        else:
            fps += 1
        fpr_list.append(fps / total_neg)
        tpr_list.append(tps / total_pos)
        
    auc = roc_auc_score(labels, scores)
    ax1.plot(fpr_list, tpr_list, color="#2b6cb0", lw=2.5, label=f"Model ROC (AUC = {auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="#718096", linestyle="--", lw=1.5)
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(True, alpha=0.15)
    
    # PR Curve calculation
    precision_list = [1.0]
    recall_list = [0.0]
    tps = 0
    fps = 0
    for idx, (score, label) in enumerate(sorted_pairs, start=1):
        if label == 1:
            tps += 1
        else:
            fps += 1
        precision_list.append(tps / idx)
        recall_list.append(tps / total_pos)
        
    pr_auc = average_precision_score(labels, scores)
    ax2.plot(recall_list, precision_list, color="#319795", lw=2.5, label=f"Model PR (AUC = {pr_auc:.4f})")
    ax2.axhline(total_pos / len(labels), color="#718096", linestyle="--", lw=1.5, label=f"No Skill (Prevalence = {total_pos/len(labels):.4f})")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax2.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=False)
    ax2.grid(True, alpha=0.15)
    
    fig.tight_layout()
    plot_path_roc_pr = os.path.join(PROJECT_ROOT, "test_roc_pr_curves.png")
    fig.savefig(plot_path_roc_pr, dpi=200)
    plt.close(fig)
    print(f"Saved ROC/PR Curves to {plot_path_roc_pr}")
    
    # ----------------------------------------------------
    # PLOT 2: Calibration & Reliability Diagram
    # ----------------------------------------------------
    print("Generating Calibration Curve & Reliability Diagram...")
    valid_centers, true_props, pred_probs, all_bin_centers, bin_counts = compute_calibration_curve(labels, scores, n_bins=10)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Calibration Curve
    ax1.plot(pred_probs, true_props, marker="o", color="#d69e2e", lw=2, label="Model Calibration")
    ax1.plot([0, 1], [0, 1], color="#718096", linestyle="--", lw=1.5, label="Perfect Calibration")
    ax1.set_ylabel("Observed Proportion of Positives", fontsize=11)
    ax1.set_title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, alpha=0.15)
    
    # Distribution Histogram
    ax2.bar(all_bin_centers, bin_counts, width=0.08, color="#4a5568", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.grid(True, alpha=0.15)
    
    fig.tight_layout()
    plot_path_calib = os.path.join(PROJECT_ROOT, "test_calibration_curve.png")
    fig.savefig(plot_path_calib, dpi=200)
    plt.close(fig)
    print(f"Saved Calibration Curve to {plot_path_calib}")
    
    # ----------------------------------------------------
    # PLOT 3: Decision Curve Analysis (DCA)
    # ----------------------------------------------------
    print("Generating Decision Curve Analysis (DCA)...")
    pts, nb_model, nb_all, nb_none = compute_decision_curve(labels, scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pts, nb_model, color="#e53e3e", lw=2.5, label="Multimodal Framework")
    ax.plot(pts, nb_all, color="#3182ce", linestyle="--", lw=1.5, label="Triage All Patients")
    ax.plot(pts, nb_none, color="#718096", linestyle=":", lw=1.5, label="Triage None")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, max(np.max(nb_model), np.max(nb_all)) + 0.05)
    ax.set_xlabel("Threshold Probability ($P_t$)", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Clinical Decision Curve Analysis (DCA)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.15)
    
    fig.tight_layout()
    plot_path_dca = os.path.join(PROJECT_ROOT, "test_decision_curve.png")
    fig.savefig(plot_path_dca, dpi=200)
    plt.close(fig)
    print(f"Saved Decision Curve to {plot_path_dca}")
    
    # ----------------------------------------------------
    # Write summary metrics to json/txt
    # ----------------------------------------------------
    metrics_summary_path = os.path.join(PROJECT_ROOT, "evaluation_metrics.txt")
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("      INDEPENDENT TEST SET REPORT - 95% CONFIDENCE INTERVALS\n")
        f.write("======================================================================\n\n")
        f.write(f"Total Test Cohort Size: {len(labels)}\n")
        f.write(f"Critical Escalation Cases: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}% prevalence)\n\n")
        f.write("BOOTSTRAP PERFORMANCE METRICS (95% CI):\n")
        for metric, (mean_val, low, high) in ci.items():
            if metric in ["roc_auc", "pr_auc"]:
                f.write(f"  - {metric.upper():<12}: {mean_val:6.4f} (95% CI: {low:.4f} to {high:.4f})\n")
            else:
                f.write(f"  - {metric.capitalize():<12}: {mean_val*100:6.2f}% (95% CI: {low*100:.2f}% to {high*100:.2f}%)\n")
        f.write("\n======================================================================\n")
    print(f"Saved textual report to {metrics_summary_path}")
    print("All evaluation components completed successfully!")

if __name__ == "__main__":
    main()
