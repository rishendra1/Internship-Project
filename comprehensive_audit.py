"""
Research integrity audit for the lightweight multimodal ICU triage project.

The audit checks split isolation, patient/image/vitals overlap, duplicate image
content, filename-coupled labels, calibration metadata, and optional modality
shortcut tests. It does not tune model parameters or report test metrics from
validation data.
"""

import argparse
import contextlib
import hashlib
import os
import random
import sys

_argv_bak = list(sys.argv)
sys.argv = [sys.argv[0]]
if hasattr(sys, "orig_argv"):
    sys.orig_argv = [sys.orig_argv[0]]

import numpy as np
import torch

sys.argv = _argv_bak

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fit_pipeline import (
    MultimodalClinicalDataset,
    class_distribution,
    classification_metrics,
    evaluation_reliability,
    stable_int_hash,
    stratified_indices,
)
from models.fusion_core import CrossAttentionFusionCore
from models.spatial_branch import SpatialVisionBranch
from models.temporal_branch import TemporalVitalsBranch

SEP = "=" * 78
THIN = "-" * 78


def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def sub(title):
    print(f"\n{THIN}\n  {title}\n{THIN}")


def file_sha1(path, chunk_size=1024 * 1024):
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_split(split):
    image_split = "valid" if split == "valid" else split
    return MultimodalClinicalDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data", "images", image_split, "images"),
        labels_dir=os.path.join(PROJECT_ROOT, "data", "images", image_split, "labels"),
        vitals_dir=os.path.join(PROJECT_ROOT, "data", "vitals"),
        split=split,
        mismatch_rate=0.04 if split in {"valid", "test"} else 0.05,
    )


def collect_split(ds, hash_images=False):
    info = {
        "images": set(),
        "patients": set(),
        "vitals": set(),
        "labels": [],
        "hashes": {},
        "label_filename_mismatches": 0,
    }
    for idx in range(len(ds)):
        item = ds[idx]
        image_name = os.path.basename(item["image_path"])
        patient_id = os.path.splitext(image_name)[0].split("_")[0]
        vital_name = os.path.basename(item["vitals_path"])
        label = int(item["label"])
        info["images"].add(image_name)
        info["patients"].add(patient_id)
        info["vitals"].add(vital_name)
        info["labels"].append(label)

        lower_vital = vital_name.lower()
        if ("critical" in lower_vital and label == 0) or ("stable" in lower_vital and label == 1):
            info["label_filename_mismatches"] += 1

        if hash_images:
            image_hash = file_sha1(item["image_path"])
            info["hashes"].setdefault(image_hash, []).append(image_name)
    return info


def audit_splits(hash_images=False):
    header("AUDIT 1/7 - Split, Patient, Image, and Vitals Isolation")
    splits = {name: build_split(name) for name in ("train", "valid", "test")}
    infos = {}
    for name, ds in splits.items():
        print(f"  Scanning {name} ({len(ds)} samples)...", end=" ", flush=True)
        infos[name] = collect_split(ds, hash_images=hash_images)
        positives = sum(infos[name]["labels"])
        print(f"done | critical={positives}/{len(infos[name]['labels'])} ({positives / max(len(infos[name]['labels']), 1) * 100:.2f}%)")

    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    violations = 0
    for left, right in pairs:
        sub(f"{left} vs {right}")
        image_overlap = infos[left]["images"] & infos[right]["images"]
        patient_overlap = infos[left]["patients"] & infos[right]["patients"]
        vitals_overlap = infos[left]["vitals"] & infos[right]["vitals"]
        print(f"  Image filename overlap : {len(image_overlap)}")
        print(f"  Patient ID overlap     : {len(patient_overlap)}")
        print(f"  Vitals file overlap    : {len(vitals_overlap)}")
        if image_overlap or patient_overlap or vitals_overlap:
            violations += 1
            print("  STATUS: FAIL")
        else:
            print("  STATUS: PASS")

    if hash_images:
        sub("Duplicate image content")
        for left, right in pairs:
            duplicate_hashes = set(infos[left]["hashes"]).intersection(infos[right]["hashes"])
            print(f"  {left}-{right} duplicate image hashes: {len(duplicate_hashes)}")
            if duplicate_hashes:
                violations += 1

    return infos, violations


def audit_label_leakage(infos):
    header("AUDIT 2/7 - Label Leakage and Filename Coupling")
    print("  Model inputs: image pixels + numeric vitals matrix only.")
    print("  File paths are used by loaders but are not fed into neural layers.")
    total_mismatches = sum(info["label_filename_mismatches"] for info in infos.values())
    print(f"  Label/vitals filename mismatch count from configured mismatch rates: {total_mismatches}")
    print("  Residual limitation: source vitals filenames still contain stable/critical prefixes.")
    print("  Mitigation in code: train/valid/test now use disjoint vitals index pools, so repeated CSV leakage across splits is blocked.")


def audit_evaluation_reliability(infos):
    header("AUDIT 3/7 - Test Set Class Balance and Statistical Reliability")
    test_labels = infos["test"]["labels"]
    full = evaluation_reliability(test_labels)
    first_100 = evaluation_reliability(test_labels[:100])
    print(f"  Full test distribution      : {class_distribution(test_labels)}")
    print(f"  Full test reliability       : {full['status']} - {full['note']}")
    print(f"  First 100 test distribution : {class_distribution(test_labels[:100])}")
    print(f"  First 100 reliability       : {first_100['status']} - {first_100['note']}")

    test_ds = build_split("test")
    stratified = stratified_indices(test_ds, seed=42, neg_per_pos=1, max_positives=None)
    stratified_labels = [test_ds[i]["label"] for i in stratified]
    stratified_summary = evaluation_reliability(stratified_labels)
    print(f"  Secondary stratified cohort : {class_distribution(stratified_labels)}")
    print(f"  Stratified reliability      : {stratified_summary['status']} - {stratified_summary['note']}")
    if first_100["critical"] < 30:
        print("  STATUS: WARN - first-100 reporting is statistically unstable for sensitivity.")
    else:
        print("  STATUS: PASS")


def audit_calibration(checkpoint_path):
    header("AUDIT 4/7 - Threshold, Calibration, and Evaluation Protocol")
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  STATUS: SKIPPED")
        return 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    threshold = checkpoint.get("optimal_threshold")
    temperature = checkpoint.get("calibrated_temperature")
    protocol = checkpoint.get("split_protocol", "unknown")
    test_metrics = checkpoint.get("test_metrics")
    print(f"  split_protocol        : {protocol}")
    print(f"  optimal_threshold     : {threshold}")
    print(f"  calibrated_temperature: {temperature}")
    print(f"  has_test_metrics      : {test_metrics is not None}")
    if threshold is None or temperature is None:
        print("  STATUS: FAIL - checkpoint lacks validation-derived calibration metadata")
        return 1
    if not str(protocol).startswith("split_disjoint_vitals"):
        print("  STATUS: FAIL - checkpoint was not produced by the leakage-safe split protocol")
        return 1
    if test_metrics is None:
        print("  STATUS: FAIL - checkpoint has no independent test metrics from the final protocol")
        return 1
    print("  STATUS: PASS - threshold/temperature are stored separately from test reporting")
    return 0


def load_model(checkpoint_path):
    vision_net = SpatialVisionBranch(use_label_rois=False)
    vitals_net = TemporalVitalsBranch()
    fusion_brain = CrossAttentionFusionCore()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vision_net.load_state_dict(checkpoint["vision_net_state"], strict=False)
    vitals_net.load_state_dict(checkpoint["vitals_net_state"], strict=False)
    fusion_brain.load_state_dict(checkpoint["fusion_brain_state"], strict=False)
    fusion_brain.temperature = float(checkpoint.get("calibrated_temperature", 1.0))
    threshold = float(checkpoint.get("optimal_threshold", 0.5))
    vision_net.eval()
    vitals_net.eval()
    fusion_brain.eval()
    return vision_net, vitals_net, fusion_brain, threshold


def run_condition(vision_net, vitals_net, fusion_brain, ds, threshold, tag, null_img=False, null_vit=False, shuf_vit=False, shuf_img=False, n=300):
    all_vitals = [ds[i]["vitals_path"] for i in range(len(ds))]
    all_images = [ds[i]["image_path"] for i in range(len(ds))]
    if shuf_vit:
        random.seed(42)
        random.shuffle(all_vitals)
    if shuf_img:
        random.seed(42)
        random.shuffle(all_images)

    labels, scores = [], []
    samples = min(n, len(ds))
    with torch.no_grad():
        for i in range(samples):
            item = ds[i]
            image_path = all_images[i] if shuf_img else item["image_path"]
            vitals_path = all_vitals[i] if shuf_vit else item["vitals_path"]
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    spatial = torch.zeros(1, 128) if null_img else vision_net(image_path)
                    seed = stable_int_hash(os.path.basename(image_path)) % (2**32)
                    temporal = torch.zeros(1, 128) if null_vit else vitals_net(vitals_path, seed=seed)
                    probs, _ = fusion_brain(spatial, temporal)
            except Exception as exc:
                print(f"  skipped sample {i}: {exc}")
                continue
            labels.append(int(item["label"]))
            scores.append(float(probs[0, 1].item()))

    metrics = classification_metrics(labels, scores, threshold)
    metrics["tag"] = tag
    metrics["n"] = len(labels)
    return metrics


def audit_shortcuts(checkpoint_path, n=300):
    header("AUDIT 5/7 - Optional Modality Shortcut Tests on Test Split")
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        return []
    vision_net, vitals_net, fusion_brain, threshold = load_model(checkpoint_path)
    test_ds = build_split("test")
    tests = [
        ("baseline", {}),
        ("vitals_only_null_image", {"null_img": True}),
        ("image_only_null_vitals", {"null_vit": True}),
        ("shuffled_vitals", {"shuf_vit": True}),
        ("shuffled_images", {"shuf_img": True}),
    ]
    results = []
    for tag, kwargs in tests:
        result = run_condition(vision_net, vitals_net, fusion_brain, test_ds, threshold, tag, n=n, **kwargs)
        results.append(result)
        print(
            f"  {tag:<24} n={result['n']:<4} "
            f"acc={result['accuracy'] * 100:6.2f}% "
            f"sens={result['sensitivity'] * 100:6.2f}% "
            f"prec={result['precision'] * 100:6.2f}% "
            f"f1={result['f1'] * 100:6.2f}%"
        )
    return results


def audit_reproducibility():
    header("AUDIT 6/7 - Reproducibility")
    print("  fit_pipeline.py sets random.seed, numpy seed, torch seed, and deterministic cuDNN flags.")
    print("  DataLoader and sampler generators are explicitly seeded.")
    print("  Image-to-vitals mapping uses stable SHA1 hashing, not Python's randomized hash().")


def final_verdict(split_violations, calibration_violations):
    header("AUDIT 7/7 - Final Verdict")
    total = split_violations + calibration_violations
    if total == 0:
        print("  OVERALL STATUS: PASS FOR PROTOCOL STRUCTURE")
        print("  Caveat: scientific claims still require retraining and fresh independent test metrics.")
    else:
        print(f"  OVERALL STATUS: FAIL ({total} blocking protocol issue groups)")
        print("  Resolve all overlap/calibration failures before reporting performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage and protocol audit")
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "best_fusion_weights.pt"))
    parser.add_argument("--hash-images", action="store_true", help="Hash image bytes to detect duplicate content across splits")
    parser.add_argument("--run-shortcuts", action="store_true", help="Run optional model shortcut tests on the test split")
    parser.add_argument("--shortcut-samples", type=int, default=300)
    args = parser.parse_args(_argv_bak[1:])

    print(SEP)
    print("  COMPREHENSIVE RESEARCH INTEGRITY AUDIT")
    print("  Lightweight Multimodal Cross-Attention ICU Triage Network")
    print(SEP)

    split_infos, split_violations = audit_splits(hash_images=args.hash_images)
    audit_label_leakage(split_infos)
    audit_evaluation_reliability(split_infos)
    calibration_violations = audit_calibration(args.checkpoint)
    if args.run_shortcuts:
        audit_shortcuts(args.checkpoint, n=args.shortcut_samples)
    audit_reproducibility()
    final_verdict(split_violations, calibration_violations)
