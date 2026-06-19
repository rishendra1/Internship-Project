import contextlib
import hashlib
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
import colorama
from colorama import Fore, Style

# Initialize colorama for clean Windows CLI coloring
colorama.init(autoreset=True)


def _stable_int_hash(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


class SpatialVisionBranch(nn.Module):
    def __init__(
        self,
        out_dim=128,
        compat_mode=False,
        use_label_rois=False,
        apply_clahe=True,
        bilateral_filter=True,
        preserve_aspect=True,
        roi_padding=0.16,
        fine_tune_backbone=False,
        clahe_clip_limit=1.6,
        clahe_tile_grid=(8, 8),
        bilateral_d=3,
        bilateral_sigma_color=25,
        bilateral_sigma_space=25,
        min_yolo_conf=0.20,
        max_yolo_rois=1,
        context_roi_weight=0.45,
    ):
        super(SpatialVisionBranch, self).__init__()
        self.out_dim = out_dim
        self.compat_mode = compat_mode
        self.use_label_rois = use_label_rois
        self.apply_clahe = apply_clahe
        self.bilateral_filter = bilateral_filter
        self.preserve_aspect = preserve_aspect
        self.roi_padding = roi_padding
        self.fine_tune_backbone = fine_tune_backbone
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = tuple(clahe_tile_grid)
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.min_yolo_conf = min_yolo_conf
        self.max_yolo_rois = max_yolo_rois
        self.context_roi_weight = context_roi_weight
        self.last_roi_metadata = []

        # Load a pre-trained MobileNetV3-Small backbone as our feature pyramid base.
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Modify entry convolution block to accept 1-channel grayscale chest radiographs.
        # Luminance conversion preserves the pretrained RGB filter scale better than summing channels.
        old_conv = base_model.features[0][0]
        base_model.features[0][0] = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            luminance = torch.tensor([0.2989, 0.5870, 0.1140], dtype=old_conv.weight.dtype).view(1, 3, 1, 1)
            gray_weight = (old_conv.weight * luminance.to(old_conv.weight.device)).sum(dim=1, keepdim=True)
            base_model.features[0][0].weight.copy_(gray_weight)

        # Isolate explicit feature blocks to process multi-scale image granularity.
        self.features_entry = base_model.features[0:3]  # Outputs 24 channels
        self.features_mid = base_model.features[3:9]    # Outputs 48 channels
        self.features_deep = base_model.features[9:]    # Outputs 576 channels

        self.mid_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.deep_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mid_proj = nn.Linear(48, out_dim // 2)
        self.deep_proj = nn.Linear(576, out_dim // 2)
        self.spatial_norm = nn.Identity()

        # Approximate grayscale ImageNet statistics after luminance conversion.
        self.register_buffer("image_mean", torch.tensor([0.449], dtype=torch.float32).view(1, 1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.226], dtype=torch.float32).view(1, 1, 1, 1), persistent=False)

        # Preserve YOLOv8n localization; keep it outside PyTorch module registration.
        # Lazy load YOLOv8n only if compat_mode is True to save RAM/latency in native mode.
        if self.compat_mode:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.yolo_model = [YOLO(os.path.join(root_dir, "yolov8n.pt"))]
        else:
            self.yolo_model = None

        # Learnable spatial attention pooling for non-compat mode
        self.spatial_attention = nn.Sequential(
            nn.Linear(624, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Cache frozen MobileNet feature vectors to speed CPU training after the first pass.
        self.feature_cache = {}

    def _enhance_grayscale(self, image):
        if self.compat_mode:
            return image
        enhanced = image
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
            enhanced = clahe.apply(enhanced)
        if self.bilateral_filter:
            enhanced = cv2.bilateralFilter(
                enhanced,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )

        low, high = np.percentile(enhanced, (1.0, 99.0))
        if high > low:
            enhanced = np.clip((enhanced.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        return enhanced

    def _pad_box(self, x1, y1, x2, y2, img_w, img_h):
        pad_w = int((x2 - x1) * self.roi_padding)
        pad_h = int((y2 - y1) * self.roi_padding)
        return (
            max(0, x1 - pad_w),
            max(0, y1 - pad_h),
            min(img_w - 1, x2 + pad_w),
            min(img_h - 1, y2 + pad_h),
        )

    def _sanitize_box(self, box, img_w, img_h):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(float(img_w * img_h), 1.0)
        if area_ratio < 0.015 or area_ratio > 0.92:
            return None
        return self._pad_box(x1, y1, x2, y2, img_w, img_h)

    def _box_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1)

    def _append_candidate(self, candidates, box, source, confidence, weight):
        for existing in candidates:
            if self._box_iou(existing["box"], box) > 0.85:
                if confidence > existing["confidence"]:
                    existing.update({"box": box, "source": source, "confidence": confidence, "weight": weight})
                return
        candidates.append({"box": box, "source": source, "confidence": confidence, "weight": weight})

    def _thoracic_context_box(self, img_w, img_h):
        # Deterministic chest context fallback. It avoids the previous hash-random crop.
        x1 = int(img_w * 0.08)
        y1 = int(img_h * 0.06)
        x2 = int(img_w * 0.92)
        y2 = int(img_h * 0.94)
        return (x1, y1, x2, y2)

    def _label_candidates(self, image_path, img_w, img_h):
        if not self.use_label_rois:
            return []
        abs_img_path = os.path.abspath(image_path)
        img_dir = os.path.dirname(abs_img_path)
        img_name = os.path.basename(abs_img_path)
        patient_id = os.path.splitext(img_name)[0]
        if "images" not in img_dir:
            return []

        label_path = os.path.join(img_dir.replace("images", "labels"), f"{patient_id}.txt")
        if not os.path.exists(label_path):
            return []

        candidates = []
        with open(label_path, "r", encoding="utf-8") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = [float(v) for v in parts[1:5]]
                except ValueError:
                    continue
                x1 = int((x_center - w / 2.0) * img_w)
                y1 = int((y_center - h / 2.0) * img_h)
                x2 = int((x_center + w / 2.0) * img_w)
                y2 = int((y_center + h / 2.0) * img_h)
                box = self._sanitize_box((x1, y1, x2, y2), img_w, img_h)
                if box is not None:
                    priority = 1.0 if class_id in [1, 4, 10, 12] else 0.75
                    candidates.append({"box": box, "source": "debug_label_roi", "confidence": priority, "weight": priority})
        return sorted(candidates, key=lambda item: item["confidence"], reverse=True)[: self.max_yolo_rois]

    def _yolo_candidates(self, image_path, img_w, img_h):
        candidates = []
        results = self.yolo_model[0](image_path, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return candidates

        boxes = []
        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence < self.min_yolo_conf:
                continue
            sanitized = self._sanitize_box(box.xyxy[0].cpu().numpy(), img_w, img_h)
            if sanitized is None:
                continue
            boxes.append((confidence, int(box.cls[0]), sanitized))

        for confidence, class_id, sanitized in sorted(boxes, key=lambda item: item[0], reverse=True)[: self.max_yolo_rois]:
            candidates.append(
                {
                    "box": sanitized,
                    "source": f"yolov8n_cls_{class_id}",
                    "confidence": confidence,
                    "weight": max(confidence, 0.35),
                }
            )
        return candidates

    def _deterministic_quadrants(self, img_w, img_h):
        # 1. Mediastinum (center of chest)
        mediastinum = (
            int(img_w * 0.30),
            int(img_h * 0.15),
            int(img_w * 0.70),
            int(img_h * 0.85)
        )
        # 2. Anatomical Left Lung (viewer's right)
        left_lung = (
            int(img_w * 0.50),
            int(img_h * 0.10),
            int(img_w * 0.90),
            int(img_h * 0.90)
        )
        # 3. Anatomical Right Lung (viewer's left)
        right_lung = (
            int(img_w * 0.10),
            int(img_h * 0.10),
            int(img_w * 0.50),
            int(img_h * 0.90)
        )
        # 4. Global Chest (thoracic context)
        global_chest = self._thoracic_context_box(img_w, img_h)
        
        return [
            {"box": mediastinum, "source": "mediastinum", "confidence": 1.0, "weight": 1.0},
            {"box": left_lung, "source": "left_lung", "confidence": 1.0, "weight": 1.0},
            {"box": right_lung, "source": "right_lung", "confidence": 1.0, "weight": 1.0},
            {"box": global_chest, "source": "global_chest", "confidence": 1.0, "weight": 1.0}
        ]

    def _collect_roi_candidates(self, image_path, img_w, img_h):
        if self.compat_mode:
            return [{"box": self._thoracic_context_box(img_w, img_h), "source": "compat_context", "confidence": 1.0, "weight": 1.0}]
        return self._deterministic_quadrants(img_w, img_h)

    def _resize_roi(self, roi_patch, size=224):
        if not self.preserve_aspect:
            return cv2.resize(roi_patch, (size, size))
        h, w = roi_patch.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((size, size), dtype=np.uint8)
        scale = min(size / h, size / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(roi_patch, (new_w, new_h))
        canvas = np.zeros((size, size), dtype=resized.dtype)
        top = (size - new_h) // 2
        left = (size - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    def _project_features(self, mid_vector, deep_vector):
        spatial_embeddings = torch.cat((self.mid_proj(mid_vector), self.deep_proj(deep_vector)), dim=1)
        return self.spatial_norm(spatial_embeddings)

    def forward(self, image_path):
        if not self.fine_tune_backbone and hasattr(self, "feature_cache") and image_path in self.feature_cache:
            cached = self.feature_cache[image_path]
            mid_cached = cached[:, :48].to(self.mid_proj.weight.device)
            deep_cached = cached[:, 48:].to(self.deep_proj.weight.device)
            return self._project_features(mid_cached, deep_cached)

        print(f"\n{Fore.CYAN}[STAGE 2.1: Spatial Input Ingestion]{Style.RESET_ALL}")
        print(f"  Reading chest radiograph from: {image_path}")

        raw_matrix = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw_matrix is None:
            print(f"  {Fore.RED}Failed to load image. Using deterministic zero fallback tensor.{Style.RESET_ALL}")
            return torch.zeros(1, self.out_dim, device=self.mid_proj.weight.device)

        raw_matrix = self._enhance_grayscale(raw_matrix)
        img_h, img_w = raw_matrix.shape[:2]
        print(f"  Loaded grayscale matrix. Dimensions: {Fore.WHITE}{img_h}x{img_w}")

        print(f"{Fore.CYAN}[STAGE 2.2: YOLOv8n ROI Detection + Thoracic Context Fallback]{Style.RESET_ALL}")
        candidates = self._collect_roi_candidates(image_path, img_w, img_h)
        self.last_roi_metadata = candidates
        for idx, item in enumerate(candidates, start=1):
            x1, y1, x2, y2 = item["box"]
            print(
                f"  ROI {idx}: source={item['source']} box={Fore.WHITE}[x1={x1}, y1={y1}, x2={x2}, y2={y2}] "
                f"confidence={item['confidence']:.3f} weight={item['weight']:.3f}"
            )

        print(f"{Fore.CYAN}[STAGE 2.3: Multi-ROI Cropping & Resizing]{Style.RESET_ALL}")
        resized_patches = []
        valid_candidates = []
        for item in candidates:
            x1, y1, x2, y2 = item["box"]
            roi_patch = raw_matrix[y1:y2, x1:x2]
            if roi_patch.size == 0:
                continue
            resized_patch = self._resize_roi(roi_patch, 224)
            resized_patches.append(resized_patch)
            valid_candidates.append(item)
            print(f"  {item['source']} crop dimensions: {Fore.WHITE}{roi_patch.shape[0]}x{roi_patch.shape[1]}")

        if not resized_patches:
            fallback_box = self._thoracic_context_box(img_w, img_h)
            x1, y1, x2, y2 = fallback_box
            resized_patches = [self._resize_roi(raw_matrix[y1:y2, x1:x2], 224)]
            valid_candidates = [{"box": fallback_box, "source": "thoracic_context", "confidence": 1.0, "weight": 1.0}]
            self.last_roi_metadata = valid_candidates

        device = self.mid_proj.weight.device
        patch_array = np.stack(resized_patches, axis=0)
        patch_tensor = torch.tensor(patch_array, dtype=torch.float32, device=device).unsqueeze(1) / 255.0
        patch_tensor = (patch_tensor - self.image_mean.to(device)) / self.image_std.to(device).clamp_min(1e-6)
        print(f"  Preprocessed multi-ROI tensor shape: {Fore.WHITE}{list(patch_tensor.shape)}")

        print(f"{Fore.CYAN}[STAGE 2.4: Spatial Deep Feature Extraction (MobileNetV3-Small)]{Style.RESET_ALL}")
        grad_context = contextlib.nullcontext() if self.fine_tune_backbone else torch.no_grad()
        with grad_context:
            x_feat = self.features_entry(patch_tensor)
            print(f"  MobileNetV3 features_entry output shape: {Fore.WHITE}{list(x_feat.shape)}")

            mid_features = self.features_mid(x_feat)
            print(f"  Mid-level features output shape: {Fore.WHITE}{list(mid_features.shape)}")
            mid_vectors = self.mid_pool(mid_features).view(len(valid_candidates), -1)

            deep_features = self.features_deep(mid_features)
            print(f"  Deep-level features output shape: {Fore.WHITE}{list(deep_features.shape)}")
            deep_vectors = self.deep_pool(deep_features).view(len(valid_candidates), -1)

            if self.compat_mode:
                weights = torch.tensor([max(item["weight"], 1e-3) for item in valid_candidates], dtype=torch.float32, device=device)
                weights = weights / weights.sum().clamp_min(1e-6)
                mid_vector = torch.sum(mid_vectors * weights.unsqueeze(1), dim=0, keepdim=True)
                deep_vector = torch.sum(deep_vectors * weights.unsqueeze(1), dim=0, keepdim=True)
            else:
                cand_feats = torch.cat([mid_vectors, deep_vectors], dim=1)
                attn_logits = self.spatial_attention(cand_feats)
                attn_weights = torch.softmax(attn_logits, dim=0)
                mid_vector = torch.sum(mid_vectors * attn_weights, dim=0, keepdim=True)
                deep_vector = torch.sum(deep_vectors * attn_weights, dim=0, keepdim=True)

            fused_spatial_multiscale = torch.cat((mid_vector, deep_vector), dim=1)
            print(f"  Weighted multi-scale vector shape: {Fore.WHITE}{list(fused_spatial_multiscale.shape)}")

            if not self.fine_tune_backbone and hasattr(self, "feature_cache"):
                self.feature_cache[image_path] = fused_spatial_multiscale.detach().cpu()

        spatial_embeddings = self._project_features(mid_vector, deep_vector)
        print(f"  Projected spatial embedding vector (Keys, Values) shape: {Fore.WHITE}{list(spatial_embeddings.shape)}")
        print(f"  Sample Keys/Values values (first 5 elements): {Fore.YELLOW}{spatial_embeddings[0][:5].detach().cpu().numpy()}")

        return spatial_embeddings
