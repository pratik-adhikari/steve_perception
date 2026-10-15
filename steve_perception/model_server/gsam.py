"""
gsam.py — Grounded SAM: text prompt -> boxes (GroundingDINO) -> masks (SAM).

Runs in the vgn_env interpreter (torch), inside server.py. Sources and weights are external
(not pip-installed): the GroundingDINO and segment-anything repos are put on sys.path.

GroundingDINO's multi-scale deformable attention is a compiled CUDA op (groundingdino._C).
Without nvcc it cannot be built, and on CUDA tensors GroundingDINO would call the missing op;
_use_pytorch_deform_attn() routes it to the library's own pure-PyTorch implementation,
which runs on the GPU as well (grid_sample), only somewhat slower.
"""

import sys
import time
import types
from pathlib import Path

import numpy as np


def _use_pytorch_deform_attn():
    from groundingdino.models.GroundingDINO import ms_deform_attn as m
    try:
        from groundingdino import _C  # noqa: F401  compiled op available: keep it
        return False
    except ImportError:
        pass

    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return m.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

    m.MultiScaleDeformableAttnFunction = types.SimpleNamespace(apply=apply)
    return True


class GroundedSAM:
    def __init__(self, gdino_repo, gdino_config, gdino_weights, sam_repo, sam_weights, sam_type="vit_b",
                 device="cuda", image_size=800, fp16=True):
        import torch
        for p in (gdino_repo, sam_repo):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        t0 = time.monotonic()
        patched = _use_pytorch_deform_attn()
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        from segment_anything import SamPredictor, sam_model_registry

        self.torch, self.device, self.image_size = torch, device, image_size
        # fp16 autocast: GroundingDINO ~105 -> 85 ms, SAM ViT-B encoder ~230 -> 120 ms (RTX 3070 laptop,
        # 640x480), half the attention memory, same masks
        self.fp16 = fp16 and device.startswith("cuda")
        cfg = SLConfig.fromfile(str(gdino_config))
        cfg.device = device
        self.gdino = build_model(cfg)
        ckpt = torch.load(str(gdino_weights), map_location="cpu", weights_only=False)
        self.gdino.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        self.gdino.eval().to(device)
        self.tokenizer = self.gdino.tokenizer

        sam = sam_model_registry[sam_type](checkpoint=str(sam_weights))
        sam.to(device).eval()
        self.sam = SamPredictor(sam)
        print(f"[model_server] Grounded SAM ready in {time.monotonic() - t0:.1f} s "
              f"(GroundingDINO {Path(gdino_weights).name}, SAM {sam_type}"
              f"{', PyTorch deformable attention' if patched else ''}{', fp16' if self.fp16 else ''})", flush=True)

    def _gdino_input(self, rgb):
        """GroundingDINO's test transform: shorter side -> image_size, ImageNet normalisation."""
        torch = self.torch
        import torch.nn.functional as F
        h, w = rgb.shape[:2]
        s = self.image_size / min(h, w)
        if max(h, w) * s > 1333:
            s = 1333 / max(h, w)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float()[None] / 255.0
        x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def segment(self, rgb, prompt, box_threshold=0.35, text_threshold=0.25, max_dets=5, nms_iou=0.5):
        """rgb HxWx3 uint8 -> list of {box [x0 y0 x1 y1] px, score, phrase, mask HxW bool}, best first."""
        torch = self.torch
        from groundingdino.util.utils import get_phrases_from_posmap
        from torchvision.ops import box_convert, nms
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        h, w = rgb.shape[:2]
        caption = prompt.lower().strip()
        caption = caption if caption.endswith(".") else caption + "."
        amp = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16)
        with torch.no_grad(), amp():
            out = self.gdino(self._gdino_input(rgb), captions=[caption])
        logits = out["pred_logits"].float().sigmoid()[0]  # (queries, 256 tokens)
        boxes = out["pred_boxes"][0].float()              # (queries, 4) cx cy w h, normalised
        keep = logits.max(dim=1)[0] > box_threshold
        logits, boxes = logits[keep], boxes[keep]
        if len(boxes) == 0:
            return []
        scores = logits.max(dim=1)[0]
        xyxy = box_convert(boxes * torch.tensor([w, h, w, h], device=boxes.device), "cxcywh", "xyxy")
        idx = nms(xyxy, scores, nms_iou)[:max_dets]
        tokenized = self.tokenizer(caption)
        phrases = [get_phrases_from_posmap(logits[i] > text_threshold, tokenized, self.tokenizer).strip()
                   for i in idx.tolist()]
        xyxy, scores = xyxy[idx], scores[idx]

        with torch.no_grad(), amp():
            self.sam.set_image(rgb)
            tb = self.sam.transform.apply_boxes_torch(xyxy, (h, w))
            masks, _, _ = self.sam.predict_torch(point_coords=None, point_labels=None, boxes=tb,
                                                 multimask_output=False)
        masks = masks[:, 0].cpu().numpy()
        xyxy, scores = xyxy.cpu().numpy(), scores.cpu().numpy()
        return [{"box": xyxy[i].tolist(), "score": float(scores[i]), "phrase": phrases[i], "mask": masks[i]}
                for i in range(len(scores))]
