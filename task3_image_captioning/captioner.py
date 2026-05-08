"""
Image Captioning Engine — Task 3  |  CODSOFT AI Internship
Fixed version: proper model_type default, num_return_sequences clamped to num_beams.
"""

from __future__ import annotations
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class CaptionResult:
    caption: str
    confidence: float
    model_name: str
    generate_ms: float
    image_size: Tuple[int, int]
    alternatives: List[str] = field(default_factory=list)


class ImageCaptioner:
    """
    Multi-strategy image captioning.
    Primary:  Salesforce/blip-image-captioning-base  (fast, good quality)
    Optional: Salesforce/blip-image-captioning-large (best, heavier)
    Fallback: nlpconnect/vit-gpt2-image-captioning   (smallest)
    """

    MODEL_OPTIONS = {
        "BLIP Large (Best Quality)": "Salesforce/blip-image-captioning-large",
        "BLIP Base (Balanced)":      "Salesforce/blip-image-captioning-base",
        "ViT-GPT2 (Fastest)":        "nlpconnect/vit-gpt2-image-captioning",
    }

    def __init__(self):
        self.processor        = None
        self.model            = None
        self._tokenizer       = None
        self._model_type: str = "blip"      # ← default avoids AttributeError
        self.loaded_model_name: Optional[str] = None
        self._available       = self._check_deps()

    # ── Dependency check ──────────────────────────────────────────────────────
    def _check_deps(self) -> bool:
        try:
            import transformers, torch  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Model loading ─────────────────────────────────────────────────────────
    def load_model(self, model_key: str = "BLIP Base (Balanced)") -> bool:
        if not self._available:
            return False
        model_id = self.MODEL_OPTIONS.get(model_key)
        if not model_id:
            return False
        if self.loaded_model_name == model_key:
            return True  # already loaded

        import torch
        try:
            if "blip" in model_id.lower():
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.processor = BlipProcessor.from_pretrained(model_id)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32
                )
                self._model_type = "blip"
            else:
                from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
                self.model     = VisionEncoderDecoderModel.from_pretrained(model_id)
                self.processor = ViTImageProcessor.from_pretrained(model_id)
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model_type = "vitgpt2"

            self.model.eval()
            self.loaded_model_name = model_key
            return True

        except Exception as exc:
            self.loaded_model_name = None
            raise RuntimeError(f"Failed to load '{model_key}': {exc}") from exc

    # ── Public caption API ────────────────────────────────────────────────────
    def caption(
        self,
        image: Image.Image,
        conditional_text: str = "",
        num_captions: int = 3,
        max_length: int = 50,
        num_beams: int = 5,
    ) -> CaptionResult:
        if not self._available or self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start = time.perf_counter()
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_size = image.size

        if self._model_type == "blip":
            captions = self._blip_caption(image, conditional_text, num_captions, max_length, num_beams)
        else:
            captions = self._vitgpt2_caption(image, num_captions, max_length)

        elapsed_ms = (time.perf_counter() - start) * 1000
        primary     = captions[0] if captions else "No caption generated."
        alternatives = captions[1:] if len(captions) > 1 else []
        confidence  = min(1.0, len(primary.split()) / 12.0)

        return CaptionResult(
            caption=primary,
            confidence=round(confidence, 3),
            model_name=self.loaded_model_name or "Unknown",
            generate_ms=round(elapsed_ms, 1),
            image_size=img_size,
            alternatives=alternatives,
        )

    # ── BLIP captioning ───────────────────────────────────────────────────────
    def _blip_caption(self, image, text, num_captions, max_length, num_beams) -> List[str]:
        import torch

        # num_return_sequences must be ≤ num_beams
        n_seqs   = min(num_captions, num_beams)
        n_beams  = max(n_seqs, num_beams)

        inputs = (
            self.processor(image, text, return_tensors="pt")
            if text.strip()
            else self.processor(image, return_tensors="pt")
        )

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=n_beams,
                num_return_sequences=n_seqs,
                early_stopping=True,
            )

        raw = [
            self.processor.decode(ids, skip_special_tokens=True).strip().capitalize()
            for ids in out
        ]
        # deduplicate preserving order
        seen, unique = set(), []
        for c in raw:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    # ── ViT-GPT2 captioning ───────────────────────────────────────────────────
    def _vitgpt2_caption(self, image, num_captions, max_length) -> List[str]:
        import torch

        n_seqs = min(num_captions, 4)
        pv = self.processor(images=[image], return_tensors="pt").pixel_values

        gen_kwargs = {
            "max_new_tokens": max_length,
            "num_beams": max(n_seqs, 4),
            "num_return_sequences": n_seqs,
            "early_stopping": True,
        }

        with torch.no_grad():
            out = self.model.generate(pv, **gen_kwargs)

        return [
            c.strip().capitalize()
            for c in self._tokenizer.batch_decode(out, skip_special_tokens=True)
        ]

    # ── Image analysis ────────────────────────────────────────────────────────
    def analyze_image(self, image: Image.Image) -> Dict:
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return {
            "width":            image.width,
            "height":           image.height,
            "aspect_ratio":     round(image.width / image.height, 2),
            "brightness":       round(float(arr.mean()), 1),
            "contrast":         round(float(arr.std()), 1),
            "dominant_channel": ["Red","Green","Blue"][int(np.argmax([r.mean(), g.mean(), b.mean()]))],
            "mean_rgb":         (int(r.mean()), int(g.mean()), int(b.mean())),
        }

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def available(self) -> bool:
        return self._available
