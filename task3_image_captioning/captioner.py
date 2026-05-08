"""
Image Captioning Engine — Task 3 (CODSOFT AI Internship)
Uses Salesforce BLIP (Bootstrapped Language-Image Pre-training) via HuggingFace Transformers.
Falls back to a lightweight alternative if BLIP is unavailable.
"""

from __future__ import annotations
import io
import time
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image


# ── Caption Result ────────────────────────────────────────────────────────────

@dataclass
class CaptionResult:
    caption: str
    confidence: float
    model_name: str
    generate_ms: float
    image_size: Tuple[int, int]
    alternatives: List[str] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


# ── Caption Engine ────────────────────────────────────────────────────────────

class ImageCaptioner:
    """
    Multi-strategy image captioning using pre-trained vision-language models.

    Primary:   Salesforce/blip-image-captioning-large  (best quality)
    Fallback1: Salesforce/blip-image-captioning-base   (faster)
    Fallback2: nlpconnect/vit-gpt2-image-captioning    (smallest)
    """

    MODEL_OPTIONS = {
        "BLIP Large (Best Quality)": "Salesforce/blip-image-captioning-large",
        "BLIP Base (Balanced)":      "Salesforce/blip-image-captioning-base",
        "ViT-GPT2 (Fastest)":        "nlpconnect/vit-gpt2-image-captioning",
    }

    def __init__(self):
        self.processor = None
        self.model = None
        self.loaded_model_name = None
        self._transformers_available = self._check_transformers()

    def _check_transformers(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def load_model(self, model_key: str = "BLIP Base (Balanced)") -> bool:
        """Load a model. Returns True on success."""
        if not self._transformers_available:
            return False

        model_id = self.MODEL_OPTIONS.get(model_key)
        if not model_id:
            return False

        if self.loaded_model_name == model_key:
            return True  # already loaded

        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

            if "blip" in model_id.lower():
                self.processor = BlipProcessor.from_pretrained(model_id)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32
                )
                self._model_type = "blip"
            else:
                # ViT-GPT2
                self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
                self.processor = ViTImageProcessor.from_pretrained(model_id)
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model_type = "vitgpt2"

            self.model.eval()
            self.loaded_model_name = model_key
            return True

        except Exception as e:
            self.loaded_model_name = None
            raise RuntimeError(f"Failed to load model '{model_key}': {e}")

    def caption(
        self,
        image: Image.Image,
        conditional_text: str = "",
        num_captions: int = 3,
        max_length: int = 50,
        num_beams: int = 5,
    ) -> CaptionResult:
        """Generate captions for the given PIL image."""

        if not self._transformers_available or self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        import torch

        start = time.perf_counter()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_size = image.size

        if self._model_type == "blip":
            results = self._caption_blip(
                image, conditional_text, num_captions, max_length, num_beams
            )
        else:
            results = self._caption_vitgpt2(image, num_captions, max_length)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Primary caption is the first (best beam)
        primary = results[0] if results else "Unable to generate caption."
        alternatives = results[1:] if len(results) > 1 else []

        # Rough confidence proxy: caption length normalized
        confidence = min(1.0, len(primary.split()) / 12)

        return CaptionResult(
            caption=primary,
            confidence=confidence,
            model_name=self.loaded_model_name or "Unknown",
            generate_ms=round(elapsed_ms, 1),
            image_size=img_size,
            alternatives=alternatives,
        )

    def _caption_blip(
        self, image: Image.Image,
        conditional_text: str,
        num_captions: int,
        max_length: int,
        num_beams: int,
    ) -> List[str]:
        import torch

        if conditional_text.strip():
            inputs = self.processor(image, conditional_text, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            # Beam search for best caption
            out_beam = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=min(num_captions, num_beams),
                early_stopping=True,
            )

        captions = [
            self.processor.decode(ids, skip_special_tokens=True).strip().capitalize()
            for ids in out_beam
        ]

        # De-duplicate
        seen = set()
        unique = []
        for c in captions:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _caption_vitgpt2(
        self, image: Image.Image, num_captions: int, max_length: int
    ) -> List[str]:
        import torch

        pixel_values = self.processor(images=[image], return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=4,
                num_return_sequences=min(num_captions, 4),
                early_stopping=True,
            )

        captions = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [c.strip().capitalize() for c in captions]

    # ── Image Analysis Utilities ───────────────────────────────────────────────

    def analyze_image(self, image: Image.Image) -> Dict:
        """Return basic image statistics."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        brightness = float(np.mean(arr))
        dominant_channel = ["Red", "Green", "Blue"][np.argmax([r.mean(), g.mean(), b.mean()])]
        contrast = float(np.std(arr))
        return {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.width / image.height, 2),
            "mode": image.mode,
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "dominant_channel": dominant_channel,
            "mean_rgb": (int(r.mean()), int(g.mean()), int(b.mean())),
        }

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def available(self) -> bool:
        return self._transformers_available
