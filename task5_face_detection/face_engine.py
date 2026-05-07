"""
Face Detection & Recognition Engine — Task 5 (CODSOFT AI Internship)
Uses OpenCV Haar Cascades + DNN face detector + face_recognition library.
"""

import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

# ── Constants ─────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = Path("known_faces")
ENCODINGS_FILE = KNOWN_FACES_DIR / "encodings.json"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_EYE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
CASCADE_SMILE_PATH = cv2.data.haarcascades + "haarcascade_smile.xml"

# DNN face detector model URLs (will be downloaded on first use)
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    name: str = "Unknown"
    similarity: float = 0.0
    has_eyes: bool = False
    has_smile: bool = False

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


# ── Face Detector ─────────────────────────────────────────────────────────────

class FaceDetector:
    """Multi-method face detector supporting Haar Cascades and DNN."""

    def __init__(self, method: str = "haar", scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size: int = 30, dnn_confidence: float = 0.5):
        self.method = method
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.dnn_confidence = dnn_confidence

        # Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(CASCADE_EYE_PATH)
        self.smile_cascade = cv2.CascadeClassifier(CASCADE_SMILE_PATH)

        # DNN model (optional)
        self.dnn_net = None
        if method == "dnn":
            self._load_dnn()

    def _load_dnn(self):
        """Load OpenCV DNN face detector."""
        if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
            self.dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        else:
            # Fallback to haar if DNN weights not available
            self.method = "haar"

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in image, returns list of FaceDetection objects."""
        if self.method == "dnn" and self.dnn_net:
            return self._detect_dnn(image)
        return self._detect_haar(image)

    def _detect_haar(self, image: np.ndarray) -> List[FaceDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections = []
        for (x, y, w, h) in (faces if len(faces) > 0 else []):
            roi_gray = gray[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            smile = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            
            detections.append(FaceDetection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                confidence=0.9,
                has_eyes=len(eyes) > 0,
                has_smile=len(smile) > 0,
            ))
        return detections

    def _detect_dnn(self, image: np.ndarray) -> List[FaceDetection]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections_raw = self.dnn_net.forward()
        
        detections = []
        for i in range(detections_raw.shape[2]):
            conf = float(detections_raw[0, 0, i, 2])
            if conf > self.dnn_confidence:
                box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                fw, fh = max(1, x2 - x1), max(1, y2 - y1)
                detections.append(FaceDetection(
                    x=x1, y=y1, w=fw, h=fh, confidence=conf
                ))
        return detections


# ── Face Recognizer ───────────────────────────────────────────────────────────

class FaceRecognizer:
    """Face recognition using face_recognition library (dlib-based)."""

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance
        self.known_encodings: Dict[str, List] = {}  # name → list of encodings
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        self._load_encodings()
        
        # Check if face_recognition is available
        try:
            import face_recognition as fr
            self.fr = fr
            self.available = True
        except ImportError:
            self.available = False

    def _load_encodings(self):
        if ENCODINGS_FILE.exists():
            with open(ENCODINGS_FILE) as f:
                data = json.load(f)
            self.known_encodings = {
                name: [np.array(enc) for enc in encs]
                for name, encs in data.items()
            }

    def _save_encodings(self):
        data = {
            name: [enc.tolist() for enc in encs]
            for name, encs in self.known_encodings.items()
        }
        with open(ENCODINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def register_face(self, name: str, image: np.ndarray) -> bool:
        """Register a new face with the given name."""
        if not self.available:
            return False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = self.fr.face_encodings(rgb)
        if not encodings:
            return False
        if name not in self.known_encodings:
            self.known_encodings[name] = []
        self.known_encodings[name].append(encodings[0])
        self._save_encodings()
        return True

    def recognize(self, image: np.ndarray, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Recognize detected faces in the image."""
        if not self.available or not self.known_encodings:
            return detections

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face locations from detections
        face_locations = [(d.y, d.x + d.w, d.y + d.h, d.x) for d in detections]
        
        if not face_locations:
            return detections

        encodings = self.fr.face_encodings(rgb, face_locations)
        
        known_names = []
        known_encs = []
        for name, encs in self.known_encodings.items():
            for enc in encs:
                known_names.append(name)
                known_encs.append(enc)

        for i, (enc, det) in enumerate(zip(encodings, detections)):
            if not known_encs:
                break
            distances = self.fr.face_distance(known_encs, enc)
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            if best_dist <= self.tolerance:
                det.name = known_names[best_idx]
                det.similarity = round(1.0 - best_dist, 3)
            else:
                det.name = "Unknown"
                det.similarity = round(1.0 - best_dist, 3)

        return detections

    def delete_face(self, name: str) -> bool:
        if name in self.known_encodings:
            del self.known_encodings[name]
            self._save_encodings()
            return True
        return False

    @property
    def registered_names(self) -> List[str]:
        return list(self.known_encodings.keys())

    @property
    def registered_count(self) -> int:
        return sum(len(v) for v in self.known_encodings.values())


# ── Image Processing ──────────────────────────────────────────────────────────

class FaceProcessor:
    """Processes images/frames with detection + recognition and renders overlays."""

    # Color palette (BGR)
    COLORS = {
        "known": (0, 255, 150),      # green
        "unknown": (0, 100, 255),    # orange-red
        "box": (99, 102, 241),       # indigo
        "text_bg": (15, 15, 30),
    }

    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer):
        self.detector = detector
        self.recognizer = recognizer

    def process(self, image: np.ndarray, do_recognize: bool = True) -> Tuple[np.ndarray, List[FaceDetection]]:
        """Detect (and optionally recognize) faces. Returns annotated image + detections."""
        detections = self.detector.detect(image)
        if do_recognize:
            detections = self.recognizer.recognize(image, detections)
        annotated = self.draw_detections(image.copy(), detections)
        return annotated, detections

    def draw_detections(self, image: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        for det in detections:
            color = self.COLORS["known"] if det.name != "Unknown" else self.COLORS["unknown"]
            x, y, w, h = det.x, det.y, det.w, det.h

            # Main bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Corner accents
            corner_len = min(w, h) // 6
            for cx, cy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
                cv2.line(image, (cx, cy), (cx + dx * corner_len, cy), color, 3)
                cv2.line(image, (cx, cy), (cx, cy + dy * corner_len), color, 3)

            # Label background
            label = f"{det.name}"
            if det.name != "Unknown" and det.similarity > 0:
                label += f" ({det.similarity:.0%})"
            conf_label = f"Conf: {det.confidence:.0%}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (lw, lh), _ = cv2.getTextSize(label, font, 0.55, 1)
            label_y = max(y - 10, lh + 5)
            cv2.rectangle(image, (x, label_y - lh - 8), (x + lw + 8, label_y + 4), self.COLORS["text_bg"], -1)
            cv2.putText(image, label, (x + 4, label_y), font, 0.55, color, 1, cv2.LINE_AA)

            # Eye/smile indicators
            if det.has_eyes:
                cv2.putText(image, "👁", (x + w - 30, y + 20), font, 0.4, (255, 220, 0), 1)
            if det.has_smile:
                cv2.putText(image, "😊", (x + w - 30, y + 40), font, 0.4, (0, 255, 200), 1)

        # Count overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (220, 35), (15, 15, 30), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        cv2.putText(image, f"Faces: {len(detections)}", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (99, 200, 241), 2)

        return image

    def extract_face_crops(self, image: np.ndarray, detections: List[FaceDetection],
                           size: int = 128) -> List[np.ndarray]:
        """Extract and resize face crops."""
        crops = []
        for det in detections:
            pad = 20
            x1, y1 = max(0, det.x - pad), max(0, det.y - pad)
            x2 = min(image.shape[1], det.x + det.w + pad)
            y2 = min(image.shape[0], det.y + det.h + pad)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(cv2.resize(crop, (size, size)))
        return crops
