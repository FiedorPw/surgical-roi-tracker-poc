#!/usr/bin/env python3
"""
Synthetic benchmark for ROI tracker evaluation.

Generates controlled video sequences and measures tracker performance on:
  1. Normal tracking (smooth motion)
  2. Occlusion recovery
  3. Scale stability (object stays same size, should not jump to larger region)
  4. Fast motion
  5. Appearance distractor (similar object nearby)

Each tracker under test must implement a common interface — see TrackerAdapter below.

Usage:
    python benchmark.py                    # run all benchmarks on all trackers
    python benchmark.py --tracker vit      # run on single tracker
    python benchmark.py --visualize        # show video while benchmarking
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Frame size for synthetic scenes
FRAME_W, FRAME_H = 640, 480
NUM_FRAMES = 300


# ---------------------------------------------------------------------------
# Synthetic scene generators
# ---------------------------------------------------------------------------

def _draw_target(frame, cx, cy, w, h, color=(0, 180, 60), label="T", texture=True):
    """Draw a textured rectangle (simulating a trackable object like a hand)."""
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(FRAME_W, x2), min(FRAME_H, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    if texture and (x2 - x1) > 10 and (y2 - y1) > 10:
        for i in range(x1 + 5, x2, 12):
            for j in range(y1 + 5, y2, 12):
                c2 = tuple(max(0, min(255, v + 40)) for v in color)
                cv2.circle(frame, (i, j), 3, c2, -1)
    cv2.putText(frame, label, (x1 + 2, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def _draw_body(frame, cx, cy, body_w, body_h, color=(60, 60, 180)):
    """Draw a larger 'body' rectangle that contains the target (simulates full person)."""
    x1, y1 = int(cx - body_w / 2), int(cy - body_h / 2)
    x2, y2 = int(cx + body_w / 2), int(cy + body_h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(FRAME_W, x2), min(FRAME_H, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    for i in range(x1, x2, 20):
        cv2.line(frame, (i, y1), (i + 10, y2), (80, 80, 200), 1)


def _draw_occluder(frame, cx, cy, w, h, color=(40, 40, 40)):
    """Draw an occluding rectangle (tool/hand crossing over target)."""
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)


def _make_background(seed=42):
    """Create a textured background (simulates surgical field)."""
    rng = np.random.RandomState(seed)
    bg = np.full((FRAME_H, FRAME_W, 3), (140, 120, 110), dtype=np.uint8)
    for _ in range(200):
        x, y = rng.randint(0, FRAME_W), rng.randint(0, FRAME_H)
        r = rng.randint(5, 30)
        c = tuple(int(v) for v in rng.randint(100, 180, 3))
        cv2.circle(bg, (x, y), r, c, -1)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg


@dataclass
class GroundTruth:
    cx: float
    cy: float
    w: float
    h: float
    visible: bool = True


def generate_normal_motion(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """Target moves smoothly, no occlusion."""
    bg = _make_background(seed=10)
    frames, gts = [], []
    tw, th = 60, 45
    for i in range(n_frames):
        t = i / n_frames
        cx = 200 + 240 * np.sin(2 * np.pi * t)
        cy = 200 + 100 * np.cos(2 * np.pi * t * 0.7)
        frame = bg.copy()
        _draw_target(frame, cx, cy, tw, th)
        frames.append(frame)
        gts.append(GroundTruth(cx, cy, tw, th, True))
    return frames, gts


def generate_occlusion(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """Target moves, gets occluded by a crossing object, must re-acquire."""
    bg = _make_background(seed=20)
    frames, gts = [], []
    tw, th = 60, 45
    body_w, body_h = 160, 280
    for i in range(n_frames):
        t = i / n_frames
        # Target (hand) moves slowly
        tcx = 320 + 80 * np.sin(2 * np.pi * t * 0.5)
        tcy = 240 + 40 * np.cos(2 * np.pi * t * 0.3)

        # Body is nearby (person torso)
        bcx = tcx - 20
        bcy = tcy + 80

        # Occluder (tool) sweeps across
        occ_cx = 640 * t * 2 - 200  # moves left to right across frame
        occ_cy = tcy
        occ_w, occ_h = 80, 120

        # Check if target is occluded
        occ_overlap_x = abs(occ_cx - tcx) < (occ_w + tw) / 2
        occ_overlap_y = abs(occ_cy - tcy) < (occ_h + th) / 2
        visible = not (occ_overlap_x and occ_overlap_y)

        frame = bg.copy()
        _draw_body(frame, bcx, bcy, body_w, body_h)
        _draw_target(frame, tcx, tcy, tw, th)
        if 0 < occ_cx < FRAME_W + 100:
            _draw_occluder(frame, occ_cx, occ_cy, occ_w, occ_h)
        frames.append(frame)
        gts.append(GroundTruth(tcx, tcy, tw, th, visible))
    return frames, gts


def generate_scale_trap(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """
    Target (hand-sized) is on top of a larger 'body'. After brief occlusion,
    tracker should NOT expand bbox to cover the body.
    This is the specific failure mode from the ViT analysis.
    """
    bg = _make_background(seed=30)
    frames, gts = [], []
    tw, th = 60, 45
    body_w, body_h = 180, 300

    occ_start = int(n_frames * 0.35)
    occ_end = int(n_frames * 0.50)

    for i in range(n_frames):
        t = i / n_frames
        tcx = 320 + 60 * np.sin(2 * np.pi * t * 0.3)
        tcy = 160 + 30 * np.cos(2 * np.pi * t * 0.2)

        bcx = tcx
        bcy = tcy + 120

        frame = bg.copy()
        _draw_body(frame, bcx, bcy, body_w, body_h)

        is_occluded = occ_start <= i <= occ_end
        if not is_occluded:
            _draw_target(frame, tcx, tcy, tw, th)
        else:
            _draw_occluder(frame, tcx, tcy - 10, tw + 40, th + 40, color=(30, 30, 30))

        frames.append(frame)
        gts.append(GroundTruth(tcx, tcy, tw, th, not is_occluded))
    return frames, gts


def generate_fast_motion(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """Target moves quickly with sudden direction changes."""
    bg = _make_background(seed=40)
    frames, gts = [], []
    tw, th = 60, 45
    cx, cy = 320.0, 240.0
    vx, vy = 8.0, 5.0
    for i in range(n_frames):
        cx += vx
        cy += vy
        if cx < tw / 2 + 10 or cx > FRAME_W - tw / 2 - 10:
            vx = -vx * (0.8 + 0.4 * np.random.random())
        if cy < th / 2 + 10 or cy > FRAME_H - th / 2 - 10:
            vy = -vy * (0.8 + 0.4 * np.random.random())
        if i % 30 == 0:
            vx += np.random.randn() * 3
            vy += np.random.randn() * 3
        cx = np.clip(cx, tw / 2 + 5, FRAME_W - tw / 2 - 5)
        cy = np.clip(cy, th / 2 + 5, FRAME_H - th / 2 - 5)

        frame = bg.copy()
        _draw_target(frame, cx, cy, tw, th)
        frames.append(frame)
        gts.append(GroundTruth(cx, cy, tw, th, True))
    return frames, gts


def generate_distractor(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """Two similar-looking objects; tracker must stay on the original."""
    bg = _make_background(seed=50)
    frames, gts = [], []
    tw, th = 60, 45

    for i in range(n_frames):
        t = i / n_frames
        # Real target
        tcx = 250 + 80 * np.sin(2 * np.pi * t * 0.4)
        tcy = 240 + 50 * np.cos(2 * np.pi * t * 0.3)
        # Distractor (same appearance) crosses nearby
        dcx = 400 + 100 * np.sin(2 * np.pi * t * 0.6 + 1.0)
        dcy = 240 + 80 * np.cos(2 * np.pi * t * 0.5)

        frame = bg.copy()
        _draw_target(frame, dcx, dcy, tw, th, color=(0, 180, 60), label="D")
        _draw_target(frame, tcx, tcy, tw, th, color=(0, 180, 60), label="T")
        frames.append(frame)
        gts.append(GroundTruth(tcx, tcy, tw, th, True))
    return frames, gts


def _draw_face(frame, cx, cy, w, h):
    """Draw a 'face' — high-texture, high-gradient attractor with distinct features."""
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(FRAME_W, x2), min(FRAME_H, y2)
    # Skin tone base
    cv2.ellipse(frame, (int(cx), int(cy)), (int(w / 2), int(h / 2)),
                0, 0, 360, (140, 170, 220), -1)
    # Eyes — strong gradient features
    eye_y = int(cy - h * 0.12)
    for ex in [int(cx - w * 0.2), int(cx + w * 0.2)]:
        cv2.ellipse(frame, (ex, eye_y), (int(w * 0.1), int(h * 0.06)),
                    0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (ex, eye_y), int(w * 0.05), (40, 30, 20), -1)
    # Nose
    cv2.line(frame, (int(cx), int(cy - h * 0.05)),
             (int(cx), int(cy + h * 0.1)), (100, 130, 170), 2)
    # Mouth
    cv2.ellipse(frame, (int(cx), int(cy + h * 0.22)),
                (int(w * 0.15), int(h * 0.05)),
                0, 0, 180, (80, 80, 160), 2)
    # Hair / forehead edge — high contrast
    cv2.ellipse(frame, (int(cx), int(cy - h * 0.3)),
                (int(w * 0.45), int(h * 0.12)),
                0, 180, 360, (30, 30, 50), -1)


def _draw_hand(frame, cx, cy, w, h, color=(160, 190, 220)):
    """Draw a 'hand' — low-texture, uniform skin-tone blob."""
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(FRAME_W, x2), min(FRAME_H, y2)
    # Rounded rectangle (hand shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    # Subtle finger lines
    for i in range(4):
        fx = x1 + int((i + 1) * w / 5)
        cv2.line(frame, (fx, y1 + 5), (fx, y1 + int(h * 0.4)),
                 tuple(max(0, c - 20) for c in color), 1)
    # Knuckle dots
    for i in range(4):
        fx = x1 + int((i + 1) * w / 5)
        cv2.circle(frame, (fx, y1 + int(h * 0.5)), 3,
                   tuple(min(255, c + 15) for c in color), -1)


def generate_face_steal(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """
    CRITICAL TEST: Hand (target) passes in front of face.
    Face is a much stronger visual attractor — tracker should NOT jump to face.

    Sequence:
      0-30%:   hand far from face, normal tracking
      30-50%:  hand approaches face, enters face's vicinity
      50-60%:  hand crosses directly over face (partial overlap)
      60-70%:  hand moves away from face
      70-100%: hand continues independently
    """
    bg = _make_background(seed=60)
    frames, gts = [], []
    hand_w, hand_h = 65, 50
    face_w, face_h = 90, 110

    # Face is stationary (or nearly)
    face_cx, face_cy = 320.0, 200.0

    for i in range(n_frames):
        t = i / n_frames

        # Hand path: starts left, sweeps across face, continues right
        if t < 0.3:
            hand_cx = 120 + (200 * t / 0.3)
            hand_cy = 260
        elif t < 0.5:
            progress = (t - 0.3) / 0.2
            hand_cx = 320 + 40 * (progress - 0.5)
            hand_cy = 200 + 60 * (1 - abs(progress - 0.5) * 2)
        elif t < 0.7:
            progress = (t - 0.5) / 0.2
            hand_cx = 340 + 180 * progress
            hand_cy = 260 - 40 * (1 - progress)
        else:
            progress = (t - 0.7) / 0.3
            hand_cx = 520 - 100 * progress + 40 * np.sin(2 * np.pi * progress)
            hand_cy = 260 + 30 * np.cos(2 * np.pi * progress)

        frame = bg.copy()
        # Draw face first (background layer)
        _draw_face(frame, face_cx, face_cy, face_w, face_h)
        # Draw hand on top
        _draw_hand(frame, hand_cx, hand_cy, hand_w, hand_h)

        # Visibility: hand is always there, question is whether tracker sticks with it
        frames.append(frame)
        gts.append(GroundTruth(hand_cx, hand_cy, hand_w, hand_h, True))

    return frames, gts


def generate_face_steal_with_occlusion(n_frames=NUM_FRAMES) -> Tuple[List[np.ndarray], List[GroundTruth]]:
    """
    CRITICAL TEST #2: Hand crosses face, gets briefly occluded by a tool while near face.
    After occlusion clears, tracker must return to hand, NOT lock onto face.

    This is the worst-case scenario: hand near face + occlusion = maximum confusion.
    """
    bg = _make_background(seed=70)
    frames, gts = [], []
    hand_w, hand_h = 65, 50
    face_w, face_h = 90, 110

    face_cx, face_cy = 320.0, 200.0

    occ_start = int(n_frames * 0.45)
    occ_end = int(n_frames * 0.55)

    for i in range(n_frames):
        t = i / n_frames

        # Hand path: approaches face, hovers near it during occlusion, then leaves
        if t < 0.3:
            hand_cx = 130 + (190 * t / 0.3)
            hand_cy = 240
        elif t < 0.6:
            progress = (t - 0.3) / 0.3
            hand_cx = 320 + 30 * np.sin(2 * np.pi * progress)
            hand_cy = 220 + 20 * np.cos(2 * np.pi * progress)
        elif t < 0.8:
            progress = (t - 0.6) / 0.2
            hand_cx = 320 + 200 * progress
            hand_cy = 220 + 40 * progress
        else:
            progress = (t - 0.8) / 0.2
            hand_cx = 520 - 50 * progress
            hand_cy = 260 + 20 * np.sin(2 * np.pi * progress)

        is_occluded = occ_start <= i <= occ_end

        frame = bg.copy()
        _draw_face(frame, face_cx, face_cy, face_w, face_h)

        if not is_occluded:
            _draw_hand(frame, hand_cx, hand_cy, hand_w, hand_h)
        # Occluder passes over the hand (near the face)
        if is_occluded:
            occ_progress = (i - occ_start) / max(1, occ_end - occ_start)
            occ_cx = 250 + 140 * occ_progress
            occ_cy = hand_cy - 10
            _draw_occluder(frame, occ_cx, occ_cy, 100, 70, color=(30, 30, 40))

        frames.append(frame)
        gts.append(GroundTruth(hand_cx, hand_cy, hand_w, hand_h, not is_occluded))

    return frames, gts


SCENARIOS = {
    'normal':           generate_normal_motion,
    'occlusion':        generate_occlusion,
    'scale_trap':       generate_scale_trap,
    'fast_motion':      generate_fast_motion,
    'distractor':       generate_distractor,
    'face_steal':       generate_face_steal,
    'face_steal_occ':   generate_face_steal_with_occlusion,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    tracker_name: str
    center_errors: List[float] = field(default_factory=list)
    scale_ratios: List[float] = field(default_factory=list)
    lost_frames: int = 0
    total_frames: int = 0
    recovery_frames: int = 0     # frames after occlusion until re-lock
    fps: float = 0.0

    @property
    def mean_center_error(self) -> float:
        errs = [e for e in self.center_errors if e is not None]
        return float(np.mean(errs)) if errs else float('inf')

    @property
    def median_center_error(self) -> float:
        errs = [e for e in self.center_errors if e is not None]
        return float(np.median(errs)) if errs else float('inf')

    @property
    def mean_scale_ratio(self) -> float:
        rats = [r for r in self.scale_ratios if r is not None]
        return float(np.mean(rats)) if rats else float('inf')

    @property
    def scale_deviation(self) -> float:
        rats = [r for r in self.scale_ratios if r is not None]
        return float(np.std(rats)) if rats else float('inf')

    @property
    def lost_pct(self) -> float:
        return 100.0 * self.lost_frames / max(1, self.total_frames)

    @property
    def success_rate(self) -> float:
        good = sum(1 for e in self.center_errors if e is not None and e < 30)
        return 100.0 * good / max(1, self.total_frames)


# ---------------------------------------------------------------------------
# Tracker adapter interface
# ---------------------------------------------------------------------------

class TrackerAdapter:
    """Base class. Each PoC tracker must implement init/update."""
    name: str = "base"

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        raise NotImplementedError

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float, float, float]], float]:
        """Returns (ok, (x, y, w, h) or None, confidence 0-1)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in tracker adapters (loaded on demand)
# ---------------------------------------------------------------------------

class ViTTrackerAdapter(TrackerAdapter):
    name = "vit_improved"

    def __init__(self):
        from stabilize_vit_improved import create_tracker as _create
        self._create = _create
        self._tracker = None
        self._original_size = None
        self._original_hist = None
        self._last_good_bbox = None
        self._max_scale_jump = 1.8

    def init(self, frame, bbox):
        self._tracker = self._create()
        self._tracker.init(frame, bbox)
        self._original_size = (bbox[2], bbox[3])
        x, y, w, h = [int(v) for v in bbox]
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            self._original_hist = cv2.calcHist([roi], [0, 1, 2], None,
                                               [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(self._original_hist, self._original_hist)
        self._last_good_bbox = bbox

    def update(self, frame):
        if self._tracker is None:
            return False, None, 0.0
        try:
            ok, bbox = self._tracker.update(frame)
            score = self._tracker.getTrackingScore() if ok else 0.0
        except cv2.error:
            return False, None, 0.0

        if not ok or score < 0.25:
            return False, None, score

        x, y, w, h = bbox
        ow, oh = self._original_size
        scale_ratio = max(w / max(1, ow), h / max(1, oh))
        if scale_ratio > self._max_scale_jump:
            return False, None, score * 0.5

        ix, iy, iw, ih = int(x), int(y), int(w), int(h)
        fh_, fw_ = frame.shape[:2]
        ix, iy = max(0, ix), max(0, iy)
        iw = min(iw, fw_ - ix)
        ih = min(ih, fh_ - iy)
        if iw > 5 and ih > 5 and self._original_hist is not None:
            roi = frame[iy:iy+ih, ix:ix+iw]
            h2 = cv2.calcHist([roi], [0, 1, 2], None,
                              [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(h2, h2)
            sim = cv2.compareHist(self._original_hist, h2, cv2.HISTCMP_CORREL)
            if sim < 0.3:
                return False, None, score * 0.3

        self._last_good_bbox = bbox
        return True, bbox, score


class DaSiamRPNAdapter(TrackerAdapter):
    name = "dasiamrpn"

    def __init__(self):
        self._tracker = None
        self._original_size = None
        self._max_scale_jump = 2.0

    def init(self, frame, bbox):
        p = cv2.TrackerDaSiamRPN_Params()
        p.model = os.path.join(MODELS_DIR, 'dasiamrpn_model.onnx')
        p.kernel_cls1 = os.path.join(MODELS_DIR, 'dasiamrpn_kernel_cls1.onnx')
        p.kernel_r1 = os.path.join(MODELS_DIR, 'dasiamrpn_kernel_r1.onnx')
        self._tracker = cv2.TrackerDaSiamRPN.create(p)
        self._tracker.init(frame, bbox)
        self._original_size = (bbox[2], bbox[3])

    def update(self, frame):
        if self._tracker is None:
            return False, None, 0.0
        try:
            ok, bbox = self._tracker.update(frame)
            score = self._tracker.getTrackingScore() if ok else 0.0
        except cv2.error:
            return False, None, 0.0

        if not ok:
            return False, None, score

        x, y, w, h = bbox
        ow, oh = self._original_size
        scale_ratio = max(w / max(1, ow), h / max(1, oh))
        if scale_ratio > self._max_scale_jump:
            return False, None, score * 0.5

        return True, bbox, score


class HybridCSRTAdapter(TrackerAdapter):
    name = "hybrid_csrt"

    def __init__(self):
        from stabilize_hybrid import HybridTracker
        self._ht = HybridTracker()
        self._initialized = False

    def init(self, frame, bbox):
        self._ht.init(frame, bbox)
        self._initialized = True

    def update(self, frame):
        if not self._initialized:
            return False, None, 0.0
        ok, bbox, conf = self._ht.update(frame)
        return ok, bbox if ok else None, conf


def get_adapter(name: str) -> TrackerAdapter:
    if name == 'vit_improved':
        return ViTTrackerAdapter()
    elif name == 'dasiamrpn':
        return DaSiamRPNAdapter()
    elif name == 'hybrid_csrt':
        return HybridCSRTAdapter()
    else:
        raise ValueError(f'Unknown tracker: {name}')


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_scenario(adapter: TrackerAdapter, frames: List[np.ndarray],
                 gts: List[GroundTruth], scenario_name: str,
                 visualize: bool = False) -> ScenarioResult:
    result = ScenarioResult(scenario=scenario_name, tracker_name=adapter.name,
                            total_frames=len(frames))

    gt0 = gts[0]
    init_bbox = (int(gt0.cx - gt0.w / 2), int(gt0.cy - gt0.h / 2),
                 int(gt0.w), int(gt0.h))
    adapter.init(frames[0], init_bbox)

    in_occlusion = False
    recovery_start = None
    t_start = time.time()

    for i in range(1, len(frames)):
        gt = gts[i]
        ok, bbox, conf = adapter.update(frames[i])

        if ok and bbox is not None:
            x, y, w, h = bbox
            pred_cx = x + w / 2
            pred_cy = y + h / 2
            center_err = np.sqrt((pred_cx - gt.cx) ** 2 + (pred_cy - gt.cy) ** 2)
            scale_ratio = max(w / max(1, gt.w), h / max(1, gt.h))
            result.center_errors.append(center_err)
            result.scale_ratios.append(scale_ratio)

            if recovery_start is not None and center_err < 30:
                result.recovery_frames = i - recovery_start
                recovery_start = None
        else:
            result.center_errors.append(None)
            result.scale_ratios.append(None)
            result.lost_frames += 1

        if not gt.visible and not in_occlusion:
            in_occlusion = True
        elif gt.visible and in_occlusion:
            in_occlusion = False
            if recovery_start is None:
                recovery_start = i

        if visualize:
            vis = frames[i].copy()
            # GT box green
            gx1 = int(gt.cx - gt.w / 2)
            gy1 = int(gt.cy - gt.h / 2)
            cv2.rectangle(vis, (gx1, gy1), (gx1 + int(gt.w), gy1 + int(gt.h)),
                          (0, 255, 0), 1)
            if ok and bbox is not None:
                bx, by, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
            cv2.putText(vis, f'{adapter.name} | {scenario_name} | f{i}',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Benchmark', vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    elapsed = time.time() - t_start
    result.fps = len(frames) / max(0.001, elapsed)

    if visualize:
        cv2.destroyWindow('Benchmark')

    return result


def print_results(all_results: Dict[str, List[ScenarioResult]]):
    scenarios = list(SCENARIOS.keys())
    trackers = list(all_results.keys())

    print('\n' + '=' * 100)
    print('BENCHMARK RESULTS')
    print('=' * 100)

    # Per-scenario tables
    for sc in scenarios:
        print(f'\n--- {sc.upper()} ---')
        print(f'{"Tracker":<20} {"MeanErr":>8} {"MedErr":>8} {"ScaleRatio":>11} {"ScaleDev":>9} '
              f'{"Lost%":>7} {"Success%":>9} {"Recovery":>9} {"FPS":>7}')
        print('-' * 100)
        for tr in trackers:
            r = [x for x in all_results[tr] if x.scenario == sc]
            if not r:
                continue
            r = r[0]
            rec = f'{r.recovery_frames}f' if r.recovery_frames > 0 else '-'
            print(f'{r.tracker_name:<20} {r.mean_center_error:>8.1f} {r.median_center_error:>8.1f} '
                  f'{r.mean_scale_ratio:>11.2f} {r.scale_deviation:>9.3f} '
                  f'{r.lost_pct:>6.1f}% {r.success_rate:>8.1f}% {rec:>9} {r.fps:>7.1f}')

    # Aggregate summary
    print(f'\n{"=" * 100}')
    print('AGGREGATE SUMMARY')
    print(f'{"=" * 100}')
    print(f'{"Tracker":<20} {"AvgMeanErr":>11} {"AvgSuccess%":>12} {"AvgLost%":>9} {"AvgScaleDev":>12} {"AvgFPS":>8}')
    print('-' * 80)
    for tr in trackers:
        results = all_results[tr]
        avg_err = np.mean([r.mean_center_error for r in results if r.mean_center_error < 1e6])
        avg_succ = np.mean([r.success_rate for r in results])
        avg_lost = np.mean([r.lost_pct for r in results])
        avg_sd = np.mean([r.scale_deviation for r in results if r.scale_deviation < 1e6])
        avg_fps = np.mean([r.fps for r in results])
        print(f'{tr:<20} {avg_err:>11.1f} {avg_succ:>11.1f}% {avg_lost:>8.1f}% {avg_sd:>12.3f} {avg_fps:>8.1f}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Tracker Benchmark')
    parser.add_argument('--tracker', nargs='*', default=None,
                        help='Tracker names to test (default: all)')
    parser.add_argument('--scenario', nargs='*', default=None,
                        help='Scenario names to run (default: all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show video during benchmark')
    args = parser.parse_args()

    tracker_names = args.tracker or ['vit_improved', 'dasiamrpn', 'hybrid_csrt']
    scenario_names = args.scenario or list(SCENARIOS.keys())

    print(f'Trackers: {tracker_names}')
    print(f'Scenarios: {scenario_names}')
    print()

    # Pre-generate all scenarios
    print('Generating synthetic sequences...')
    scenario_data = {}
    for sc_name in scenario_names:
        if sc_name not in SCENARIOS:
            print(f'  Warning: unknown scenario "{sc_name}", skipping')
            continue
        frames, gts = SCENARIOS[sc_name]()
        scenario_data[sc_name] = (frames, gts)
        n_visible = sum(1 for g in gts if g.visible)
        print(f'  {sc_name}: {len(frames)} frames, {n_visible} visible')

    all_results: Dict[str, List[ScenarioResult]] = {}

    for tr_name in tracker_names:
        print(f'\nTesting tracker: {tr_name}')
        all_results[tr_name] = []

        for sc_name, (frames, gts) in scenario_data.items():
            try:
                adapter = get_adapter(tr_name)
            except Exception as e:
                print(f'  ERROR creating {tr_name}: {e}')
                break

            print(f'  Running {sc_name}...', end=' ', flush=True)
            try:
                result = run_scenario(adapter, frames, gts, sc_name,
                                      visualize=args.visualize)
                all_results[tr_name].append(result)
                print(f'OK (err={result.mean_center_error:.1f}, '
                      f'lost={result.lost_pct:.1f}%, '
                      f'scale_dev={result.scale_deviation:.3f}, '
                      f'fps={result.fps:.0f})')
            except Exception as e:
                print(f'FAILED: {e}')
                import traceback
                traceback.print_exc()

    print_results(all_results)


if __name__ == '__main__':
    main()
