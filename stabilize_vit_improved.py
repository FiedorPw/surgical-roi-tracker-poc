#!/usr/bin/env python3
"""
PoC 1: Improved ViT tracker with anti-drift protections.

Enhancements over baseline stabilize_tracker.py:
  - Scale-jump blocking: reject bbox that grows too fast vs original ROI
  - Appearance check: histogram correlation vs original template
  - Velocity-based prediction during lost state
  - Smarter re-acquisition with size/appearance validation

Usage:
    python stabilize_vit_improved.py [--source 0] [--padding 80]

Controls:
    Mouse drag   - select ROI on wide-angle feed
    r            - re-select ROI
    +/-          - zoom in/out (change padding)
    q/ESC        - quit
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCORE_LOST = 0.25
SCORE_WEAK = 0.40
REACQUIRE_INTERVAL = 5
MAX_SCALE_JUMP = 1.8
HIST_SIM_THRESHOLD = 0.3
MAX_LOST_FRAMES = 90


class ROISelector:
    def __init__(self, window_name):
        self.window = window_name
        self.dragging = False
        self.start = None
        self.end = None
        self.roi = None
        self.active = False
        cv2.setMouseCallback(self.window, self._on_mouse)

    def begin(self):
        self.active = True
        self.roi = None
        self.start = None
        self.end = None

    def _on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)
            x1 = min(self.start[0], self.end[0])
            y1 = min(self.start[1], self.end[1])
            x2 = max(self.start[0], self.end[0])
            y2 = max(self.start[1], self.end[1])
            w, h = x2 - x1, y2 - y1
            if w > 10 and h > 10:
                self.roi = (x1, y1, w, h)
                self.active = False

    def draw_on(self, frame):
        if self.active and self.start and self.end:
            cv2.rectangle(frame, self.start, self.end, (255, 255, 0), 2)


class SmoothedSize:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.w = None
        self.h = None

    def update(self, w, h):
        if self.w is None:
            self.w, self.h = w, h
        else:
            a = self.alpha
            self.w = a * w + (1 - a) * self.w
            self.h = a * h + (1 - a) * self.h
        return self.w, self.h


class VelocityPredictor:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_cx = None
        self.prev_cy = None
        self.vx = 0.0
        self.vy = 0.0

    def observe(self, cx, cy):
        if self.prev_cx is not None:
            raw_vx = cx - self.prev_cx
            raw_vy = cy - self.prev_cy
            self.vx = self.alpha * raw_vx + (1 - self.alpha) * self.vx
            self.vy = self.alpha * raw_vy + (1 - self.alpha) * self.vy
        self.prev_cx = cx
        self.prev_cy = cy

    def predict(self, cx, cy, n_steps=1):
        return cx + self.vx * n_steps, cy + self.vy * n_steps


def extract_stable_crop(frame, cx, cy, crop_w, crop_h):
    fh, fw = frame.shape[:2]
    cw, ch = int(crop_w), int(crop_h)
    if cw <= 0 or ch <= 0 or cw > fw or ch > fh:
        return None
    cx = max(cw / 2, min(fw - cw / 2, cx))
    cy = max(ch / 2, min(fh - ch / 2, cy))
    x1 = int(cx - cw / 2)
    y1 = int(cy - ch / 2)
    return frame[y1:y1+ch, x1:x1+cw]


def create_tracker():
    params = cv2.TrackerVit.Params()
    params.net = os.path.join(SCRIPT_DIR, 'models', 'vitTracker.onnx')
    return cv2.TrackerVit.create(params)


def compute_hist(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, fw - x), min(h, fh - y)
    if w < 5 or h < 5:
        return None
    roi = frame[y:y+h, x:x+w]
    hist = cv2.calcHist([roi], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def validate_bbox(bbox, original_size, original_hist, frame):
    x, y, w, h = bbox
    ow, oh = original_size
    scale_ratio = max(w / max(1, ow), h / max(1, oh))
    if scale_ratio > MAX_SCALE_JUMP:
        return False, "scale_jump"

    if original_hist is not None:
        curr_hist = compute_hist(frame, bbox)
        if curr_hist is not None:
            sim = cv2.compareHist(original_hist, curr_hist, cv2.HISTCMP_CORREL)
            if sim < HIST_SIM_THRESHOLD:
                return False, f"appearance_mismatch(sim={sim:.2f})"

    return True, "ok"


def try_reacquire(frame, predicted_bbox, original_size, original_hist):
    fh, fw = frame.shape[:2]
    x, y, w, h = predicted_bbox
    x = max(0, min(x, fw - w))
    y = max(0, min(y, fh - h))
    w = min(w, fw - x)
    h = min(h, fh - y)
    if w < 10 or h < 10:
        return None

    try:
        tracker = create_tracker()
        tracker.init(frame, (int(x), int(y), int(w), int(h)))
        ok, bbox = tracker.update(frame)
        if ok:
            score = tracker.getTrackingScore()
            if score > SCORE_WEAK:
                valid, reason = validate_bbox(bbox, original_size, original_hist, frame)
                if valid:
                    return tracker, bbox, score
    except cv2.error:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Improved ViT ROI Stabilizer')
    parser.add_argument('--source', default='0', help='Camera index or video file path')
    parser.add_argument('--padding', type=int, default=80)
    parser.add_argument('--output-size', type=int, default=512)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Error: cannot open {source}')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    padding = args.padding
    smoother = SmoothedSize(alpha=0.25)
    velocity = VelocityPredictor(alpha=0.5)
    tracker = None

    state = 'idle'
    last_good_bbox = None
    last_good_crop = None
    original_size = None
    original_hist = None
    lost_frame_count = 0
    cam_fail_count = 0

    win_main = 'Wide-Angle Feed'
    win_crop = 'Stabilized Crop (ViT Improved)'
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_crop, cv2.WINDOW_NORMAL)

    selector = ROISelector(win_main)
    selector.begin()

    print('[ViT Improved] Draw a box around the ROI. Press r to re-select, q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            cam_fail_count += 1
            if cam_fail_count > 5:
                cap.release()
                time.sleep(0.3)
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cam_fail_count = 0
            continue
        cam_fail_count = 0

        display = frame.copy()
        crop_display = None

        if selector.roi is not None:
            roi = selector.roi
            try:
                tracker = create_tracker()
                tracker.init(frame, roi)
            except cv2.error as e:
                print(f'Tracker init failed: {e}')
                selector.begin()
                continue
            smoother = SmoothedSize(alpha=0.25)
            smoother.update(roi[2], roi[3])
            velocity = VelocityPredictor(alpha=0.5)
            velocity.observe(roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
            last_good_bbox = roi
            last_good_crop = None
            original_size = (roi[2], roi[3])
            original_hist = compute_hist(frame, roi)
            state = 'tracking'
            selector.roi = None
            print(f'Tracking ROI: {roi}, padding: {padding}px')

        if state == 'tracking' and tracker is not None:
            try:
                ok, bbox = tracker.update(frame)
                score = tracker.getTrackingScore() if ok else 0.0
            except cv2.error:
                ok, score = False, 0.0

            fh, fw = frame.shape[:2]
            if ok:
                x, y, w, h = bbox
                if w < 5 or h < 5 or x + w < 0 or y + h < 0 or x > fw or y > fh:
                    ok = False

            if ok and score > SCORE_LOST:
                valid, reason = validate_bbox(bbox, original_size, original_hist, frame)
                if not valid:
                    ok = False
                    print(f'Rejected: {reason} (score={score:.2f})')

            if ok and score > SCORE_LOST:
                x, y, w, h = bbox
                cx, cy = x + w / 2, y + h / 2
                sw, sh = smoother.update(w, h)
                velocity.observe(cx, cy)
                last_good_bbox = (x, y, w, h)

                cv2.rectangle(display, (int(x), int(y)),
                              (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.circle(display, (int(cx), int(cy)), 5, (0, 255, 255), -1)

                side = max(sw, sh) + padding * 2
                crop = extract_stable_crop(frame, cx, cy, side, side)

                if crop is not None and crop.size > 0:
                    out = args.output_size
                    crop_display = cv2.resize(crop, (out, out))
                    last_good_crop = crop_display.copy()

                    color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
                    cv2.putText(crop_display, f'Score: {score:.2f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                state = 'lost'
                lost_frame_count = 0
                print(f'Lost (score={score:.2f}). Auto-reacquiring...')

        if state == 'lost' and last_good_bbox is not None:
            lost_frame_count += 1

            if last_good_crop is not None:
                crop_display = last_good_crop.copy()
                cv2.rectangle(crop_display, (0, 0),
                              (crop_display.shape[1]-1, crop_display.shape[0]-1),
                              (0, 0, 255), 4)
                cv2.putText(crop_display, 'REACQUIRING...', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            bx, by, bw, bh = [int(v) for v in last_good_bbox]
            pred_cx, pred_cy = velocity.predict(bx + bw / 2, by + bh / 2, lost_frame_count)
            pred_bbox = (int(pred_cx - bw / 2), int(pred_cy - bh / 2), bw, bh)

            cv2.rectangle(display, (pred_bbox[0], pred_bbox[1]),
                          (pred_bbox[0] + bw, pred_bbox[1] + bh), (0, 0, 255), 2)
            cv2.putText(display, f'LOST ({lost_frame_count}f) - reacquiring...',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if lost_frame_count % REACQUIRE_INTERVAL == 0:
                result = try_reacquire(frame, pred_bbox, original_size, original_hist)
                if result is not None:
                    tracker, bbox, score = result
                    x, y, w, h = bbox
                    smoother = SmoothedSize(alpha=0.25)
                    smoother.update(w, h)
                    velocity.observe(x + w / 2, y + h / 2)
                    last_good_bbox = (x, y, w, h)
                    state = 'tracking'
                    print(f'Re-acquired! (score={score:.2f})')

            if lost_frame_count > MAX_LOST_FRAMES:
                state = 'idle'
                tracker = None
                last_good_crop = None
                selector.begin()
                print('Lost for too long — please re-select ROI.')

        if state == 'idle' and not selector.active:
            cv2.putText(display, 'Draw ROI with mouse', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if selector.active:
            cv2.putText(display, 'Draw ROI with mouse', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            selector.draw_on(display)

        cv2.imshow(win_main, display)
        if crop_display is not None:
            cv2.imshow(win_crop, crop_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            state = 'idle'
            tracker = None
            last_good_crop = None
            selector.begin()
            print('Re-select ROI...')
        elif key == ord('+') or key == ord('='):
            padding = max(10, padding - 20)
            print(f'Zoom in -> padding: {padding}px')
        elif key == ord('-'):
            padding = min(400, padding + 20)
            print(f'Zoom out -> padding: {padding}px')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
