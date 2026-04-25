#!/usr/bin/env python3
"""
PoC 3: Hybrid CSRT + Optical Flow + Global Motion Compensation tracker.

Architecture:
  Layer A — CSRT tracker for appearance-based ROI following
  Layer B — LK optical flow inside ROI for short-term motion bridging
  Layer C — Global camera motion estimation (features outside ROI, RANSAC affine)
  Fusion  — Kalman-filtered crop center with dead-zone controller
  Safety  — Confidence scoring, scale constraints, appearance validation, fallback

Usage:
    python stabilize_hybrid.py [--source 0] [--padding 80]

Controls:
    Mouse drag   - select ROI on wide-angle feed
    r            - re-select ROI
    +/-          - zoom in/out
    q/ESC        - quit
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# LK optical flow params
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

FEATURE_PARAMS = dict(
    maxCorners=150,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)

MIN_INLIERS = 4
MAX_SCALE_JUMP = 1.8
HIST_SIM_THRESHOLD = 0.25
MAX_LOST_FRAMES = 90
REACQUIRE_INTERVAL = 5


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


class KalmanCropController:
    """Kalman filter for smooth crop center + dead-zone."""
    def __init__(self, dead_zone=15.0, process_noise=1.0, measure_noise=4.0):
        self.dead_zone = dead_zone
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, cx, cy):
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.initialized = True

    def update(self, cx, cy):
        if not self.initialized:
            self.init(cx, cy)
            return cx, cy

        predicted = self.kf.predict()
        px, py = predicted[0, 0], predicted[1, 0]

        dx = cx - px
        dy = cy - py
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < self.dead_zone:
            measurement = np.array([[px], [py]], dtype=np.float32)
        else:
            measurement = np.array([[cx], [cy]], dtype=np.float32)

        corrected = self.kf.correct(measurement)
        return corrected[0, 0], corrected[1, 0]

    def predict_only(self):
        if not self.initialized:
            return 0, 0
        predicted = self.kf.predict()
        return predicted[0, 0], predicted[1, 0]


class HybridTracker:
    """Combined CSRT + optical flow + global motion compensation tracker.

    Exposed as a class for benchmark integration.
    """
    def __init__(self):
        self.csrt = None
        self.prev_gray = None
        self.roi_pts = None
        self.roi_corners = None  # 4 corners as float64
        self.original_size = None
        self.original_hist = None
        self.roi_cx = 0.0
        self.roi_cy = 0.0
        self.roi_w = 0.0
        self.roi_h = 0.0
        self.confidence = 0.0

    def init(self, frame, bbox):
        x, y, w, h = bbox
        self.csrt = cv2.TrackerCSRT.create()
        self.csrt.init(frame, (int(x), int(y), int(w), int(h)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.roi_cx = x + w / 2.0
        self.roi_cy = y + h / 2.0
        self.roi_w = float(w)
        self.roi_h = float(h)
        self.original_size = (w, h)

        self.roi_corners = np.array([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h],
        ], dtype=np.float64)

        ix, iy, iw, ih = int(x), int(y), int(w), int(h)
        fh_, fw_ = frame.shape[:2]
        ix, iy = max(0, ix), max(0, iy)
        iw = min(iw, fw_ - ix)
        ih = min(ih, fh_ - iy)
        if iw > 5 and ih > 5:
            roi_img = frame[iy:iy+ih, ix:ix+iw]
            self.original_hist = cv2.calcHist([roi_img], [0, 1, 2], None,
                                              [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(self.original_hist, self.original_hist)

        self._detect_roi_features(gray, bbox)

    def _detect_roi_features(self, gray, bbox):
        x, y, w, h = [int(v) for v in bbox]
        ih, iw = gray.shape[:2]
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = min(w, iw - x)
        h = min(h, ih - y)
        if w <= 0 or h <= 0:
            self.roi_pts = None
            return
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        self.roi_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)

    def _estimate_global_motion(self, prev_gray, gray, roi_bbox):
        """Estimate global (camera) motion from features OUTSIDE the ROI."""
        rx, ry, rw, rh = [int(v) for v in roi_bbox]
        fh, fw = prev_gray.shape[:2]
        margin = 30
        mask = np.ones(prev_gray.shape, dtype=np.uint8) * 255
        x1 = max(0, rx - margin)
        y1 = max(0, ry - margin)
        x2 = min(fw, rx + rw + margin)
        y2 = min(fh, ry + rh + margin)
        mask[y1:y2, x1:x2] = 0

        global_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask,
                                             maxCorners=200, qualityLevel=0.01,
                                             minDistance=10, blockSize=7)
        if global_pts is None or len(global_pts) < MIN_INLIERS:
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, global_pts, None, **LK_PARAMS)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, next_pts, None, **LK_PARAMS)

        fb_err = np.linalg.norm(global_pts - back_pts, axis=2).reshape(-1)
        good = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & (fb_err < 2.0)

        if np.sum(good) < MIN_INLIERS:
            return None

        src = global_pts[good]
        dst = next_pts[good]
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                   ransacReprojThreshold=3.0)
        if M is not None and inliers is not None and int(inliers.sum()) >= MIN_INLIERS:
            return M
        return None

    def _track_roi_flow(self, prev_gray, gray):
        """Track ROI features with LK optical flow, return updated center or None."""
        if self.roi_pts is None or len(self.roi_pts) < MIN_INLIERS:
            return None, 0.0

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, self.roi_pts, None, **LK_PARAMS)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, next_pts, None, **LK_PARAMS)

        fb_err = np.linalg.norm(self.roi_pts - back_pts, axis=2).reshape(-1)
        good = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & (fb_err < 2.0)

        if np.sum(good) < MIN_INLIERS:
            return None, 0.0

        src = self.roi_pts[good]
        dst = next_pts[good]
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                   ransacReprojThreshold=3.0)
        if M is None or inliers is None:
            return None, 0.0

        n_inliers = int(inliers.sum())
        confidence = n_inliers / max(1, len(src))
        if n_inliers < MIN_INLIERS:
            return None, 0.0

        ones = np.ones((self.roi_corners.shape[0], 1))
        pts_h = np.hstack([self.roi_corners, ones])
        new_corners = (M @ pts_h.T).T

        flow_cx = np.mean(new_corners[:, 0])
        flow_cy = np.mean(new_corners[:, 1])
        flow_w = np.linalg.norm(new_corners[1] - new_corners[0])
        flow_h = np.linalg.norm(new_corners[3] - new_corners[0])

        self.roi_pts = dst[inliers.ravel() == 1].reshape(-1, 1, 2)
        self.roi_corners = new_corners

        return (flow_cx, flow_cy, flow_w, flow_h), confidence

    def update(self, frame):
        """Returns (ok, (x, y, w, h) or None, confidence 0-1)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]

        # Layer A: CSRT tracker
        csrt_ok = False
        csrt_bbox = None
        if self.csrt is not None:
            try:
                csrt_ok, csrt_bbox = self.csrt.update(frame)
            except cv2.error:
                csrt_ok = False

        # Layer B: Optical flow inside ROI
        flow_result = None
        flow_conf = 0.0
        if self.prev_gray is not None:
            flow_result, flow_conf = self._track_roi_flow(self.prev_gray, gray)

        # Layer C: Global motion compensation
        global_M = None
        if self.prev_gray is not None:
            roi_bbox = (self.roi_cx - self.roi_w / 2, self.roi_cy - self.roi_h / 2,
                        self.roi_w, self.roi_h)
            global_M = self._estimate_global_motion(self.prev_gray, gray, roi_bbox)

        # --- Fusion ---
        fused_cx, fused_cy, fused_w, fused_h = self.roi_cx, self.roi_cy, self.roi_w, self.roi_h
        confidence = 0.0

        if csrt_ok and csrt_bbox is not None:
            cx_c = csrt_bbox[0] + csrt_bbox[2] / 2
            cy_c = csrt_bbox[1] + csrt_bbox[3] / 2
            w_c, h_c = csrt_bbox[2], csrt_bbox[3]

            if flow_result is not None and flow_conf > 0.3:
                fcx, fcy, fw_, fh_ = flow_result
                alpha = 0.6
                fused_cx = alpha * cx_c + (1 - alpha) * fcx
                fused_cy = alpha * cy_c + (1 - alpha) * fcy
                fused_w = alpha * w_c + (1 - alpha) * fw_
                fused_h = alpha * h_c + (1 - alpha) * fh_
                confidence = 0.5 + 0.3 * flow_conf
            else:
                fused_cx, fused_cy = cx_c, cy_c
                fused_w, fused_h = w_c, h_c
                confidence = 0.5

        elif flow_result is not None and flow_conf > 0.3:
            fused_cx, fused_cy, fused_w, fused_h = flow_result
            confidence = 0.3 * flow_conf

        elif global_M is not None:
            pt = np.array([[self.roi_cx, self.roi_cy, 1.0]])
            new_pt = (global_M @ pt.T).T
            fused_cx = new_pt[0, 0]
            fused_cy = new_pt[0, 1]
            confidence = 0.15

        # Validate scale
        ow, oh = self.original_size
        scale_ratio = max(fused_w / max(1, ow), fused_h / max(1, oh))
        if scale_ratio > MAX_SCALE_JUMP:
            fused_w = ow * MAX_SCALE_JUMP
            fused_h = oh * MAX_SCALE_JUMP
            confidence *= 0.5

        if scale_ratio < 0.3:
            fused_w = ow * 0.5
            fused_h = oh * 0.5
            confidence *= 0.5

        # Validate appearance
        if confidence > 0.2 and self.original_hist is not None:
            bx = int(fused_cx - fused_w / 2)
            by = int(fused_cy - fused_h / 2)
            bw, bh = int(fused_w), int(fused_h)
            bx, by = max(0, bx), max(0, by)
            bw = min(bw, fw - bx)
            bh = min(bh, fh - by)
            if bw > 5 and bh > 5:
                roi_img = frame[by:by+bh, bx:bx+bw]
                curr_hist = cv2.calcHist([roi_img], [0, 1, 2], None,
                                         [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(curr_hist, curr_hist)
                sim = cv2.compareHist(self.original_hist, curr_hist, cv2.HISTCMP_CORREL)
                if sim < HIST_SIM_THRESHOLD:
                    confidence *= 0.3

        fused_cx = max(0, min(fw, fused_cx))
        fused_cy = max(0, min(fh, fused_cy))

        self.roi_cx = fused_cx
        self.roi_cy = fused_cy
        self.roi_w = fused_w
        self.roi_h = fused_h
        self.confidence = confidence

        # Re-detect features if low
        if self.roi_pts is None or len(self.roi_pts) < 10:
            roi_rect = (int(fused_cx - fused_w / 2), int(fused_cy - fused_h / 2),
                        int(fused_w), int(fused_h))
            self._detect_roi_features(gray, roi_rect)

        self.prev_gray = gray

        if confidence > 0.15:
            bbox = (fused_cx - fused_w / 2, fused_cy - fused_h / 2, fused_w, fused_h)
            return True, bbox, confidence
        else:
            return False, None, confidence


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


def main():
    parser = argparse.ArgumentParser(description='Hybrid CSRT+OptFlow ROI Stabilizer')
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
    crop_ctrl = KalmanCropController(dead_zone=10.0)
    hybrid = None

    state = 'idle'
    last_good_crop = None
    lost_frame_count = 0
    cam_fail_count = 0

    win_main = 'Wide-Angle Feed'
    win_crop = 'Stabilized Crop (Hybrid CSRT+Flow)'
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_crop, cv2.WINDOW_NORMAL)

    selector = ROISelector(win_main)
    selector.begin()

    print('[Hybrid] Draw a box around the ROI. Press r to re-select, q to quit.')

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
            hybrid = HybridTracker()
            hybrid.init(frame, roi)
            smoother = SmoothedSize(alpha=0.25)
            smoother.update(roi[2], roi[3])
            crop_ctrl = KalmanCropController(dead_zone=10.0)
            crop_ctrl.init(roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
            last_good_crop = None
            state = 'tracking'
            lost_frame_count = 0
            selector.roi = None
            print(f'Tracking ROI: {roi}, padding: {padding}px')

        if state == 'tracking' and hybrid is not None:
            ok, bbox, conf = hybrid.update(frame)

            if ok and bbox is not None:
                x, y, w, h = bbox
                cx, cy = x + w / 2, y + h / 2
                sw, sh = smoother.update(w, h)

                smooth_cx, smooth_cy = crop_ctrl.update(cx, cy)

                cv2.rectangle(display, (int(x), int(y)),
                              (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.circle(display, (int(smooth_cx), int(smooth_cy)), 5, (0, 255, 255), -1)

                if hybrid.roi_pts is not None:
                    for pt in hybrid.roi_pts:
                        px, py = pt.ravel()
                        cv2.circle(display, (int(px), int(py)), 2, (0, 200, 200), -1)

                side = max(sw, sh) + padding * 2
                crop = extract_stable_crop(frame, smooth_cx, smooth_cy, side, side)

                if crop is not None and crop.size > 0:
                    out = args.output_size
                    crop_display = cv2.resize(crop, (out, out))
                    last_good_crop = crop_display.copy()

                    color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                    cv2.putText(crop_display, f'Conf: {conf:.2f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(crop_display, 'Hybrid CSRT+Flow', (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                lost_frame_count = 0
            else:
                lost_frame_count += 1
                if last_good_crop is not None:
                    crop_display = last_good_crop.copy()
                    cv2.rectangle(crop_display, (0, 0),
                                  (crop_display.shape[1]-1, crop_display.shape[0]-1),
                                  (0, 0, 255), 4)
                    cv2.putText(crop_display, f'LOW CONF ({conf:.2f})',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(display, f'LOW CONFIDENCE ({lost_frame_count}f)',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                if lost_frame_count > MAX_LOST_FRAMES:
                    state = 'idle'
                    hybrid = None
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
            hybrid = None
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
