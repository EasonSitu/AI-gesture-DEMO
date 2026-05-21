import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
HOLD_SECONDS = 0.5
LOST_SECONDS_BEFORE_NO = 1.0
GESTURE_SWITCH_MARGIN = 0.08
FRAME_READ_RETRY_SECONDS = 10.0
WINDOW_NAME = "CIC 吊运手号识别 Demo"

ARM_SCORE_THRESHOLD = 0.60
ARM_HORIZONTAL_MAX_ANGLE_DEG = 30.0
WRIST_SHOULDER_MAX_Y_DIFF = 0.20
WRIST_OUTSIDE_MIN_SHOULDER_WIDTHS = 0.60
ELBOW_STRAIGHT_MIN_ANGLE = 140.0
MIN_LANDMARK_VISIBILITY = 0.45

PALM_VISIBLE_SCORE_THRESHOLD = 0.56
THUMB_UP_SCORE_THRESHOLD = 0.56
INDEX_DOWN_SCORE_THRESHOLD = 0.58
TWO_HAND_OVERHEAD_SCORE_THRESHOLD = 0.62
SLOW_RAISE_BOOM_SCORE_THRESHOLD = 0.56
HAND_WRIST_MATCH_MAX_DISTANCE = 0.24
STOP_PALM_MODE = "opposite_hand"

JPEG_QUALITY = 90
ROI_HANDS_ENABLED = True
ROI_BASE_SIZE = 260
ROI_SCALE = 2
ROI_MIN_SIZE = 220
ROI_MAX_SIZE = 360
DRAW_ALL_HANDS_DEBUG = True
ENABLE_IMAGE_ENHANCEMENT = False
OVEREXPOSURE_LUMA_THRESHOLD = 245
OVEREXPOSURE_WARN_RATIO = 0.08
MOTION_HISTORY_SECONDS = 1.4
MOTION_MIN_PATH = 0.12
MOTION_MIN_DIRECTION_CHANGES = 2
DEFAULT_AUTO_EXPOSURE = False
DEFAULT_EXPOSURE_VALUE = -6.0
EXPOSURE_MIN_VALUE = -13.0
EXPOSURE_MAX_VALUE = 0.0
EXPOSURE_STEP = 0.5
AUTO_EXPOSURE_MANUAL_VALUE = 0.25
AUTO_EXPOSURE_AUTO_VALUE = 0.75


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


CAMERA_BACKENDS = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("Default", cv2.CAP_ANY),
]

CAMERA_SETTINGS = [
    (1280, 720, 30),
    (1920, 1080, 30),
    (640, 480, 30),
]


class SimpleLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


class SimpleHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks


class StopGestureState:
    def __init__(self):
        self.confirmed_code = None
        self.candidate_code = None
        self.good_start_time = None
        self.lost_start_time = None
        self.motion_history = {"left": deque(), "right": deque()}

    @property
    def is_confirmed_stop(self):
        return self.confirmed_code == "STOP"

    def update(self, raw_gesture_code, now):
        if raw_gesture_code:
            if raw_gesture_code != self.candidate_code:
                self.candidate_code = raw_gesture_code
                self.good_start_time = now

            if self.good_start_time is None:
                self.good_start_time = now
            self.lost_start_time = None

            good_duration = now - self.good_start_time
            if good_duration >= HOLD_SECONDS:
                self.confirmed_code = raw_gesture_code

            return {
                "confirmed": self.confirmed_code is not None,
                "confirmed_code": self.confirmed_code,
                "raw_code": raw_gesture_code,
                "good_duration": good_duration,
                "lost_duration": 0.0,
                "hold_progress": clamp(good_duration / HOLD_SECONDS),
                "lost_progress": 0.0,
            }

        self.good_start_time = None
        self.candidate_code = None
        if self.confirmed_code is None:
            self.lost_start_time = None
            return {
                "confirmed": False,
                "confirmed_code": None,
                "raw_code": None,
                "good_duration": 0.0,
                "lost_duration": 0.0,
                "hold_progress": 0.0,
                "lost_progress": 0.0,
            }

        if self.lost_start_time is None:
            self.lost_start_time = now

        lost_duration = now - self.lost_start_time
        if lost_duration >= LOST_SECONDS_BEFORE_NO:
            self.confirmed_code = None

        return {
            "confirmed": self.confirmed_code is not None,
            "confirmed_code": self.confirmed_code,
            "raw_code": None,
            "good_duration": 0.0,
            "lost_duration": lost_duration,
            "hold_progress": 1.0 if self.confirmed_code else 0.0,
            "lost_progress": clamp(lost_duration / LOST_SECONDS_BEFORE_NO),
        }

    def update_motion(self, hand_candidates, now):
        for side in ("left", "right"):
            hand = select_best_hand(hand_candidates, side)
            if hand is None:
                continue
            point = hand["landmarks"].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
            history = self.motion_history[side]
            history.append((now, float(point.x), float(point.y)))
            while history and now - history[0][0] > MOTION_HISTORY_SECONDS:
                history.popleft()

    def motion_features(self, side):
        history = list(self.motion_history.get(side) or [])
        if len(history) < 4:
            return {
                "path": 0.0,
                "range": 0.0,
                "direction_changes": 0,
                "motion_score": 0.0,
            }

        points = np.array([(x, y) for _, x, y in history], dtype=np.float32)
        deltas = np.diff(points, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        path = float(np.sum(distances))
        x_range = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        y_range = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        motion_range = max(x_range, y_range)

        signs = np.sign(deltas)
        x_changes = int(np.sum(np.abs(np.diff(signs[:, 0])) > 1))
        y_changes = int(np.sum(np.abs(np.diff(signs[:, 1])) > 1))
        direction_changes = x_changes + y_changes

        path_score = clamp(path / MOTION_MIN_PATH)
        range_score = clamp(motion_range / 0.08)
        change_score = clamp(direction_changes / MOTION_MIN_DIRECTION_CHANGES)
        motion_score = clamp(path_score * 0.45 + range_score * 0.25 + change_score * 0.30)

        return {
            "path": round(path, 3),
            "range": round(motion_range, 3),
            "direction_changes": direction_changes,
            "motion_score": round(motion_score, 3),
        }


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def distance(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def angle_degrees(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
    bc = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0
    cosine = clamp(float(np.dot(ba, bc) / denominator), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def opposite_side(side):
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return None


def side_label(side):
    return {"left": "左手", "right": "右手"}.get(side, "未确定")


def gesture_label(code):
    return {
        "STOP": "停止",
        "RAISE_BOOM": "提升吊臂",
        "SLOW_RAISE_BOOM": "慢慢提升吊臂",
        "LOWER_HOIST": "吊钩下降",
        "STOP_EVERYTHING": "停止所有操作",
    }.get(code, "未识别")


GESTURE_TEST_OPTIONS = [
    ("STOP", gesture_label("STOP")),
    ("RAISE_BOOM", gesture_label("RAISE_BOOM")),
    ("SLOW_RAISE_BOOM", gesture_label("SLOW_RAISE_BOOM")),
    ("LOWER_HOIST", gesture_label("LOWER_HOIST")),
    ("STOP_EVERYTHING", gesture_label("STOP_EVERYTHING")),
]


def point_to_pixel(point, frame):
    height, width = frame.shape[:2]
    return int(point.x * width), int(point.y * height)


def analyze_exposure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overexposed_ratio = float(np.mean(gray >= OVEREXPOSURE_LUMA_THRESHOLD))
    mean_luma = float(np.mean(gray))
    return {
        "overexposed_ratio": round(overexposed_ratio, 3),
        "mean_luma": round(mean_luma, 1),
        "overexposed": overexposed_ratio >= OVEREXPOSURE_WARN_RATIO,
    }


def enhance_frame_for_detection(frame):
    if not ENABLE_IMAGE_ENHANCEMENT:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(enhanced_bgr, 0.75, frame, 0.25, 0)


def get_arm_landmarks(landmarks, side):
    if side == "right":
        return {
            "side": "right",
            "shoulder": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            "elbow": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            "wrist": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
        }

    return {
        "side": "left",
        "shoulder": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        "elbow": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        "wrist": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
    }


def pose_wrist(landmarks, side):
    if not landmarks or side not in ("left", "right"):
        return None
    return get_arm_landmarks(landmarks, side)["wrist"]


def shoulder_width_norm(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    return max(abs(left_shoulder.x - right_shoulder.x), 0.05)


def default_arm_match():
    return {
        "ok": False,
        "score": 0.0,
        "side": None,
        "wrist": None,
        "extended_ok": False,
        "horizontal_ok": False,
        "straight_ok": False,
        "arm_angle": None,
        "wrist_shoulder_y_diff": None,
        "elbow_angle": None,
        "side_extension_score": 0.0,
        "horizontal_score": 0.0,
        "elbow_straight_score": 0.0,
    }


def compute_arm_pose_score(landmarks):
    """Return the best side-extended, near-horizontal signal arm match."""
    shoulder_width = shoulder_width_norm(landmarks)
    best = default_arm_match()

    for side in ("right", "left"):
        arm = get_arm_landmarks(landmarks, side)
        shoulder = arm["shoulder"]
        elbow = arm["elbow"]
        wrist = arm["wrist"]

        visibility = min(shoulder.visibility, elbow.visibility, wrist.visibility)
        if visibility < MIN_LANDMARK_VISIBILITY:
            continue

        if side == "right":
            outside_distance = shoulder.x - wrist.x
        else:
            outside_distance = wrist.x - shoulder.x

        side_extension_score = clamp(
            outside_distance / (shoulder_width * WRIST_OUTSIDE_MIN_SHOULDER_WIDTHS)
        )
        extended_ok = side_extension_score >= 0.72

        wrist_shoulder_y_diff = abs(wrist.y - shoulder.y)
        arm_angle = abs(float(np.degrees(np.arctan2(wrist.y - shoulder.y, wrist.x - shoulder.x))))
        arm_angle = min(arm_angle, abs(180.0 - arm_angle))
        y_values = [shoulder.y, elbow.y, wrist.y]
        y_spread = max(y_values) - min(y_values)
        angle_score = clamp(1.0 - arm_angle / ARM_HORIZONTAL_MAX_ANGLE_DEG)
        y_diff_score = clamp(1.0 - wrist_shoulder_y_diff / WRIST_SHOULDER_MAX_Y_DIFF)
        y_spread_score = clamp(1.0 - y_spread / (WRIST_SHOULDER_MAX_Y_DIFF * 1.15))
        horizontal_score = angle_score * 0.45 + y_diff_score * 0.35 + y_spread_score * 0.20
        horizontal_ok = (
            arm_angle <= ARM_HORIZONTAL_MAX_ANGLE_DEG
            and wrist_shoulder_y_diff <= WRIST_SHOULDER_MAX_Y_DIFF
        )

        elbow_angle = angle_degrees(shoulder, elbow, wrist)
        elbow_straight_score = clamp(
            (elbow_angle - ELBOW_STRAIGHT_MIN_ANGLE) / (180.0 - ELBOW_STRAIGHT_MIN_ANGLE)
        )
        straight_ok = elbow_angle >= ELBOW_STRAIGHT_MIN_ANGLE

        score = (
            side_extension_score * 0.35
            + horizontal_score * 0.35
            + elbow_straight_score * 0.30
        )
        ok = score >= ARM_SCORE_THRESHOLD and extended_ok and horizontal_ok and straight_ok

        if score > best["score"]:
            best = {
                "ok": bool(ok),
                "score": clamp(score),
                "side": side,
                "wrist": wrist,
                "extended_ok": bool(extended_ok),
                "horizontal_ok": bool(horizontal_ok),
                "straight_ok": bool(straight_ok),
                "arm_angle": round(float(arm_angle), 1),
                "wrist_shoulder_y_diff": round(float(wrist_shoulder_y_diff), 3),
                "elbow_angle": round(float(elbow_angle), 1),
                "side_extension_score": round(float(side_extension_score), 3),
                "horizontal_score": round(float(horizontal_score), 3),
                "elbow_straight_score": round(float(elbow_straight_score), 3),
            }

    return best


def compute_palm_visible_score(hand_landmarks):
    """Return a loose 0..1 score for a visible, non-fist palm."""
    landmarks = hand_landmarks.landmark
    in_frame_count = sum(
        1 for point in landmarks if -0.08 <= point.x <= 1.08 and -0.08 <= point.y <= 1.08
    )
    completeness_score = in_frame_count / len(landmarks)

    palm_width = distance(
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value],
        landmarks[mp_hands.HandLandmark.PINKY_MCP.value],
    )
    palm_length = distance(
        landmarks[mp_hands.HandLandmark.WRIST.value],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value],
    )
    palm_ratio = palm_width / max(palm_length, 0.001)
    palm_facing_score = clamp((palm_ratio - 0.28) / 0.40)

    fingertip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
        mp_hands.HandLandmark.RING_FINGER_TIP.value,
        mp_hands.HandLandmark.PINKY_TIP.value,
    ]
    fingertip_distances = [
        distance(landmarks[tip], landmarks[mp_hands.HandLandmark.WRIST.value])
        for tip in fingertip_ids
    ]
    average_tip_distance = sum(fingertip_distances) / len(fingertip_distances)
    not_fist_score = clamp((average_tip_distance / max(palm_length, 0.001) - 0.85) / 0.65)

    return clamp(
        completeness_score * 0.30
        + palm_facing_score * 0.25
        + not_fist_score * 0.45
    )


def crop_hand_roi(frame, wrist, roi_size):
    height, width = frame.shape[:2]
    center_x = int(wrist.x * width)
    center_y = int(wrist.y * height)
    half = int(roi_size / 2)
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(width, center_x + half)
    y2 = min(height, center_y + half)
    if x2 - x1 < 30 or y2 - y1 < 30:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def convert_roi_hand_landmarks(hand_landmarks, rect, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = rect
    roi_w = x2 - x1
    roi_h = y2 - y1
    converted = []
    for point in hand_landmarks.landmark:
        converted.append(
            SimpleLandmark(
                (x1 + point.x * roi_w) / width,
                (y1 + point.y * roi_h) / height,
                point.z,
                1.0,
            )
        )
    return SimpleHandLandmarks(converted)


def assign_whole_hand_side(hand_landmarks, pose_landmarks):
    if not pose_landmarks:
        return None, None

    hand_wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value]
    distances = {}
    for side in ("left", "right"):
        wrist = pose_wrist(pose_landmarks, side)
        if wrist is not None:
            distances[side] = distance(hand_wrist, wrist)

    if not distances:
        return None, None

    side = min(distances, key=distances.get)
    return side, distances[side]


def collect_hand_candidates(frame, hands_model, whole_hands_results, pose_landmarks):
    candidates = []
    frame_h, frame_w = frame.shape[:2]
    roi_size = ROI_BASE_SIZE
    if pose_landmarks:
        roi_size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, shoulder_width_norm(pose_landmarks) * frame_w * 1.15)))

    if ROI_HANDS_ENABLED and pose_landmarks:
        for side in ("left", "right"):
            wrist = pose_wrist(pose_landmarks, side)
            if wrist is None or wrist.visibility < MIN_LANDMARK_VISIBILITY:
                continue

            roi, rect = crop_hand_roi(frame, wrist, roi_size)
            if roi is None:
                continue

            enlarged = cv2.resize(
                roi,
                None,
                fx=ROI_SCALE,
                fy=ROI_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            rgb_roi = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
            roi_results = hands_model.process(rgb_roi)
            if not roi_results.multi_hand_landmarks:
                continue

            for hand_landmarks in roi_results.multi_hand_landmarks:
                converted = convert_roi_hand_landmarks(hand_landmarks, rect, frame.shape)
                hand_wrist = converted.landmark[mp_hands.HandLandmark.WRIST.value]
                wrist_distance = distance(hand_wrist, wrist)
                score = compute_palm_visible_score(converted)
                candidates.append(
                    {
                        "landmarks": converted,
                        "side": side,
                        "source": "ROI",
                        "score": score,
                        "ok": score >= PALM_VISIBLE_SCORE_THRESHOLD,
                        "wrist_distance": wrist_distance,
                        "wrist_match": wrist_distance <= HAND_WRIST_MATCH_MAX_DISTANCE,
                        "roi_rect": rect,
                    }
                )

    if whole_hands_results.multi_hand_landmarks:
        for hand_landmarks in whole_hands_results.multi_hand_landmarks:
            assigned_side, wrist_distance = assign_whole_hand_side(hand_landmarks, pose_landmarks)
            score = compute_palm_visible_score(hand_landmarks)
            candidates.append(
                {
                    "landmarks": hand_landmarks,
                    "side": assigned_side,
                    "source": "whole",
                    "score": score,
                    "ok": score >= PALM_VISIBLE_SCORE_THRESHOLD,
                    "wrist_distance": wrist_distance,
                    "wrist_match": (
                        wrist_distance is not None
                        and wrist_distance <= HAND_WRIST_MATCH_MAX_DISTANCE
                    ),
                    "roi_rect": None,
                }
            )

    return candidates, roi_size


def default_palm_match(side=None):
    return {
        "ok": False,
        "score": 0.0,
        "matched": False,
        "wrist_match": False,
        "wrist_distance": None,
        "side": side,
        "source": None,
        "landmarks": None,
        "roi_rect": None,
    }


def select_palm_candidate(candidates, required_side):
    best = default_palm_match(required_side)
    for candidate in candidates:
        if candidate["side"] != required_side:
            continue
        if best["wrist_match"] and not candidate["wrist_match"]:
            continue
        candidate_prefers_roi = candidate["source"] == "ROI" and best["source"] != "ROI"
        if (
            (candidate["wrist_match"] and not best["wrist_match"])
            or candidate_prefers_roi
            or candidate["score"] > best["score"]
        ):
            best = {
                "ok": bool(candidate["ok"]),
                "score": clamp(candidate["score"]),
                "matched": True,
                "wrist_match": bool(candidate["wrist_match"]),
                "wrist_distance": candidate["wrist_distance"],
                "side": candidate["side"],
                "source": candidate["source"],
                "landmarks": candidate["landmarks"],
                "roi_rect": candidate["roi_rect"],
            }
    return best


def select_best_hand(candidates, side=None):
    best = None
    for candidate in candidates:
        if side is not None and candidate["side"] != side:
            continue
        if best is None:
            best = candidate
            continue
        candidate_prefers_roi = candidate["source"] == "ROI" and best["source"] != "ROI"
        if candidate_prefers_roi or candidate["score"] > best["score"]:
            best = candidate
    return best


def landmark_span(hand_landmarks):
    points = hand_landmarks.landmark
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return max(max(xs) - min(xs), max(ys) - min(ys), 0.001)


def thumb_up_score(hand_landmarks):
    points = hand_landmarks.landmark
    thumb_tip = points[mp_hands.HandLandmark.THUMB_TIP.value]
    thumb_ip = points[mp_hands.HandLandmark.THUMB_IP.value]
    thumb_mcp = points[mp_hands.HandLandmark.THUMB_MCP.value]
    index_mcp = points[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
    pinky_mcp = points[mp_hands.HandLandmark.PINKY_MCP.value]
    span = landmark_span(hand_landmarks)

    vertical_score = clamp((thumb_mcp.y - thumb_tip.y) / (span * 0.55))
    ordered_score = 1.0 if thumb_tip.y < thumb_ip.y < thumb_mcp.y else 0.35
    palm_width = distance(index_mcp, pinky_mcp)
    visible_score = clamp(palm_width / max(span * 0.35, 0.001))
    return clamp(vertical_score * 0.55 + ordered_score * 0.25 + visible_score * 0.20)


def index_down_score(hand_landmarks):
    points = hand_landmarks.landmark
    wrist = points[mp_hands.HandLandmark.WRIST.value]
    index_tip = points[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
    index_pip = points[mp_hands.HandLandmark.INDEX_FINGER_PIP.value]
    index_mcp = points[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
    span = landmark_span(hand_landmarks)

    downward_score = clamp((index_tip.y - index_mcp.y) / (span * 0.55))
    ordered_score = 1.0 if index_tip.y > index_pip.y > index_mcp.y else 0.35
    below_wrist_score = clamp((index_tip.y - wrist.y) / (span * 0.45))
    return clamp(downward_score * 0.50 + ordered_score * 0.25 + below_wrist_score * 0.25)


def index_up_score(hand_landmarks):
    points = hand_landmarks.landmark
    index_tip = points[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
    index_pip = points[mp_hands.HandLandmark.INDEX_FINGER_PIP.value]
    index_mcp = points[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
    span = landmark_span(hand_landmarks)
    upward_score = clamp((index_mcp.y - index_tip.y) / (span * 0.55))
    ordered_score = 1.0 if index_tip.y < index_pip.y < index_mcp.y else 0.35
    return clamp(upward_score * 0.70 + ordered_score * 0.30)


def gesture_result(code, ok, score, reason, details=None):
    return {
        "code": code,
        "label": gesture_label(code),
        "ok": bool(ok),
        "score": round(float(clamp(score)), 3),
        "reason": reason,
        "details": details or {},
    }


def build_hands_debug(candidates, stop_state=None):
    rows = []
    for candidate in candidates:
        landmarks = candidate["landmarks"]
        side = candidate.get("side")
        motion = stop_state.motion_features(side) if stop_state and side in ("left", "right") else {}
        rows.append(
            {
                "side": side,
                "label": side_label(side),
                "source": candidate.get("source"),
                "palm_score": round(float(candidate.get("score", 0.0)), 3),
                "thumb_up_score": round(float(thumb_up_score(landmarks)), 3),
                "index_up_score": round(float(index_up_score(landmarks)), 3),
                "index_down_score": round(float(index_down_score(landmarks)), 3),
                "wrist_distance": (
                    None
                    if candidate.get("wrist_distance") is None
                    else round(float(candidate["wrist_distance"]), 3)
                ),
                "wrist_match": bool(candidate.get("wrist_match")),
                "motion_score": motion.get("motion_score", 0.0),
                "motion_path": motion.get("path", 0.0),
                "motion_range": motion.get("range", 0.0),
                "direction_changes": motion.get("direction_changes", 0),
            }
        )
    return rows


def detect_raise_boom(arm_match, candidates):
    hand = select_best_hand(candidates, arm_match["side"])
    thumb_score = thumb_up_score(hand["landmarks"]) if hand else 0.0
    score = min(arm_match["score"], thumb_score)
    ok = bool(arm_match["ok"] and hand and thumb_score >= THUMB_UP_SCORE_THRESHOLD)
    reason = "水平伸出手臂，拇指向上。" if ok else "需要水平伸出手臂，并让同侧手拇指向上。"
    return gesture_result(
        "RAISE_BOOM",
        ok,
        score,
        reason,
        {
            "signal_arm_side": arm_match["side"],
            "thumb_score": round(float(thumb_score), 3),
        },
    )


def detect_slow_raise_boom(arm_match, candidates):
    signal_hand = select_best_hand(candidates, arm_match["side"])
    helper_hand = select_best_hand(candidates, opposite_side(arm_match["side"]))
    palm_score = compute_palm_visible_score(signal_hand["landmarks"]) if signal_hand else 0.0
    index_score = index_up_score(helper_hand["landmarks"]) if helper_hand else 0.0
    proximity_score = 0.0

    if signal_hand and helper_hand:
        signal_wrist = signal_hand["landmarks"].landmark[mp_hands.HandLandmark.WRIST.value]
        helper_index = helper_hand["landmarks"].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
        proximity_score = clamp(1.0 - distance(signal_wrist, helper_index) / 0.35)

    score = min(arm_match["score"], max(palm_score, 0.45), index_score, max(proximity_score, 0.45))
    ok = bool(
        arm_match["ok"]
        and signal_hand
        and helper_hand
        and palm_score >= 0.45
        and index_score >= 0.55
        and proximity_score >= 0.25
    )
    reason = "水平手臂配合另一只手食指向上，判定为慢慢提升吊臂。" if ok else "需要一只手臂水平，另一只手食指向上靠近水平手臂。"
    return gesture_result(
        "SLOW_RAISE_BOOM",
        ok,
        score,
        reason,
        {
            "signal_arm_side": arm_match["side"],
            "helper_side": opposite_side(arm_match["side"]),
            "palm_score": round(float(palm_score), 3),
            "index_up_score": round(float(index_score), 3),
            "proximity_score": round(float(proximity_score), 3),
        },
    )


def detect_lower_hoist(pose_landmarks, candidates):
    best_score = 0.0
    best_side = None
    best_index_score = 0.0
    for candidate in candidates:
        side = candidate["side"]
        if side not in ("left", "right") or not pose_landmarks:
            continue
        wrist = pose_wrist(pose_landmarks, side)
        shoulder = get_arm_landmarks(pose_landmarks, side)["shoulder"]
        index_score = index_down_score(candidate["landmarks"])
        below_shoulder_score = clamp((wrist.y - shoulder.y) / 0.28)
        score = index_score * 0.70 + below_shoulder_score * 0.30
        if score > best_score:
            best_score = score
            best_side = side
            best_index_score = index_score

    ok = bool(best_score >= INDEX_DOWN_SCORE_THRESHOLD)
    reason = "手部在肩部下方，食指向下，判定为吊钩下降。" if ok else "需要手部低于肩部，并让食指明显向下。"
    return gesture_result(
        "LOWER_HOIST",
        ok,
        best_score,
        reason,
        {
            "hand_side": best_side,
            "index_down_score": round(float(best_index_score), 3),
        },
    )


def detect_stop_everything(pose_landmarks):
    if not pose_landmarks:
        return gesture_result("STOP_EVERYTHING", False, 0.0, "未检测到人体姿态。")

    left_wrist = pose_wrist(pose_landmarks, "left")
    right_wrist = pose_wrist(pose_landmarks, "right")
    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    visibility = min(left_wrist.visibility, right_wrist.visibility, nose.visibility)
    wrists_above_head = max(left_wrist.y, right_wrist.y) < nose.y + 0.10
    wrists_above_shoulders = max(left_wrist.y, right_wrist.y) < min(left_shoulder.y, right_shoulder.y) - 0.12
    wrists_close_score = clamp(1.0 - distance(left_wrist, right_wrist) / 0.28)
    height_score = 1.0 if wrists_above_head else clamp((min(left_shoulder.y, right_shoulder.y) - max(left_wrist.y, right_wrist.y)) / 0.28)
    score = min(visibility / max(MIN_LANDMARK_VISIBILITY, 0.001), 1.0) * 0.20 + height_score * 0.45 + wrists_close_score * 0.35
    ok = bool(visibility >= MIN_LANDMARK_VISIBILITY and wrists_above_shoulders and wrists_close_score >= 0.35 and score >= TWO_HAND_OVERHEAD_SCORE_THRESHOLD)
    reason = "双手举过头顶并靠近，判定为停止所有操作。" if ok else "需要双手举过头顶并靠近。"
    return gesture_result(
        "STOP_EVERYTHING",
        ok,
        score,
        reason,
        {
            "wrists_above_head": bool(wrists_above_head),
            "wrists_close_score": round(float(wrists_close_score), 3),
        },
    )


def detect_raise_boom(arm_match, candidates):
    hand = select_best_hand(candidates, arm_match["side"])
    thumb_score = thumb_up_score(hand["landmarks"]) if hand else 0.0
    score = min(arm_match["score"], thumb_score)
    ok = bool(arm_match["ok"] and hand and thumb_score >= THUMB_UP_SCORE_THRESHOLD)
    return gesture_result(
        "RAISE_BOOM",
        ok,
        score,
        "水平伸出手臂，并且同侧手拇指向上。",
        {"signal_arm_side": arm_match["side"], "thumb_score": round(float(thumb_score), 3)},
    )


def detect_slow_raise_boom(arm_match, candidates):
    signal_hand = select_best_hand(candidates, arm_match["side"])
    helper_hand = select_best_hand(candidates, opposite_side(arm_match["side"]))
    palm_score = compute_palm_visible_score(signal_hand["landmarks"]) if signal_hand else 0.0
    index_score = index_up_score(helper_hand["landmarks"]) if helper_hand else 0.0
    proximity_score = 0.0

    if signal_hand and helper_hand:
        signal_wrist = signal_hand["landmarks"].landmark[mp_hands.HandLandmark.WRIST.value]
        helper_index = helper_hand["landmarks"].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
        proximity_score = clamp(1.0 - distance(signal_wrist, helper_index) / 0.35)

    score = min(arm_match["score"], max(palm_score, 0.45), index_score, max(proximity_score, 0.45))
    ok = bool(
        arm_match["ok"]
        and signal_hand
        and helper_hand
        and palm_score >= 0.45
        and index_score >= 0.55
        and proximity_score >= 0.25
    )
    return gesture_result(
        "SLOW_RAISE_BOOM",
        ok,
        score,
        "一只手臂水平伸出，另一只手食指向上并靠近水平手臂。",
        {
            "signal_arm_side": arm_match["side"],
            "helper_side": opposite_side(arm_match["side"]),
            "palm_score": round(float(palm_score), 3),
            "index_up_score": round(float(index_score), 3),
            "proximity_score": round(float(proximity_score), 3),
        },
    )


def detect_lower_hoist(pose_landmarks, candidates):
    best_score = 0.0
    best_side = None
    best_index_score = 0.0
    for candidate in candidates:
        side = candidate["side"]
        if side not in ("left", "right") or not pose_landmarks:
            continue
        wrist = pose_wrist(pose_landmarks, side)
        shoulder = get_arm_landmarks(pose_landmarks, side)["shoulder"]
        index_score = index_down_score(candidate["landmarks"])
        below_shoulder_score = clamp((wrist.y - shoulder.y) / 0.28)
        score = index_score * 0.70 + below_shoulder_score * 0.30
        if score > best_score:
            best_score = score
            best_side = side
            best_index_score = index_score

    ok = bool(best_score >= INDEX_DOWN_SCORE_THRESHOLD)
    return gesture_result(
        "LOWER_HOIST",
        ok,
        best_score,
        "手部在肩部下方，食指明显向下。",
        {"hand_side": best_side, "index_down_score": round(float(best_index_score), 3)},
    )


def detect_stop_everything(pose_landmarks):
    if not pose_landmarks:
        return gesture_result("STOP_EVERYTHING", False, 0.0, "未检测到人体姿态。")

    left_wrist = pose_wrist(pose_landmarks, "left")
    right_wrist = pose_wrist(pose_landmarks, "right")
    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    visibility = min(left_wrist.visibility, right_wrist.visibility, nose.visibility)
    wrists_above_head = max(left_wrist.y, right_wrist.y) < nose.y + 0.10
    wrists_above_shoulders = max(left_wrist.y, right_wrist.y) < min(left_shoulder.y, right_shoulder.y) - 0.12
    wrists_close_score = clamp(1.0 - distance(left_wrist, right_wrist) / 0.28)
    height_score = 1.0 if wrists_above_head else clamp((min(left_shoulder.y, right_shoulder.y) - max(left_wrist.y, right_wrist.y)) / 0.28)
    score = min(visibility / max(MIN_LANDMARK_VISIBILITY, 0.001), 1.0) * 0.20 + height_score * 0.45 + wrists_close_score * 0.35
    ok = bool(visibility >= MIN_LANDMARK_VISIBILITY and wrists_above_shoulders and wrists_close_score >= 0.35 and score >= TWO_HAND_OVERHEAD_SCORE_THRESHOLD)
    return gesture_result(
        "STOP_EVERYTHING",
        ok,
        score,
        "双手举过头顶并靠近。",
        {"wrists_above_head": bool(wrists_above_head), "wrists_close_score": round(float(wrists_close_score), 3)},
    )


def detect_slow_raise_boom(arm_match, candidates, stop_state=None):
    signal_hand = select_best_hand(candidates, arm_match["side"])
    helper_side = opposite_side(arm_match["side"])
    helper_hand = select_best_hand(candidates, helper_side)
    palm_score = compute_palm_visible_score(signal_hand["landmarks"]) if signal_hand else 0.0
    index_score = index_up_score(helper_hand["landmarks"]) if helper_hand else 0.0
    motion = stop_state.motion_features(helper_side) if stop_state and helper_side else {}
    motion_score = motion.get("motion_score", 0.0)
    proximity_score = 0.0

    if signal_hand and helper_hand:
        signal_wrist = signal_hand["landmarks"].landmark[mp_hands.HandLandmark.WRIST.value]
        helper_index = helper_hand["landmarks"].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
        proximity_score = clamp(1.0 - distance(signal_wrist, helper_index) / 0.55)

    score = (
        arm_match["score"] * 0.30
        + max(index_score, 0.25) * 0.20
        + proximity_score * 0.20
        + motion_score * 0.30
    )
    ok = bool(
        arm_match["ok"]
        and helper_hand
        and index_score >= 0.25
        and motion_score >= 0.30
        and proximity_score >= 0.08
        and score >= 0.42
    )
    return gesture_result(
        "SLOW_RAISE_BOOM",
        ok,
        score,
        "水平手臂保持，另一只手在水平臂附近做小范围慢速运动。",
        {
            "signal_arm_side": arm_match["side"],
            "helper_side": helper_side,
            "palm_score": round(float(palm_score), 3),
            "index_up_score": round(float(index_score), 3),
            "proximity_score": round(float(proximity_score), 3),
            **motion,
        },
    )


def detect_lower_hoist(pose_landmarks, candidates, stop_state=None):
    best_score = 0.0
    best_side = None
    best_index_score = 0.0
    best_motion = {}
    for candidate in candidates:
        side = candidate["side"]
        if side not in ("left", "right") or not pose_landmarks:
            continue
        wrist = pose_wrist(pose_landmarks, side)
        shoulder = get_arm_landmarks(pose_landmarks, side)["shoulder"]
        index_score = index_down_score(candidate["landmarks"])
        below_shoulder_score = clamp((wrist.y - shoulder.y) / 0.20)
        motion = stop_state.motion_features(side) if stop_state else {}
        motion_score = motion.get("motion_score", 0.0)
        score = index_score * 0.45 + below_shoulder_score * 0.35 + motion_score * 0.20
        if score > best_score:
            best_score = score
            best_side = side
            best_index_score = index_score
            best_motion = motion

    ok = bool(
        best_score >= 0.38
        and best_index_score >= 0.34
        and (
            best_motion.get("motion_score", 0.0) >= 0.22
            or best_index_score >= 0.52
        )
    )
    return gesture_result(
        "LOWER_HOIST",
        ok,
        best_score,
        "手部在肩部下方，食指或手部朝下，并出现下降/绕圈动作。",
        {
            "hand_side": best_side,
            "index_down_score": round(float(best_index_score), 3),
            **best_motion,
        },
    )


def evaluate_stop_gesture(frame, pose_results, whole_hands_results, hands_model):
    pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
    arm_match = compute_arm_pose_score(pose_landmarks) if pose_landmarks else default_arm_match()
    signal_arm_side = arm_match["side"] if arm_match["ok"] else None
    required_palm_side = opposite_side(signal_arm_side)

    candidates, roi_size = collect_hand_candidates(frame, hands_model, whole_hands_results, pose_landmarks)
    palm_match = select_palm_candidate(candidates, required_palm_side)
    same_hand_match = select_palm_candidate(candidates, signal_arm_side)

    palm_matches_opposite_wrist = bool(
        palm_match["ok"]
        and palm_match["wrist_match"]
        and palm_match["side"] == required_palm_side
    )
    different_hands_ok = bool(
        signal_arm_side
        and palm_match["side"] == required_palm_side
        and palm_match["side"] != signal_arm_side
    )
    same_hand_rejected = bool(
        same_hand_match["ok"]
        and same_hand_match["wrist_match"]
        and not palm_matches_opposite_wrist
    )
    raw_ok = bool(
        arm_match["ok"]
        and palm_match["ok"]
        and palm_matches_opposite_wrist
        and different_hands_ok
    )

    match_score = min(
        arm_match["score"],
        palm_match["score"] if palm_match["matched"] else 0.0,
    )

    return {
        "raw_ok": raw_ok,
        "code": "STOP" if raw_ok else None,
        "label": gesture_label("STOP") if raw_ok else gesture_label(None),
        "match_score": clamp(match_score),
        "arm_match": arm_match,
        "palm_match": palm_match,
        "same_hand_match": same_hand_match,
        "signal_arm_side": signal_arm_side,
        "required_palm_side": required_palm_side,
        "detected_palm_side": palm_match["side"] if palm_match["matched"] else None,
        "palm_matches_opposite_wrist": palm_matches_opposite_wrist,
        "different_hands_ok": different_hands_ok,
        "same_hand_rejected": same_hand_rejected,
        "roi_enabled": ROI_HANDS_ENABLED,
        "roi_size": roi_size,
        "hand_candidates": candidates,
        "pose_landmarks": pose_landmarks,
    }


def evaluate_all_gestures(frame, pose_results, whole_hands_results, hands_model, target_code=None, stop_state=None):
    detection = evaluate_stop_gesture(frame, pose_results, whole_hands_results, hands_model)
    pose_landmarks = detection["pose_landmarks"]
    candidates = detection["hand_candidates"]
    arm_match = detection["arm_match"]
    if stop_state is not None:
        stop_state.update_motion(candidates, time.monotonic())

    stop_result = gesture_result(
        "STOP",
        detection["raw_ok"],
        detection["match_score"],
        "双手配合停止手号通过。" if detection["raw_ok"] else "等待双手配合的停止手号。",
        {
            "signal_arm_side": detection["signal_arm_side"],
            "required_palm_side": detection["required_palm_side"],
            "detected_palm_side": detection["detected_palm_side"],
            "same_hand_rejected": detection["same_hand_rejected"],
        },
    )
    gesture_results = [
        stop_result,
        detect_raise_boom(arm_match, candidates),
        detect_slow_raise_boom(arm_match, candidates, stop_state),
        detect_lower_hoist(pose_landmarks, candidates, stop_state),
        detect_stop_everything(pose_landmarks),
    ]

    target_result = None
    if target_code:
        target_result = next(
            (result for result in gesture_results if result["code"] == target_code),
            None,
        )

    ok_results = [result for result in gesture_results if result["ok"]]
    selected = None
    if target_result is not None:
        selected = target_result if target_result["ok"] else None
    elif ok_results:
        selected = max(ok_results, key=lambda item: item["score"])
        stop_item = next(item for item in gesture_results if item["code"] == "STOP")
        if stop_item["ok"] and stop_item["score"] + GESTURE_SWITCH_MARGIN >= selected["score"]:
            selected = stop_item

    detection["gesture_results"] = gesture_results
    detection["hands_debug"] = build_hands_debug(candidates, stop_state)
    detection["target_code"] = target_code
    detection["target_result"] = target_result
    detection["test_mode"] = target_result is not None
    detection["selected_gesture"] = selected
    detection["raw_ok"] = selected is not None
    detection["code"] = selected["code"] if selected else None
    detection["label"] = selected["label"] if selected else (
        target_result["label"] if target_result else gesture_label(None)
    )
    detection["match_score"] = (
        selected["score"]
        if selected
        else (target_result["score"] if target_result else 0.0)
    )
    return detection


def describe_reason(detection, confirmed_code, lost_duration):
    selected = detection.get("selected_gesture")
    target_result = detection.get("target_result")
    if target_result and not selected and not confirmed_code:
        return f"单项测试：{target_result['label']}。当前分数 {target_result['score']:.2f}，尚未通过。{target_result['reason']}"
    if selected and confirmed_code == selected["code"]:
        return f"{selected['label']}动作已确认。{selected['reason']}"
    if selected:
        return f"检测到{selected['label']}，正在确认。{selected['reason']}"
    if confirmed_code and lost_duration > 0:
        return f"{gesture_label(confirmed_code)}已确认，当前短暂丢失中，仍在 1 秒容忍范围内。"
    if detection.get("same_hand_rejected"):
        return "检测到同一只手同时满足手臂和掌心条件，未通过双手规则。"

    arm = detection["arm_match"]
    palm = detection["palm_match"]
    signal_side = side_label(detection["signal_arm_side"])
    required_side = side_label(detection["required_palm_side"])

    if arm["extended_ok"] and not arm["horizontal_ok"]:
        return "手臂已伸出，但未接近水平。"
    if arm["ok"] and not palm["matched"]:
        return "手臂已伸出，但未检测到另一只手掌心。"
    if arm["ok"] and palm["matched"] and not palm["ok"]:
        return f"手臂已伸出，{required_side}手掌区域不够完整或掌心不明显。"
    if arm["ok"] and palm["ok"] and not detection["palm_matches_opposite_wrist"]:
        return "手掌可见，但没有匹配到另一只手的手腕。"
    if not arm["extended_ok"]:
        return "未检测到明显侧向伸出的水平手臂。"
    if not arm["straight_ok"]:
        return "手臂已伸出，但手肘未达到大致伸直。"
    return "正在等待图中任一吊运手号。"

    if detection["raw_ok"] and confirmed:
        return f"{signal_side}水平伸出，{required_side}掌心可见，双手配合通过，动作已确认。"
    if detection["raw_ok"]:
        return f"{signal_side}水平伸出，{required_side}掌心可见，双手配合通过，正在确认。"
    if confirmed and lost_duration > 0:
        return "动作已确认，当前短暂丢失中，仍在 1 秒容忍范围内。"
    if detection["same_hand_rejected"]:
        return "检测到同一只手同时满足手臂和掌心条件，未通过双手规则。"
    if arm["extended_ok"] and not arm["horizontal_ok"]:
        return "手臂已伸出，但未接近水平，未判定为 STOP。"
    if arm["ok"] and not palm["matched"]:
        return "手臂已伸出，但未检测到另一只手掌心。"
    if arm["ok"] and palm["matched"] and not palm["ok"]:
        return f"手臂已伸出，{required_side}手掌区域不够完整或掌心不明显。"
    if arm["ok"] and palm["ok"] and not detection["palm_matches_opposite_wrist"]:
        return "手掌可见，但没有匹配到另一只手的手腕，未判定为 STOP。"
    if not arm["extended_ok"]:
        return "未检测到明显侧向伸出的水平手臂。"
    if not arm["straight_ok"]:
        return "手臂已伸出，但手肘未达到大致伸直。"
    return "正在等待双手配合的 STOP 手号。"


def draw_hand_landmarks(frame, hand_landmarks, color=(0, 255, 255)):
    for connection in mp_hands.HAND_CONNECTIONS:
        start = hand_landmarks.landmark[connection[0]]
        end = hand_landmarks.landmark[connection[1]]
        cv2.line(frame, point_to_pixel(start, frame), point_to_pixel(end, frame), color, 2)
    for point in hand_landmarks.landmark:
        cv2.circle(frame, point_to_pixel(point, frame), 3, color, -1)


def draw_status(frame, detection, timing, confirmed_code):
    arm = detection["arm_match"]
    palm = detection["palm_match"]
    raw_ok = detection["raw_ok"]
    confirmed = confirmed_code is not None
    display_code = confirmed_code or detection.get("code")
    color = (0, 220, 0) if confirmed else (0, 0, 255)
    arm_color = (0, 220, 0) if arm["ok"] else (0, 0, 255)
    palm_color = (0, 220, 0) if palm["ok"] else (0, 0, 255)
    hands_color = (0, 220, 0) if detection["different_hands_ok"] else (0, 0, 255)

    cv2.rectangle(frame, (12, 12), (520, 238), (0, 0, 0), -1)
    lines = [
        (f"Signal Arm: {side_label(detection['signal_arm_side'])}", arm_color),
        (f"Palm Hand: {side_label(detection['detected_palm_side'])}", palm_color),
        (f"Different Hands: {'OK' if detection['different_hands_ok'] else 'NO'}", hands_color),
        (f"Palm Wrist Match: {'OK' if detection['palm_matches_opposite_wrist'] else 'NO'}", palm_color),
        (f"Gesture: {gesture_label(display_code)}", color),
        (f"Score: {detection['match_score']:.2f} | Raw: {'OK' if raw_ok else 'NO'}", (255, 255, 255)),
    ]
    y = 42
    for text, text_color in lines:
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
        y += 30

    bar_x, bar_y, bar_w, bar_h = 24, 198, 390, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * timing["hold_progress"]), bar_y + bar_h),
        color,
        -1,
    )
    cv2.rectangle(frame, (bar_x, bar_y + 18), (bar_x + bar_w, bar_y + 18 + bar_h), (80, 80, 80), 1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y + 18),
        (bar_x + int(bar_w * timing["lost_progress"]), bar_y + 18 + bar_h),
        (0, 180, 255),
        -1,
    )

    if arm["wrist"] is not None:
        cv2.putText(
            frame,
            "Signal Arm",
            point_to_pixel(arm["wrist"], frame),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
    if palm["landmarks"] is not None:
        draw_hand_landmarks(frame, palm["landmarks"], (0, 255, 255))
        palm_wrist = palm["landmarks"].landmark[mp_hands.HandLandmark.WRIST.value]
        cv2.putText(
            frame,
            "Palm Hand",
            point_to_pixel(palm_wrist, frame),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if detection["same_hand_rejected"]:
        cv2.putText(
            frame,
            "Same hand rejected",
            (24, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    if DRAW_ALL_HANDS_DEBUG:
        for candidate in detection.get("hand_candidates", []):
            color = (255, 120, 0) if candidate.get("source") == "ROI" else (0, 180, 255)
            draw_hand_landmarks(frame, candidate["landmarks"], color)
            wrist = candidate["landmarks"].landmark[mp_hands.HandLandmark.WRIST.value]
            label = f"{side_label(candidate.get('side'))} {candidate.get('source')} P:{candidate.get('score', 0.0):.2f}"
            cv2.putText(
                frame,
                label,
                point_to_pixel(wrist, frame),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    exposure = detection.get("exposure", {})
    if exposure.get("overexposed"):
        cv2.putText(
            frame,
            f"Overexposed: {exposure.get('overexposed_ratio', 0.0):.1%}",
            (24, 302),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def draw_message(frame, lines, color=(0, 255, 255)):
    cv2.rectangle(frame, (12, 12), (760, 132), (0, 0, 0), -1)
    y = 46
    for line in lines:
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        y += 34


def fourcc_to_string(value):
    try:
        value = int(value)
        return "".join(chr((value >> 8 * i) & 0xFF) for i in range(4)).strip()
    except Exception:
        return ""


def read_camera_runtime_info(cap):
    return {
        "auto_exposure": round(float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) or 0.0), 3),
        "exposure": round(float(cap.get(cv2.CAP_PROP_EXPOSURE) or 0.0), 3),
    }


def set_camera_exposure(cap, auto_exposure=None, exposure_value=None):
    result = {
        "auto_exposure_set": None,
        "exposure_set": None,
    }

    if auto_exposure is not None:
        auto_prop = AUTO_EXPOSURE_AUTO_VALUE if auto_exposure else AUTO_EXPOSURE_MANUAL_VALUE
        result["auto_exposure_set"] = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_prop))

    if exposure_value is not None and not auto_exposure:
        exposure_value = clamp(float(exposure_value), EXPOSURE_MIN_VALUE, EXPOSURE_MAX_VALUE)
        result["exposure_set"] = bool(cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value))

    result.update(read_camera_runtime_info(cap))
    return result


def read_camera_info(cap, backend_name, requested_width, requested_height, requested_fps, mjpg_requested):
    actual_fourcc = fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
    info = {
        "backend": backend_name,
        "requested_width": requested_width,
        "requested_height": requested_height,
        "requested_fps": requested_fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 1),
        "mjpg_requested": bool(mjpg_requested),
        "fourcc": actual_fourcc,
        "mjpg": actual_fourcc.upper() == "MJPG",
    }
    info.update(read_camera_runtime_info(cap))
    return info


def open_camera(camera_index):
    for backend_name, backend in CAMERA_BACKENDS:
        for width, height, fps in CAMERA_SETTINGS:
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            mjpg_requested = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            exposure_result = set_camera_exposure(
                cap,
                DEFAULT_AUTO_EXPOSURE,
                DEFAULT_EXPOSURE_VALUE,
            )

            warmup_started = time.monotonic()
            while time.monotonic() - warmup_started < 2.0:
                ok, _ = cap.read()
                if ok:
                    camera_info = read_camera_info(cap, backend_name, width, height, fps, mjpg_requested)
                    camera_info.update(
                        {
                            "default_auto_exposure": DEFAULT_AUTO_EXPOSURE,
                            "default_exposure": DEFAULT_EXPOSURE_VALUE,
                            **exposure_result,
                        }
                    )
                    print(
                        "Opened camera "
                        f"{camera_index} with {backend_name}: "
                        f"{camera_info['width']}x{camera_info['height']} "
                        f"{camera_info['fps']} FPS FOURCC={camera_info['fourcc']} "
                        f"EXPOSURE={camera_info['exposure']}"
                    )
                    return cap, camera_info
                time.sleep(0.05)

            print(
                f"Camera {camera_index} opened with {backend_name} "
                f"at {width}x{height}, but no frames were read."
            )
            cap.release()

    raise RuntimeError(
        f"Cannot open camera index {camera_index}. Try CAMERA_INDEX = 1 or 2 in main.py."
    )


def process_frame(frame, pose, hands, stop_state, now, target_code=None):
    exposure = analyze_exposure(frame)
    detection_frame = enhance_frame_for_detection(frame)
    rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    pose_results = pose.process(rgb_frame)
    whole_hands_results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    detection = evaluate_all_gestures(detection_frame, pose_results, whole_hands_results, hands, target_code, stop_state)
    detection["exposure"] = exposure
    detection["image_enhancement"] = ENABLE_IMAGE_ENHANCEMENT
    timing = stop_state.update(detection["code"], now)
    confirmed_code = timing["confirmed_code"]
    draw_status(frame, detection, timing, confirmed_code)
    return frame, detection, timing, confirmed_code


def main():
    cap, camera_info = open_camera(CAMERA_INDEX)
    stop_state = StopGestureState()
    failed_since = None

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                now = time.monotonic()
                if failed_since is None:
                    failed_since = now

                error_frame = np.zeros((480, 800, 3), dtype=np.uint8)
                draw_message(
                    error_frame,
                    [
                        f"Camera opened, but frame read failed. Backend: {camera_info['backend']}",
                        "Close other camera apps, or change CAMERA_INDEX to 1/2.",
                        "Press q to quit. Retrying...",
                    ],
                    color=(0, 255, 255),
                )
                cv2.imshow(WINDOW_NAME, error_frame)

                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break

                if now - failed_since >= FRAME_READ_RETRY_SECONDS:
                    print("Camera frame read kept failing. Exiting after retry timeout.")
                    break
                continue

            failed_since = None
            processed_frame, _, _, _ = process_frame(frame, pose, hands, stop_state, time.monotonic())

            cv2.imshow(WINDOW_NAME, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
