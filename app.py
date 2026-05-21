import atexit
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from main import (
    CAMERA_INDEX,
    DEFAULT_AUTO_EXPOSURE,
    DEFAULT_EXPOSURE_VALUE,
    ENABLE_IMAGE_ENHANCEMENT,
    EXPOSURE_MAX_VALUE,
    EXPOSURE_MIN_VALUE,
    EXPOSURE_STEP,
    FRAME_READ_RETRY_SECONDS,
    GESTURE_TEST_OPTIONS,
    HOLD_SECONDS,
    JPEG_QUALITY,
    LOST_SECONDS_BEFORE_NO,
    ROI_BASE_SIZE,
    ROI_HANDS_ENABLED,
    StopGestureState,
    WINDOW_NAME,
    describe_reason,
    draw_message,
    gesture_label,
    mp_hands,
    mp_pose,
    open_camera,
    process_frame,
    read_camera_runtime_info,
    set_camera_exposure,
    side_label,
)


app = Flask(__name__)


VALID_GESTURE_CODES = {code for code, _ in GESTURE_TEST_OPTIONS}


def normalize_target(value):
    if not value:
        return None
    value = value.strip().upper()
    return value if value in VALID_GESTURE_CODES else None


class GestureCamera:
    def __init__(self):
        self.cap = None
        self.pose = None
        self.hands = None
        self.camera_info = self.default_camera_info()
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.last_frame = None
        self.gesture_state = StopGestureState()
        self.target_code = None
        self.failed_since = None
        self.exposure_auto = DEFAULT_AUTO_EXPOSURE
        self.exposure_value = DEFAULT_EXPOSURE_VALUE
        self.status = self.default_status("摄像头未启动")

    def default_camera_info(self):
        return {
            "backend": "",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "fourcc": "",
            "mjpg": False,
            "mjpg_requested": False,
            "auto_exposure": 0.0,
            "exposure": 0.0,
        }

    def default_status(self, message="准备就绪"):
        return {
            "gesture": "未识别",
            "confirmed_gesture": "未确认",
            "gesture_code": "NOT_DETECTED",
            "candidate_gesture": "未识别",
            "candidate_code": None,
            "result": False,
            "current_reason": message,
            "reason": message,
            "message": message,
            "signal_arm_side": None,
            "required_palm_side": None,
            "detected_palm_side": None,
            "signal_arm_label": "未确定",
            "required_palm_label": "未确定",
            "detected_palm_label": "未确定",
            "different_hands_ok": False,
            "same_hand_rejected": False,
            "arm_pose": False,
            "arm_extended": False,
            "arm_horizontal": False,
            "arm_straight": False,
            "palm_visible": False,
            "palm_matches_opposite_wrist": False,
            "wrist_match": False,
            "match_score": 0.0,
            "arm_score": 0.0,
            "palm_score": 0.0,
            "arm_angle": None,
            "wrist_shoulder_y_diff": None,
            "wrist_match_distance": None,
            "raw_ok": False,
            "confirmed_stop": False,
            "confirmed_any": False,
            "good_duration": 0.0,
            "lost_duration": 0.0,
            "hold_progress": 0.0,
            "lost_progress": 0.0,
            "hold_seconds": HOLD_SECONDS,
            "lost_seconds_before_no": LOST_SECONDS_BEFORE_NO,
            "camera_backend": self.camera_info["backend"],
            "camera_width": self.camera_info["width"],
            "camera_height": self.camera_info["height"],
            "camera_fps": self.camera_info["fps"],
            "camera_fourcc": self.camera_info["fourcc"],
            "camera_mjpg": self.camera_info["mjpg"],
            "camera_mjpg_requested": self.camera_info["mjpg_requested"],
            "jpeg_quality": JPEG_QUALITY,
            "roi_enabled": ROI_HANDS_ENABLED,
            "roi_size": ROI_BASE_SIZE,
            "image_enhancement": ENABLE_IMAGE_ENHANCEMENT,
            "overexposed": False,
            "overexposed_ratio": 0.0,
            "mean_luma": 0.0,
            "process_ms": 0.0,
            "exposure_auto_requested": self.exposure_auto,
            "exposure_requested": self.exposure_value,
            "camera_auto_exposure": self.camera_info.get("auto_exposure", 0.0),
            "camera_exposure": self.camera_info.get("exposure", 0.0),
            "exposure_min": EXPOSURE_MIN_VALUE,
            "exposure_max": EXPOSURE_MAX_VALUE,
            "exposure_step": EXPOSURE_STEP,
            "hands_debug": [],
            "gesture_scores": [],
            "test_mode": self.target_code is not None,
            "target_code": self.target_code,
            "target_gesture": gesture_label(self.target_code) if self.target_code else "全手势识别",
            "test_options": [
                {"code": code, "label": label}
                for code, label in GESTURE_TEST_OPTIONS
            ],
        }

    def start(self):
        with self.lock:
            if self.running:
                return

            try:
                self.cap, self.camera_info = open_camera(CAMERA_INDEX)
                self._apply_camera_controls_locked()
            except Exception as exc:
                self._set_error_frame(f"Camera error: {exc}")
                return

            self.pose = mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self.running = True
            self.status = self.default_status("摄像头运行中")

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _apply_camera_controls_locked(self):
        if not self.cap:
            return {}

        result = set_camera_exposure(
            self.cap,
            self.exposure_auto,
            self.exposure_value,
        )
        self.camera_info.update(result)
        return result

    def update_camera_controls(self, auto_exposure=None, exposure_value=None):
        with self.lock:
            if auto_exposure is not None:
                self.exposure_auto = bool(auto_exposure)
            if exposure_value is not None:
                self.exposure_value = max(
                    EXPOSURE_MIN_VALUE,
                    min(EXPOSURE_MAX_VALUE, float(exposure_value)),
                )

            result = self._apply_camera_controls_locked()
            self.status.update(
                {
                    "exposure_auto_requested": self.exposure_auto,
                    "exposure_requested": self.exposure_value,
                    "camera_auto_exposure": self.camera_info.get("auto_exposure", 0.0),
                    "camera_exposure": self.camera_info.get("exposure", 0.0),
                }
            )

            return {
                "ok": True,
                "exposure_auto_requested": self.exposure_auto,
                "exposure_requested": self.exposure_value,
                "camera_auto_exposure": self.camera_info.get("auto_exposure", 0.0),
                "camera_exposure": self.camera_info.get("exposure", 0.0),
                "camera_result": result,
            }

    def set_target(self, target_code):
        with self.lock:
            if target_code != self.target_code:
                self.target_code = target_code
                self.gesture_state = StopGestureState()
                self.status = self.default_status(
                    f"单项测试：{gesture_label(target_code)}" if target_code else "全手势识别"
                )

    def _set_error_frame(self, message):
        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        draw_message(
            frame,
            [
                message,
                "Check camera permission, close other camera apps,",
                "or change CAMERA_INDEX to 1/2 in main.py.",
            ],
            color=(0, 255, 255),
        )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        self.status = self.default_status(message)
        if ok:
            self.last_frame = encoded.tobytes()

    def stop(self):
        with self.lock:
            self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.pose:
                self.pose.close()
                self.pose = None
            if self.hands:
                self.hands.close()
                self.hands = None
            self.last_frame = None
            self.gesture_state = StopGestureState()
            self.status = self.default_status("摄像头已释放")

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def get_frame(self):
        with self.lock:
            return self.last_frame

    def _capture_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                cap = self.cap
                pose = self.pose
                hands = self.hands
                target_code = self.target_code

            ok, frame = cap.read()
            now = time.monotonic()

            if not ok:
                self._handle_frame_read_failure(now)
                time.sleep(0.1)
                continue

            self.failed_since = None
            process_started = time.monotonic()
            processed_frame, detection, timing, confirmed_code = process_frame(
                frame,
                pose,
                hands,
                self.gesture_state,
                now,
                target_code,
            )
            process_ms = (time.monotonic() - process_started) * 1000.0
            camera_runtime = read_camera_runtime_info(cap)
            status = self._build_status(detection, timing, confirmed_code, process_ms, camera_runtime)

            ok, encoded = cv2.imencode(
                ".jpg",
                processed_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                continue

            with self.lock:
                self.camera_info.update(camera_runtime)
                self.last_frame = encoded.tobytes()
                self.status = status

    def _handle_frame_read_failure(self, now):
        if self.failed_since is None:
            self.failed_since = now

        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        draw_message(
            frame,
            [
                f"Camera opened, but frame read failed. Backend: {self.camera_info['backend']}",
                "Close other camera apps, or change CAMERA_INDEX to 1/2.",
                "Web server is still running. Retrying...",
            ],
            color=(0, 255, 255),
        )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        status = self.default_status("摄像头读帧失败，正在重试")

        if now - self.failed_since >= FRAME_READ_RETRY_SECONDS:
            status["message"] = "摄像头持续读帧失败"
            status["reason"] = "摄像头持续读帧失败"
            status["current_reason"] = "摄像头持续读帧失败"

        with self.lock:
            if ok:
                self.last_frame = encoded.tobytes()
            self.status = status

    def _build_status(self, detection, timing, confirmed_code, process_ms=0.0, camera_runtime=None):
        arm = detection["arm_match"]
        palm = detection["palm_match"]
        candidate = detection.get("selected_gesture")
        gesture_code = confirmed_code or (candidate["code"] if candidate else None)
        reason = describe_reason(detection, confirmed_code, timing["lost_duration"])
        wrist_distance = palm["wrist_distance"]
        gesture_scores = detection.get("gesture_results", [])
        exposure = detection.get("exposure", {})
        camera_runtime = camera_runtime or {}

        return {
            "gesture": gesture_label(gesture_code),
            "confirmed_gesture": gesture_label(confirmed_code) if confirmed_code else "未确认",
            "gesture_code": gesture_code or "NOT_DETECTED",
            "candidate_gesture": candidate["label"] if candidate else "未识别",
            "candidate_code": candidate["code"] if candidate else None,
            "result": bool(confirmed_code),
            "current_reason": reason,
            "reason": reason,
            "message": "摄像头运行中",
            "signal_arm_side": detection["signal_arm_side"],
            "required_palm_side": detection["required_palm_side"],
            "detected_palm_side": detection["detected_palm_side"],
            "signal_arm_label": side_label(detection["signal_arm_side"]),
            "required_palm_label": side_label(detection["required_palm_side"]),
            "detected_palm_label": side_label(detection["detected_palm_side"]),
            "different_hands_ok": bool(detection["different_hands_ok"]),
            "same_hand_rejected": bool(detection["same_hand_rejected"]),
            "arm_pose": bool(arm["ok"]),
            "arm_extended": bool(arm["extended_ok"]),
            "arm_horizontal": bool(arm["horizontal_ok"]),
            "arm_straight": bool(arm["straight_ok"]),
            "palm_visible": bool(palm["ok"]),
            "palm_matches_opposite_wrist": bool(detection["palm_matches_opposite_wrist"]),
            "wrist_match": bool(detection["palm_matches_opposite_wrist"]),
            "match_score": round(float(detection["match_score"]), 3),
            "arm_score": round(float(arm["score"]), 3),
            "palm_score": round(float(palm["score"]), 3),
            "arm_angle": arm["arm_angle"],
            "wrist_shoulder_y_diff": arm["wrist_shoulder_y_diff"],
            "wrist_match_distance": (
                None if wrist_distance is None else round(float(wrist_distance), 3)
            ),
            "raw_ok": bool(detection["raw_ok"]),
            "confirmed_stop": confirmed_code == "STOP",
            "confirmed_any": bool(confirmed_code),
            "good_duration": round(float(timing["good_duration"]), 2),
            "lost_duration": round(float(timing["lost_duration"]), 2),
            "hold_progress": round(float(timing["hold_progress"]), 3),
            "lost_progress": round(float(timing["lost_progress"]), 3),
            "hold_seconds": HOLD_SECONDS,
            "lost_seconds_before_no": LOST_SECONDS_BEFORE_NO,
            "camera_backend": self.camera_info["backend"],
            "camera_width": self.camera_info["width"],
            "camera_height": self.camera_info["height"],
            "camera_fps": self.camera_info["fps"],
            "camera_fourcc": self.camera_info["fourcc"],
            "camera_mjpg": self.camera_info["mjpg"],
            "camera_mjpg_requested": self.camera_info["mjpg_requested"],
            "camera_auto_exposure": camera_runtime.get(
                "auto_exposure",
                self.camera_info.get("auto_exposure", 0.0),
            ),
            "camera_exposure": camera_runtime.get(
                "exposure",
                self.camera_info.get("exposure", 0.0),
            ),
            "exposure_auto_requested": self.exposure_auto,
            "exposure_requested": self.exposure_value,
            "exposure_min": EXPOSURE_MIN_VALUE,
            "exposure_max": EXPOSURE_MAX_VALUE,
            "exposure_step": EXPOSURE_STEP,
            "jpeg_quality": JPEG_QUALITY,
            "roi_enabled": bool(detection["roi_enabled"]),
            "roi_size": detection["roi_size"],
            "image_enhancement": bool(detection.get("image_enhancement", ENABLE_IMAGE_ENHANCEMENT)),
            "overexposed": bool(exposure.get("overexposed", False)),
            "overexposed_ratio": float(exposure.get("overexposed_ratio", 0.0)),
            "mean_luma": float(exposure.get("mean_luma", 0.0)),
            "process_ms": round(float(process_ms), 1),
            "hands_debug": detection.get("hands_debug", []),
            "gesture_scores": gesture_scores,
            "test_mode": bool(detection.get("test_mode")),
            "target_code": detection.get("target_code"),
            "target_gesture": gesture_label(detection.get("target_code")) if detection.get("target_code") else "全手势识别",
            "target_score": (
                detection["target_result"]["score"]
                if detection.get("target_result")
                else None
            ),
            "target_ok": (
                bool(detection["target_result"]["ok"])
                if detection.get("target_result")
                else False
            ),
            "test_options": [
                {"code": code, "label": label}
                for code, label in GESTURE_TEST_OPTIONS
            ],
        }


gesture_camera = GestureCamera()
atexit.register(gesture_camera.stop)


@app.route("/")
def index():
    target_code = normalize_target(request.args.get("target"))
    return render_template(
        "index.html",
        title=WINDOW_NAME,
        target_code=target_code or "",
        test_options=[
            {"code": code, "label": label}
            for code, label in GESTURE_TEST_OPTIONS
        ],
    )


@app.route("/video_feed")
def video_feed():
    gesture_camera.set_target(normalize_target(request.args.get("target")))
    gesture_camera.start()

    def generate():
        while True:
            frame = gesture_camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    gesture_camera.set_target(normalize_target(request.args.get("target")))
    gesture_camera.start()
    return jsonify(gesture_camera.get_status())


@app.route("/camera_control", methods=["POST"])
def camera_control():
    data = request.get_json(silent=True) or {}
    auto_value = data.get("auto_exposure")
    exposure_value = data.get("exposure")

    auto_exposure = None
    if auto_value is not None:
        if isinstance(auto_value, bool):
            auto_exposure = auto_value
        else:
            auto_exposure = str(auto_value).strip().lower() in {"1", "true", "yes", "on", "auto"}

    exposure = None
    if exposure_value is not None and str(exposure_value).strip() != "":
        try:
            exposure = float(exposure_value)
        except ValueError:
            return jsonify({"ok": False, "error": "曝光值不是有效数字"}), 400

    result = gesture_camera.update_camera_controls(auto_exposure, exposure)
    return jsonify(result)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        gesture_camera.stop()
