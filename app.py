"""
👁️  DrowsGuard — Flask Backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTALL:
    pip install flask opencv-python mediapipe numpy pygame

Run:
    python app.py

Then open: http://localhost:5000
"""

import sys, os, time, threading, urllib.request
from collections import deque

# ── dependency check ──────────────────────────────────────────────────────────
def _require(pkg, install=None):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"\n  Missing: {pkg}   →   pip install {install or pkg}\n")
        sys.exit(1)

flask_mod  = _require("flask",     "flask")
cv2        = _require("cv2",       "opencv-python")
np         = _require("numpy",     "numpy")
pygame     = _require("pygame",    "pygame")
_require("mediapipe", "mediapipe")

import mediapipe as mp
from mediapipe.tasks.python.core        import base_options as mp_base
from mediapipe.tasks.python.vision      import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_mode
from flask import Flask, Response, jsonify, send_from_directory

# ─── Config ───────────────────────────────────────────────────────────────────

BLINK_THRESHOLD    = 0.35
SMOOTH_WINDOW      = 4
VOTE_WINDOW        = 6
VOTE_MIN           = 4
ALARM_TRIGGER_SECS = 1.0
CAMERA_INDEX       = 0
ALARM_FILE         = "alarm.mp3"

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

BS_BLINK_L = "eyeBlinkLeft"
BS_BLINK_R = "eyeBlinkRight"

# ─── Model download ───────────────────────────────────────────────────────────

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("  Downloading face_landmarker.task (~1 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved → {MODEL_PATH}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Download manually from:\n  {MODEL_URL}")
        sys.exit(1)

# ─── Audio ────────────────────────────────────────────────────────────────────

def init_audio():
    pygame.mixer.init()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ALARM_FILE)
    if not os.path.exists(path):
        print(f"\n  alarm.mp3 not found at {path}")
        print(f"  Place '{ALARM_FILE}' next to app.py\n")
        sys.exit(1)
    return pygame.mixer.Sound(path)

# ─── Detection engine ─────────────────────────────────────────────────────────

class DetectionEngine:
    def __init__(self):
        ensure_model()
        self._beep    = init_audio()
        self._playing = False

        options = FaceLandmarkerOptions(
            base_options            = mp_base.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode            = mp_mode.VisionTaskRunningMode.IMAGE,
            num_faces               = 1,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
            output_face_blendshapes       = True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

        self._l_buf      = deque(maxlen=SMOOTH_WINDOW)
        self._r_buf      = deque(maxlen=SMOOTH_WINDOW)
        self._vote_buf   = deque(maxlen=VOTE_WINDOW)
        self._closed_since = None

        self._fps_buf  = deque(maxlen=30)
        self._prev_t   = time.time()

        # Shared state read by Flask routes
        self.state = {
            "status":       "open",   # "open" | "closed" | "alert" | "no_face"
            "l_score":      0.0,
            "r_score":      0.0,
            "sm_l":         0.0,
            "sm_r":         0.0,
            "votes_closed": 0,
            "vote_total":   0,
            "closed_for":   0.0,
            "fps":          0,
            "face_found":   False,
            "alarming":     False,
        }
        self._lock = threading.Lock()

        # MJPEG frame (JPEG bytes)
        self._frame_bytes = None
        self._frame_lock  = threading.Lock()

        self._running = False
        self._cap     = None
        self._thread  = None

    # ── alarm ─────────────────────────────────────────────────────────────
    def _start_alarm(self):
        if not self._playing:
            self._playing = True
            self._beep.play(loops=0)

    def _stop_alarm(self):
        if self._playing:
            self._playing = False
            self._beep.stop()

    # ── process one frame ─────────────────────────────────────────────────
    def _process(self, frame):
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._prev_t, 1e-6))
        self._prev_t = now
        fps = float(np.mean(self._fps_buf))

        h, w = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        face_found = (result.face_blendshapes is not None
                      and len(result.face_blendshapes) > 0)

        l_score = r_score = 0.0
        if face_found:
            bs      = {c.category_name: c.score for c in result.face_blendshapes[0]}
            l_score = bs.get(BS_BLINK_L, 0.0)
            r_score = bs.get(BS_BLINK_R, 0.0)

        self._l_buf.append(l_score)
        self._r_buf.append(r_score)
        sm_l = float(np.median(self._l_buf))
        sm_r = float(np.median(self._r_buf))

        if len(self._l_buf) == SMOOTH_WINDOW:
            self._vote_buf.append(sm_l >= BLINK_THRESHOLD and sm_r >= BLINK_THRESHOLD)

        votes_closed = int(sum(self._vote_buf))
        eyes_closed  = (len(self._vote_buf) == VOTE_WINDOW
                        and votes_closed >= VOTE_MIN)

        # auto-reset playing flag
        if self._playing and self._beep.get_num_channels() == 0:
            self._playing = False

        closed_for = 0.0
        if face_found and eyes_closed:
            if self._closed_since is None:
                self._closed_since = time.time()
            closed_for = time.time() - self._closed_since
            if closed_for >= ALARM_TRIGGER_SECS:
                self._start_alarm()
                status = "alert"
            else:
                status = "closed"
        else:
            self._closed_since = None
            if face_found:
                self._stop_alarm()
            status = "open" if face_found else "no_face"

        # Draw face mesh dots on frame
        if result.face_landmarks:
            for lm in result.face_landmarks[0]:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 1, (60, 180, 60), -1)

        # Draw eye outlines
        if result.face_landmarks:
            EYE_L = [362, 385, 387, 263, 373, 380]
            EYE_R = [33,  160, 158, 133, 153, 144]
            lms   = result.face_landmarks[0]
            for idxs, score in [(EYE_L, sm_l), (EYE_R, sm_r)]:
                pts = np.array(
                    [(int(lms[i].x * w), int(lms[i].y * h)) for i in idxs],
                    dtype=np.int32)
                col = (30, 30, 220) if score >= BLINK_THRESHOLD else (50, 210, 50)
                cv2.polylines(frame, [pts], isClosed=True, color=col,
                              thickness=2, lineType=cv2.LINE_AA)

        # Update shared state
        with self._lock:
            self.state.update({
                "status":       status,
                "l_score":      round(l_score, 3),
                "r_score":      round(r_score, 3),
                "sm_l":         round(sm_l, 3),
                "sm_r":         round(sm_r, 3),
                "votes_closed": votes_closed,
                "vote_total":   len(self._vote_buf),
                "closed_for":   round(closed_for, 2),
                "fps":          round(fps, 1),
                "face_found":   face_found,
                "alarming":     self._playing,
            })

        # JPEG encode for MJPEG stream
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._frame_lock:
            self._frame_bytes = jpeg.tobytes()

    # ── camera thread ─────────────────────────────────────────────────────
    def _camera_loop(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            print(f"  Cannot open camera {CAMERA_INDEX}")
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        print("  Camera started")

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)
            self._process(frame)

        self._cap.release()
        self._stop_alarm()
        print("  Camera stopped")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._frame_lock:
            return self._frame_bytes

    def get_state(self):
        with self._lock:
            return dict(self.state)


# ─── Flask app ────────────────────────────────────────────────────────────────

engine = DetectionEngine()
app    = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/alarm.mp3")
def alarm_audio():
    return send_from_directory(".", "alarm.mp3")

@app.route("/start", methods=["POST"])
def start():
    engine.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    engine.stop()
    return jsonify({"ok": True})

@app.route("/state")
def state():
    return jsonify(engine.get_state())

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = engine.get_frame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)   # ~30 fps
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)