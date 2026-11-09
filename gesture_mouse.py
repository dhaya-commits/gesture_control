import time
import math
import threading
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- Settings you can tune ----------
WINDOW_SCALE = 1.6      # preview window size (1.0 = 640x480, 2.0 = double)
FRAME_MARGIN = 0    # larger = less hand travel = faster cursor (0.1 to 0.35)
MIN_ALPHA = 0.65        # smoothing for slow movements (higher = faster, less steady)
FAST_DIST = 55          # pixels of error at which smoothing turns off (lower = snappier)
DEADZONE = 2            # ignore cursor changes smaller than this many pixels
PINCH_ON = 0.25         # pinch closes when distance < this x palm size
PINCH_OFF = 0.35        # pinch releases when distance > this x palm size
PINCH_FREEZE = 0.45     # cursor freezes when thumb is this close (prevents click drift)
DRAG_HOLD = 0.25        # seconds to hold a pinch before the drag starts
LOST_HAND_TIMEOUT = 0.4 # seconds without a hand before a held drag is dropped
SCROLL_SPEED = 1500     # higher = faster scrolling
# -------------------------------------------

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # throw mouse into a screen corner to abort
SCREEN_W, SCREEN_H = pyautogui.size()

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


class Camera:
    """Reads frames in a background thread so we always process the newest one."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.count = 0
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                    self.count += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame.copy(), self.count

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(lm, tip, pip):
    return dist(lm[0], lm[tip]) > dist(lm[0], lm[pip])


cur_x, cur_y = pyautogui.position()


def move_cursor(nx, ny):
    """Map a camera position (0-1) to the screen and move with adaptive smoothing."""
    global cur_x, cur_y
    tx = np.interp(nx, (FRAME_MARGIN, 1 - FRAME_MARGIN), (0, SCREEN_W))
    ty = np.interp(ny, (FRAME_MARGIN, 1 - FRAME_MARGIN), (0, SCREEN_H))
    tx = min(max(tx, 1), SCREEN_W - 2)
    ty = min(max(ty, 1), SCREEN_H - 2)
    dx, dy = tx - cur_x, ty - cur_y
    d = math.hypot(dx, dy)
    if d > DEADZONE:
        alpha = min(1.0, MIN_ALPHA + d / FAST_DIST)
        cur_x += dx * alpha
        cur_y += dy * alpha
        pyautogui.moveTo(cur_x, cur_y)


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cam = Camera()
start = time.time()
last_ts = -1
last_count = 0

left_down = False       # True while the mouse button is held (click or drag)
right_down = False
pinch_start = 0.0
last_seen = time.time()
prev_scroll_y = None
paused = False
fps = 0.0
t_prev = time.time()

# Resizable preview window, larger and placed in the bottom-right corner
WIN = "Gesture Mouse"
win_w, win_h = int(640 * WINDOW_SCALE), int(480 * WINDOW_SCALE)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, win_w, win_h)
cv2.moveWindow(WIN, max(SCREEN_W - win_w - 10, 0), max(SCREEN_H - win_h - 70, 0))


def release_button():
    global left_down
    if left_down:
        pyautogui.mouseUp()
        left_down = False


try:
    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            frame, count = cam.read()
            if frame is None or count == last_count:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_count = count

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.time() - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = detector.detect_for_video(mp_image, ts)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            mode = "PAUSED" if paused else "No hand"

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                last_seen = now

                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for x, y in pts:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                if not paused:
                    palm = max(dist(lm[0], lm[9]), 1e-6)
                    idx_up = finger_extended(lm, 8, 6)
                    mid_up = finger_extended(lm, 12, 10)
                    ring_up = finger_extended(lm, 16, 14)
                    pinky_up = finger_extended(lm, 20, 18)

                    d_left = dist(lm[4], lm[8]) / palm
                    d_right = dist(lm[4], lm[12]) / palm

                    # Left button: down on pinch, up on release (click or drag)
                    if not left_down and not right_down and d_left < PINCH_ON:
                        pyautogui.mouseDown()
                        left_down = True
                        pinch_start = now
                    elif left_down and d_left > PINCH_OFF:
                        pyautogui.mouseUp()
                        left_down = False

                    # Right click: thumb + middle finger
                    if not left_down:
                        if not right_down and d_right < PINCH_ON:
                            pyautogui.rightClick()
                            right_down = True
                        elif right_down and d_right > PINCH_OFF:
                            right_down = False

                    if left_down:
                        if now - pinch_start >= DRAG_HOLD:
                            mode = "DRAG"
                            # Follow the midpoint between thumb tip and index tip
                            mx = (lm[4].x + lm[8].x) / 2
                            my = (lm[4].y + lm[8].y) / 2
                            move_cursor(mx, my)
                        else:
                            mode = "CLICK"   # cursor frozen so the click lands accurately
                        prev_scroll_y = None
                    elif right_down:
                        mode = "RIGHT CLICK"
                        prev_scroll_y = None
                    elif idx_up and mid_up and not ring_up and not pinky_up:
                        mode = "SCROLL"
                        y = lm[8].y
                        if prev_scroll_y is not None:
                            delta = prev_scroll_y - y
                            if abs(delta) > 0.004:
                                pyautogui.scroll(int(delta * SCROLL_SPEED))
                        prev_scroll_y = y
                    elif idx_up and not mid_up:
                        prev_scroll_y = None
                        if d_left < PINCH_FREEZE:
                            mode = "READY"   # thumb close: hold cursor still
                        else:
                            mode = "MOVE"
                            move_cursor(lm[8].x, lm[8].y)
                    else:
                        mode = "IDLE"
                        prev_scroll_y = None

                    # Visual feedback: line between thumb and index
                    if left_down:
                        cv2.line(frame, pts[4], pts[8], (0, 165, 255), 4)
                else:
                    release_button()
            else:
                # Hand lost: never leave the mouse button stuck down
                if left_down and now - last_seen > LOST_HAND_TIMEOUT:
                    release_button()

            m = FRAME_MARGIN
            cv2.rectangle(frame, (int(m * w), int(m * h)),
                          (int((1 - m) * w), int((1 - m) * h)), (255, 0, 0), 2)
            cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "q: quit   p: pause/resume", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                if paused:
                    release_button()
finally:
    release_button()
    cam.release()
    cv2.destroyAllWindows()