import sys, os, subprocess, importlib, time, math
from pathlib import Path

# ─── AUTO INSTALLER ──────────────────────────────────────────────────────────
REQUIRED = [
    ("ultralytics", "ultralytics"),
    ("cv2",         "opencv-python"),
    ("numpy",       "numpy"),
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
    ("PIL",         "pillow"),
    ("tqdm",        "tqdm"),
]

def print_banner():
    print("\033[96m")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       AI DUMBBELL FORM CHECKER & REP COUNTER  v2            ║")
    print("║       Bicep Curl · Hammer Curl · Shoulder Press             ║")
    print("║       Rep Counter · Form Score · Injury Risk HUD            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\033[0m")

def install_step(step, total, name):
    bar_len = 30
    fill = int(bar_len * step / total)
    bar  = "█" * fill + "░" * (bar_len - fill)
    pct  = int(100 * step / total)
    print(f"\r\033[93m[{step}/{total}] {name:<35} [{bar}] {pct}%\033[0m", end="", flush=True)

def auto_install():
    print_banner()
    print(f"\033[92m[1/6] Checking Dependencies...\033[0m")
    missing = []
    for mod, pkg in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\033[92m[2/6] Installing Packages...\033[0m")
        for i, pkg in enumerate(missing):
            install_step(i+1, len(missing), f"Installing {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print()
    else:
        print(f"\033[92m[2/6] All packages present.\033[0m")
    print(f"\033[92m[3/6] Loading Pose Model...\033[0m")
    print(f"\033[92m[4/6] Initializing AI Engine...\033[0m")

auto_install()

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

# ─── GPU ─────────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
print(f"\033[94m[GPU] Device: {DEVICE.upper()}  FP16: {USE_FP16}\033[0m")

# ─── MODEL ───────────────────────────────────────────────────────────────────
print("\033[93m[MODEL] Loading YOLOv8 pose model...\033[0m")
model = YOLO("yolov8n-pose.pt")
print("\033[92m[MODEL] Ready.\033[0m")

# ─── KEYPOINT INDICES (COCO) ─────────────────────────────────────────────────
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHOULDER=5; R_SHOULDER=6
L_ELBOW=7;    R_ELBOW=8
L_WRIST=9;    R_WRIST=10
L_HIP=11;     R_HIP=12
L_KNEE=13;    R_KNEE=14
L_ANKLE=15;   R_ANKLE=16

SKELETON_PAIRS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),   (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),       (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),       (R_KNEE, R_ANKLE),
]

# ─── COLORS ──────────────────────────────────────────────────────────────────
GLOW = {
    "skeleton_good": (0, 255, 180),
    "skeleton_warn": (0, 200, 255),
    "skeleton_bad":  (60, 60, 255),
    "joint_good":    (0, 255, 120),
    "joint_warn":    (0, 180, 255),
    "joint_bad":     (60, 60, 255),
    "angle_good":    (0, 255, 100),
    "angle_warn":    (0, 180, 255),
    "angle_bad":     (60, 60, 255),
    "hud_accent":    (0, 200, 255),
    "rep_counter":   (0, 255, 120),
    "warning_text":  (40, 40, 255),
    "bar_good":      (0, 220, 100),
    "bar_bad":       (60, 60, 255),
    "left_arm":      (255, 200, 0),
    "right_arm":     (0, 200, 255),
    "bicep_glow_l":  (255, 160, 0),
    "bicep_glow_r":  (0, 160, 255),
}

# ─── EXERCISE CONFIGS ─────────────────────────────────────────────────────────
#
#  REP COUNTING LOGIC (curl style):
#  ─────────────────────────────────
#  Each arm is tracked independently with a 2-phase state machine:
#
#    Phase EXTENDED  →  arm is straight  (elbow angle > EXTEND_THR, e.g. 150°)
#    Phase CURLED    →  arm is bent      (elbow angle < CURL_THR,   e.g.  55°)
#
#  A COMPLETE REP = arm goes EXTENDED → CURLED → EXTENDED
#  (i.e. one full curl up AND back down)
#
#  Rep is counted on the RETURN to EXTENDED (bottom of the movement),
#  so the counter shows completed reps, not half-reps.
#
#  "Total reps" = max(left_reps, right_reps)  so alternating curls count once each.
#
EXERCISES = {
    "BICEP CURL": {
        "joints": [
            {"name": "L Elbow",    "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "good": (30, 170), "warn": (20, 175)},
            {"name": "R Elbow",    "a": R_SHOULDER, "b": R_ELBOW, "c": R_WRIST,
             "good": (30, 170), "warn": (20, 175)},
            {"name": "L UpperArm", "a": L_HIP,      "b": L_SHOULDER, "c": L_ELBOW,
             "good": (55, 125), "warn": (40, 140)},
            {"name": "R UpperArm", "a": R_HIP,      "b": R_SHOULDER, "c": R_ELBOW,
             "good": (55, 125), "warn": (40, 140)},
        ],
        "rep_angle_left":  (L_SHOULDER, L_ELBOW, L_WRIST),
        "rep_angle_right": (R_SHOULDER, R_ELBOW, R_WRIST),
        # arm straight  = elbow angle > EXTEND_THR  → phase EXTENDED
        # arm curled    = elbow angle < CURL_THR    → phase CURLED
        # rep counted when returning from CURLED → EXTENDED
        "EXTEND_THR": 140,  # was 150
        "CURL_THR":    65,  # was 55
        "description": "Keep elbows pinned · Full extension at bottom · Squeeze at top · No swinging",
        "tip_curl":   "SQUEEZE AT TOP!",
        "tip_extend": "FULL EXTENSION!",
        "is_press": False,
    },
    "HAMMER CURL": {
        "joints": [
            {"name": "L Elbow",    "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "good": (30, 170), "warn": (20, 175)},
            {"name": "R Elbow",    "a": R_SHOULDER, "b": R_ELBOW, "c": R_WRIST,
             "good": (30, 170), "warn": (20, 175)},
            {"name": "L UpperArm", "a": L_HIP,      "b": L_SHOULDER, "c": L_ELBOW,
             "good": (60, 120), "warn": (45, 135)},
            {"name": "R UpperArm", "a": R_HIP,      "b": R_SHOULDER, "c": R_ELBOW,
             "good": (60, 120), "warn": (45, 135)},
        ],
        "rep_angle_left":  (L_SHOULDER, L_ELBOW, L_WRIST),
        "rep_angle_right": (R_SHOULDER, R_ELBOW, R_WRIST),
        "EXTEND_THR": 150,
        "CURL_THR":    60,
        "description": "Neutral grip · Elbows stationary · Controlled tempo · Full ROM",
        "tip_curl":   "HOLD!",
        "tip_extend": "LOWER SLOW",
        "is_press": False,
    },
    "SHOULDER PRESS": {
        "joints": [
            {"name": "L Elbow", "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "good": (80, 180), "warn": (70, 180)},
            {"name": "R Elbow", "a": R_SHOULDER, "b": R_ELBOW, "c": R_WRIST,
             "good": (80, 180), "warn": (70, 180)},
            {"name": "Trunk",   "a": L_SHOULDER, "b": L_HIP,   "c": L_KNEE,
             "good": (160, 180), "warn": (145, 180)},
        ],
        "rep_angle_left":  (L_SHOULDER, L_ELBOW, L_WRIST),
        "rep_angle_right": (R_SHOULDER, R_ELBOW, R_WRIST),
        # For press: DOWN = angle < CURL_THR (arms at 90°), UP = angle > EXTEND_THR (locked out)
        # Rep counted when returning DOWN after full lockout
        "EXTEND_THR": 160,
        "CURL_THR":   100,
        "description": "Drive straight up · No arching back · Lock out at top · Control descent",
        "tip_curl":   "LOCK OUT!",
        "tip_extend": "90° RESET",
        "is_press": True,
    },
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))

def get_kp(kps, idx):
    if idx >= len(kps): return None
    kp = kps[idx]
    if kp[2] < 0.3: return None
    return (float(kp[0]), float(kp[1]), float(kp[2]))

def pts_from_kp_triple(kps, triple):
    pts = [get_kp(kps, i) for i in triple]
    if not all(pts): return None, None, None
    return pts

# ─── PER-PERSON TRACKER ───────────────────────────────────────────────────────
#
#  ARM STATE MACHINE per arm:
#
#   "WAIT_EXTEND"  →  waiting for arm to straighten first (init state)
#   "EXTENDED"     →  arm is straight, ready to curl
#   "CURLED"       →  arm has been curled up; waiting to come back down
#
#  Transition table (curl style):
#   WAIT_EXTEND  + angle > EXTEND_THR  →  EXTENDED
#   EXTENDED     + angle < CURL_THR    →  CURLED
#   CURLED       + angle > EXTEND_THR  →  EXTENDED  + COUNT REP ✓
#
#  For press style (inverted):
#   WAIT_EXTEND  + angle < CURL_THR    →  EXTENDED  (arms at 90° = "start")
#   EXTENDED     + angle > EXTEND_THR  →  CURLED    (locked out = "top")
#   CURLED       + angle < CURL_THR    →  EXTENDED  + COUNT REP ✓
#
class PersonTracker:
    def __init__(self):
        self.left_state   = defaultdict(lambda: "WAIT_EXTEND")
        self.right_state  = defaultdict(lambda: "WAIT_EXTEND")
        self.left_reps    = defaultdict(int)
        self.right_reps   = defaultdict(int)
        self.form_scores  = defaultdict(lambda: deque(maxlen=30))
        self.angle_hist_l = defaultdict(lambda: deque(maxlen=120))
        self.angle_hist_r = defaultdict(lambda: deque(maxlen=120))
        self.id_map       = {}
        self.next_id      = 0
        self.tip_timer    = defaultdict(int)
        self.current_tip  = defaultdict(str)
        # flash on rep complete
        self.flash_timer  = defaultdict(int)

    def match_id(self, cx, cy):
        best_id, best_d = None, 140
        for prev_id, (px, py) in self.id_map.items():
            d = math.hypot(cx-px, cy-py)
            if d < best_d:
                best_d, best_id = d, prev_id
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        self.id_map[best_id] = (cx, cy)
        return best_id

    def _step_arm(self, pid, angle, state_attr, rep_attr, cfg):
        """Advance one arm's state machine. Returns True if rep completed."""
        state        = getattr(self, state_attr)[pid]
        ext_thr      = cfg["EXTEND_THR"]
        curl_thr     = cfg["CURL_THR"]
        is_press     = cfg["is_press"]
        completed    = False

        if not is_press:
            # ── CURL style ──────────────────────────────────────────────────
            if state == "WAIT_EXTEND":
                if angle > ext_thr:
                    getattr(self, state_attr)[pid] = "EXTENDED"
            elif state == "EXTENDED":
                if angle < curl_thr:
                    getattr(self, state_attr)[pid] = "CURLED"
                    self.current_tip[pid] = cfg["tip_curl"]
                    self.tip_timer[pid]   = 50
            elif state == "CURLED":
                if angle > ext_thr:
                    getattr(self, state_attr)[pid] = "EXTENDED"
                    getattr(self, rep_attr)[pid]   += 1
                    completed = True
                    self.current_tip[pid] = cfg["tip_extend"]
                    self.tip_timer[pid]   = 40
                    self.flash_timer[pid] = 12
        else:
            # ── PRESS style ─────────────────────────────────────────────────
            if state == "WAIT_EXTEND":
                if angle < curl_thr:
                    getattr(self, state_attr)[pid] = "EXTENDED"   # at start position
            elif state == "EXTENDED":
                if angle > ext_thr:
                    getattr(self, state_attr)[pid] = "CURLED"     # locked out = top
                    self.current_tip[pid] = cfg["tip_curl"]
                    self.tip_timer[pid]   = 50
            elif state == "CURLED":
                if angle < curl_thr:
                    getattr(self, state_attr)[pid] = "EXTENDED"
                    getattr(self, rep_attr)[pid]   += 1
                    completed = True
                    self.current_tip[pid] = cfg["tip_extend"]
                    self.tip_timer[pid]   = 40
                    self.flash_timer[pid] = 12

        return completed

    def update(self, pid, left_angle, right_angle, cfg):
        """Update both arms. Returns (left_done, right_done)."""
        l_done = r_done = False
        if left_angle  is not None:
            l_done = self._step_arm(pid, left_angle,  "left_state",  "left_reps",  cfg)
        if right_angle is not None:
            r_done = self._step_arm(pid, right_angle, "right_state", "right_reps", cfg)
        if self.tip_timer[pid]   > 0: self.tip_timer[pid]   -= 1
        if self.flash_timer[pid] > 0: self.flash_timer[pid] -= 1
        return l_done, r_done

    def total_reps(self, pid):
        """
        Total completed reps = max(left, right).
        This means:
         - Both arms at same time (barbell style): each side counts → max = correct total
         - Alternating arms: each side counts separately → max = correct total
         - Single arm: only one side counts → correct
        """
        return max(self.left_reps[pid], self.right_reps[pid])

    def get_form_score(self, pid):
        h = list(self.form_scores[pid])
        return int(np.mean(h)) if h else 100

    def arm_phase(self, pid, arm):
        """Returns 'UP', 'DOWN', or '...' for display."""
        state = self.left_state[pid] if arm == "L" else self.right_state[pid]
        if state in ("EXTENDED", "WAIT_EXTEND"): return "DOWN"
        if state == "CURLED": return "UP"
        return "..."

# ─── DRAWING HELPERS ──────────────────────────────────────────────────────────
def glow_line(img, p1, p2, color, thickness=3, layers=4):
    for i in range(layers, 0, -1):
        alpha = 0.18 * i
        gc = tuple(min(255, int(c * alpha)) for c in color)
        cv2.line(img, p1, p2, gc, thickness + i * 3)
    cv2.line(img, p1, p2, color, thickness)

def glow_circle(img, center, radius, color, thickness=-1, layers=3):
    for i in range(layers, 0, -1):
        gc = tuple(min(255, int(c * 0.25 * i)) for c in color)
        cv2.circle(img, center, radius + i * 3, gc, 2)
    cv2.circle(img, center, radius, color, thickness)

def hud_text(img, text, pos, color=(0,200,255), scale=0.55, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2)
    cv2.putText(img, text, pos,        cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thick)

def draw_thick_bicep_arc(img, shoulder_pt, elbow_pt, wrist_pt, angle, color, label):
    """
    Draws a thick glowing arc at the elbow joint showing the curl angle,
    plus a curved muscle highlight between shoulder→elbow and elbow→wrist.
    """
    ex, ey = elbow_pt
    radius = 38

    # ── thick glow arc layers ──
    for layer, (r_off, alpha, thick) in enumerate([
        (14, 0.12, 10),
        ( 8, 0.25, 7),
        ( 3, 0.55, 4),
        ( 0, 1.00, 2),
    ]):
        gc = tuple(min(255, int(c * alpha)) for c in color)
        cv2.ellipse(img, (ex, ey), (radius+r_off, radius+r_off),
                    0, -20, int(angle)-20, gc, thick)

    # ── angle label inside arc ──
    lx = ex + int((radius+18) * math.cos(math.radians(angle/2)))
    ly = ey + int((radius+18) * math.sin(math.radians(angle/2)))
    hud_text(img, f"{int(angle)}", (lx-12, ly+5), color, 0.5, 2)

    # ── muscle line: shoulder → elbow (upper arm) ──
    draw_muscle_segment(img, shoulder_pt, elbow_pt, color)
    # ── muscle line: elbow → wrist (forearm) ──
    draw_muscle_segment(img, elbow_pt, wrist_pt, color)

def draw_muscle_segment(img, p1, p2, color):
    """Thick glowing bone/muscle line with a bright core."""
    # outer glow
    cv2.line(img, p1, p2, tuple(min(255, c//4) for c in color), 18)
    cv2.line(img, p1, p2, tuple(min(255, c//2) for c in color), 10)
    # mid glow
    cv2.line(img, p1, p2, color, 5)
    # bright white core
    cv2.line(img, p1, p2, (255,255,255), 1)
    # endpoint orbs
    for pt in [p1, p2]:
        glow_circle(img, pt, 7, color, -1, 3)

def draw_skeleton_body(frame, kps, sk_color, joint_color):
    """Draw only body/leg skeleton (arms drawn separately with muscle effect)."""
    h, w = frame.shape[:2]
    valid = {}
    for idx in range(len(kps)):
        kp = kps[idx]
        if len(kp) >= 3 and kp[2] > 0.3:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y < h:
                valid[idx] = (x, y)

    arm_kps = {L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST}
    body_pairs = [p for p in SKELETON_PAIRS
                  if p[0] not in arm_kps or p[1] not in arm_kps]

    for a, b in body_pairs:
        if a in valid and b in valid:
            glow_line(frame, valid[a], valid[b], sk_color, 2, 2)
    for idx, pt in valid.items():
        if idx not in arm_kps:
            glow_circle(frame, pt, 5, joint_color, -1, 2)
    return valid

def draw_rep_box(img, pid, tracker, pos):
    """Big clear rep counter box."""
    x, y = pos
    W, H = 200, 110
    # dark panel
    panel = np.zeros((H, W, 3), np.uint8)
    cv2.rectangle(panel, (0,0),(W-1,H-1),(8,8,24),-1)
    cv2.rectangle(panel, (0,0),(W-1,H-1),GLOW["hud_accent"],2)
    roi = img[y:y+H, x:x+W]
    if roi.shape[:2] == (H,W):
        cv2.addWeighted(roi, 0.25, panel, 0.85, 0, roi)

    total = tracker.total_reps(pid)
    l     = tracker.left_reps[pid]
    r     = tracker.right_reps[pid]
    lph   = tracker.arm_phase(pid, "L")
    rph   = tracker.arm_phase(pid, "R")

    # "REPS" label
    hud_text(img, "REPS", (x+75, y+22), GLOW["hud_accent"], 0.55, 1)

    # big number — changes color by count
    if total == 0:
        num_col = (180, 180, 180)
    elif total < 5:
        num_col = (0, 220, 100)
    elif total < 10:
        num_col = (0, 200, 255)
    else:
        num_col = (255, 200, 0)

    hud_text(img, str(total), (x+70, y+82), num_col, 2.2, 6)

    # per-arm counts + phase
    hud_text(img, f"L: {l} ({lph})", (x+6,  y+100), GLOW["left_arm"],  0.38, 1)
    hud_text(img, f"R: {r} ({rph})", (x+108, y+100), GLOW["right_arm"], 0.38, 1)

def draw_form_bar(img, score, pos, width=130, label="FORM"):
    x, y = pos
    hb = 16
    cv2.rectangle(img, (x,y),(x+width,y+hb),(25,25,45),-1)
    fill = int(width * score / 100)
    col = GLOW["bar_good"] if score >= 70 else ((0,200,255) if score >= 45 else GLOW["bar_bad"])
    cv2.rectangle(img, (x,y),(x+fill,y+hb), col, -1)
    cv2.rectangle(img, (x,y),(x+width,y+hb),(80,80,110),1)
    hud_text(img, f"{label}: {score}%", (x, y-6), col, 0.42, 1)

def draw_angle_panel(img, angle_results, pos, panel_w=215):
    x, y = pos
    row_h = 24
    ph = 22 + len(angle_results) * row_h
    panel = np.zeros((ph, panel_w, 3), np.uint8)
    cv2.rectangle(panel,(0,0),(panel_w-1,ph-1),(12,12,32),-1)
    cv2.rectangle(panel,(0,0),(panel_w-1,ph-1),GLOW["hud_accent"],1)
    for i, (name, angle, status) in enumerate(angle_results):
        col = (GLOW["angle_good"] if status=="GOOD" else
               GLOW["angle_warn"] if status=="WARN" else GLOW["angle_bad"])
        ry = 18 + i * row_h
        bf = int((panel_w-100) * min(angle,180)/180)
        cv2.rectangle(panel,(94,ry-12),(94+bf,ry+2), tuple(c//3 for c in col),-1)
        cv2.putText(panel, f"{name:<13}", (4,ry),  cv2.FONT_HERSHEY_SIMPLEX, 0.33,(180,180,210),1)
        cv2.putText(panel, f"{int(angle):3d}deg",  (90,ry), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    roi = img[y:y+ph, x:x+panel_w]
    if roi.shape == panel.shape:
        cv2.addWeighted(roi, 0.2, panel, 0.9, 0, roi)

def draw_vertical_progress(img, angle, label, pos, color, cfg):
    """Vertical progress bar: shows how far through the curl range."""
    x, y = pos
    bh, bw = 100, 20
    ext = cfg["EXTEND_THR"]
    curl= cfg["CURL_THR"]
    is_p= cfg["is_press"]

    cv2.rectangle(img,(x,y),(x+bw,y+bh),(20,20,40),-1)

    if not is_p:
        pct = np.clip((ext - angle) / (ext - curl + 1e-6), 0, 1)
    else:
        pct = np.clip((angle - curl) / (ext - curl + 1e-6), 0, 1)

    fh = int(bh * pct)
    # gradient fill
    for row in range(fh):
        t = row / (fh + 1e-6)
        c = tuple(int(a*(1-t) + b*t) for a,b in zip((0,255,0),(255,200,0)))
        yy = y + bh - row
        cv2.line(img, (x,yy),(x+bw,yy), tuple(min(255,v) for v in color), 1)

    cv2.rectangle(img,(x+1,y+bh-fh),(x+bw-1,y+bh),color,-1)
    cv2.rectangle(img,(x,y),(x+bw,y+bh),(80,80,110),1)
    hud_text(img, label,          (x+1, y+bh+15), color, 0.4, 1)
    hud_text(img, f"{int(pct*100)}%", (x-2, y-8),  color, 0.35, 1)

def draw_angle_graph(img, hist_l, hist_r, pos, gw=160, gh=55):
    x, y = pos
    g = np.zeros((gh,gw,3),np.uint8)
    cv2.rectangle(g,(0,0),(gw-1,gh-1),(12,12,32),-1)
    cv2.rectangle(g,(0,0),(gw-1,gh-1),(60,60,90),1)
    # grid lines
    for ang in [60,90,120,150]:
        gy2 = gh - int(ang/180*gh)
        cv2.line(g,(0,gy2),(gw,gy2),(40,40,60),1)
    for hist, col in [(list(hist_l), GLOW["left_arm"]),(list(hist_r), GLOW["right_arm"])]:
        pts = hist[-gw:]
        if len(pts) > 2:
            for pi in range(1, len(pts)):
                ax = int((pi-1)/len(pts)*gw)
                bx = int(pi/len(pts)*gw)
                ay = gh - int(pts[pi-1]/180*gh)
                by = gh - int(pts[pi]/180*gh)
                cv2.line(g,(ax,ay),(bx,by),col,2)
    roi = img[y:y+gh, x:x+gw]
    if roi.shape == g.shape:
        cv2.addWeighted(roi, 0.25, g, 0.85, 0, roi)
    hud_text(img,"ELBOW ANGLE HISTORY",(x,y-6),(110,110,180),0.3,1)
    hud_text(img,"■L",(x+gw-30,y-6),GLOW["left_arm"],0.3,1)
    hud_text(img,"■R",(x+gw-14,y-6),GLOW["right_arm"],0.3,1)

def draw_warning_overlay(img, warnings_list):
    if not warnings_list: return
    h = img.shape[0]
    for i, w in enumerate(warnings_list[:3]):
        hud_text(img, f"!  {w}", (14, h-75+i*20), GLOW["warning_text"], 0.46, 1)

def draw_flash(img, cx, cy, timer):
    if timer <= 0: return
    alpha = timer / 12.0
    overlay = img.copy()
    cv2.circle(overlay, (int(cx), int(cy)), int(90 * alpha), (0,255,180), 4)
    cv2.addWeighted(img, 1-alpha*0.3, overlay, alpha*0.3, 0, img)

REP_MILESTONES = {
    5:  "5 REPS! 💪",
    10: "10 REPS! BEAST!",
    15: "15! INSANE!",
    20: "20 REPS! LEGEND!",
    25: "25! GOD MODE!",
}

# ─── MAIN COMPOSE ─────────────────────────────────────────────────────────────
def moving_average(seq, window=5):
    if len(seq) < window:
        return np.mean(seq) if seq else None
    return np.mean(list(seq)[-window:])

def compose_frame(frame, results, tracker, frame_idx, total_frames, fps, exercise="BICEP CURL"):
    h, w   = frame.shape[:2]
    output = frame.copy()
    cfg    = EXERCISES[exercise]

    result   = results[0]
    n_people = 0
    total_form_list = []

    if result.boxes is not None and result.keypoints is not None:
        boxes   = result.boxes.xyxy.cpu().numpy()
        kps_all = result.keypoints.data.cpu().numpy()
        n_people = len(boxes)

        for i, (box, kps) in enumerate(zip(boxes, kps_all)):
            x1,y1,x2,y2 = map(int, box)
            cx = (x1+x2)/2; cy = (y1+y2)/2
            pid = tracker.match_id(cx, cy)

            # ── joint angles ──
            angle_results  = []
            frame_warnings = []
            joint_scores   = []

            for jcfg in cfg["joints"]:
                pa = get_kp(kps, jcfg["a"])
                pb = get_kp(kps, jcfg["b"])
                pc = get_kp(kps, jcfg["c"])
                if pa and pb and pc:
                    angle = calc_angle((pa[0],pa[1]),(pb[0],pb[1]),(pc[0],pc[1]))
                    lo_g,hi_g = jcfg["good"]; lo_w,hi_w = jcfg["warn"]
                    if lo_g <= angle <= hi_g:
                        status="GOOD"; joint_scores.append(100)
                    elif lo_w <= angle <= hi_w:
                        status="WARN"; joint_scores.append(55)
                        frame_warnings.append(f"{jcfg['name']}: {int(angle)}deg  adjust!")
                    else:
                        status="BAD";  joint_scores.append(15)
                        frame_warnings.append(f"{jcfg['name']}: {int(angle)}deg  INJURY RISK")
                    angle_results.append((jcfg["name"], angle, status))

            # ── elbow angles for rep counting ──
            la = [get_kp(kps,k) for k in cfg["rep_angle_left"]]
            ra = [get_kp(kps,k) for k in cfg["rep_angle_right"]]

            left_angle = right_angle = None
            if all(la):
                raw_left_angle = calc_angle((la[0][0],la[0][1]),(la[1][0],la[1][1]),(la[2][0],la[2][1]))
                tracker.angle_hist_l[pid].append(raw_left_angle)
                left_angle = moving_average(tracker.angle_hist_l[pid], window=5)
            if all(ra):
                raw_right_angle = calc_angle((ra[0][0],ra[0][1]),(ra[1][0],ra[1][1]),(ra[2][0],ra[2][1]))
                tracker.angle_hist_r[pid].append(raw_right_angle)
                right_angle = moving_average(tracker.angle_hist_r[pid], window=5)

            tracker.update(pid, left_angle, right_angle, cfg)

            # ── form score ──
            fscore = int(np.mean(joint_scores)) if joint_scores else 80
            tracker.form_scores[pid].append(fscore)
            smooth = tracker.get_form_score(pid)
            total_form_list.append(smooth)

            if smooth >= 75: sk_col,jt_col = GLOW["skeleton_good"],GLOW["joint_good"]
            elif smooth>=45: sk_col,jt_col = GLOW["skeleton_warn"],GLOW["joint_warn"]
            else:            sk_col,jt_col = GLOW["skeleton_bad"], GLOW["joint_bad"]

            # ── draw body skeleton ──
            valid = draw_skeleton_body(output, kps, sk_col, jt_col)

            # ── draw thick bicep/forearm muscle lines ──
            ls = valid.get(L_SHOULDER); le = valid.get(L_ELBOW); lw = valid.get(L_WRIST)
            rs = valid.get(R_SHOULDER); re = valid.get(R_ELBOW); rw = valid.get(R_WRIST)

            if ls and le and lw and left_angle is not None:
                draw_thick_bicep_arc(output, ls, le, lw, left_angle, GLOW["bicep_glow_l"], "L")
            elif ls and le:
                draw_muscle_segment(output, ls, le, GLOW["left_arm"])
            if le and lw:
                draw_muscle_segment(output, le, lw, GLOW["left_arm"])

            if rs and re and rw and right_angle is not None:
                draw_thick_bicep_arc(output, rs, re, rw, right_angle, GLOW["bicep_glow_r"], "R")
            elif rs and re:
                draw_muscle_segment(output, rs, re, GLOW["right_arm"])
            if re and rw:
                draw_muscle_segment(output, re, rw, GLOW["right_arm"])

            # ── bounding box ──
            col_box = sk_col
            glow_line(output,(x1,y1),(x2,y1),col_box,2,2)
            glow_line(output,(x1,y2),(x2,y2),col_box,2,2)
            glow_line(output,(x1,y1),(x1,y2),col_box,2,2)
            glow_line(output,(x2,y1),(x2,y2),col_box,2,2)

            # ── exercise badge ──
            bw2 = len(exercise)*9+20
            ov = output.copy()
            cv2.rectangle(ov,(x1,y1-22),(x1+bw2,y1+2),(0,0,0),-1)
            cv2.addWeighted(ov,0.5,output,0.5,0,output)
            cv2.rectangle(output,(x1,y1-22),(x1+bw2,y1+2),GLOW["hud_accent"],1)
            hud_text(output, exercise, (x1+6,y1), GLOW["hud_accent"], 0.46, 1)

            # ── REP COUNTER — prominent, top of frame ──
            rc_x = max(x1, 4)
            rc_y = max(y1 - 130, 50)
            draw_rep_box(output, pid, tracker, (rc_x, rc_y))

            # ── form bar ──
            draw_form_bar(output, smooth, (rc_x+210, rc_y+40), 130)

            # ── vertical progress bars ──
            bar_y = y1 + int((y2-y1)*0.15)
            if left_angle is not None:
                draw_vertical_progress(output, left_angle, "L", (max(x1-36,2), bar_y), GLOW["left_arm"],  cfg)
            if right_angle is not None:
                draw_vertical_progress(output, right_angle, "R", (min(x2+16,w-26), bar_y), GLOW["right_arm"], cfg)

            # ── angle panel ──
            if angle_results:
                draw_angle_panel(output, angle_results, (min(x2+8,w-220), y1), 215)

            # ── angle graph ──
            draw_angle_graph(output, tracker.angle_hist_l[pid],
                             tracker.angle_hist_r[pid], (w-170, 95))

            # ── rep flash ──
            draw_flash(output, cx, cy, tracker.flash_timer[pid])

            # ── phase tip ──
            tip = tracker.current_tip[pid] if tracker.tip_timer[pid] > 0 else ""
            if tip:
                tx = int(cx) - len(tip)*7
                hud_text(output, tip, (tx, int(y1)-60), (0,255,255), 0.85, 3)

            # ── milestone ──
            tot = tracker.total_reps(pid)
            if tot in REP_MILESTONES and tracker.flash_timer[pid] > 5:
                hud_text(output, REP_MILESTONES[tot],
                         (int(cx)-80, int(y1)-95), (255,255,0), 0.85, 3)

            # ── form tip ──
            hud_text(output, cfg["description"], (x1, min(y2+80,h-10)),
                     (160,210,255), 0.33, 1)

            # ── person ID ──
            hud_text(output, f"ID:{pid}", (x1, y1-42), GLOW["hud_accent"], 0.4, 1)

            # ── warnings ──
            draw_warning_overlay(output, frame_warnings)

    # ── HUD HEADER ────────────────────────────────────────────────────────────
    cv2.rectangle(output,(0,0),(w,48),(0,0,0),-1)
    cv2.addWeighted(output[0:48], 0.45,
                    np.zeros((48,w,3),np.uint8), 0.55, 0, output[0:48])
    hud_text(output, f"DUMBBELL FORM CHECKER & REP COUNTER  [{exercise}]",
             (10,30), GLOW["hud_accent"], 0.58, 2)
    hud_text(output, f"FPS:{fps:.0f}  {frame_idx}/{total_frames}  {DEVICE.upper()}",
             (w-280,30),(180,255,180),0.44)

    hud_text(output,"■ LEFT",  (10,65), GLOW["left_arm"],  0.4,1)
    hud_text(output,"■ RIGHT", (78,65), GLOW["right_arm"], 0.4,1)
    hud_text(output,f"PEOPLE: {n_people}", (160,65),(200,255,200),0.4)

    avg_form = int(np.mean(total_form_list)) if total_form_list else 0
    hud_text(output,f"AVG FORM: {avg_form}%",(280,65),
             GLOW["rep_counter"] if avg_form>=70 else GLOW["warning_text"],0.4)

    # progress bar bottom
    prog = frame_idx / max(total_frames,1)
    cv2.rectangle(output,(0,h-5),(w,h),(20,20,40),-1)
    cv2.rectangle(output,(0,h-5),(int(w*prog),h),GLOW["hud_accent"],-1)

    hud_text(output,"tubakhxn | github.com/tubakhxn",
             (w-268,h-16),(70,70,110),0.34)
    return output

# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────
def process_video(video_path, exercise="BICEP CURL"):
    print(f"\033[92m[5/6] Processing: {video_path}  |  Exercise: {exercise}\033[0m")

    import shutil

    base_dir = Path(__file__).parent

    out_dir = base_dir / "output_dumbbell"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(exist_ok=True)

    shots_dir = base_dir / "screenshots_dumbbell"

    if shots_dir.exists():
        shutil.rmtree(shots_dir)

    shots_dir.mkdir(exist_ok=True)

    output_path = str(base_dir / "output_dumbbell_ai.mp4")

    tmp_path = str(out_dir / "tmp.mp4")

    cap = cv2.VideoCapture(video_path)
        
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\033[91m[ERROR] Cannot open {video_path}\033[0m"); sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer  = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), orig_fps, (fw,fh))
    tracker = PersonTracker()
    frame_idx = 0; t_start = time.time()
    shot_interval = max(1, total_frames//10)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        elapsed = time.time()-t_start
        fps     = frame_idx/elapsed if elapsed>0 else orig_fps
        eta     = (total_frames-frame_idx)/fps if fps>0 else 0

        results  = model(frame, verbose=False, device=DEVICE, half=USE_FP16, classes=[0])
        composed = compose_frame(frame, results, tracker, frame_idx, total_frames, fps, exercise)
        writer.write(composed)

        if frame_idx % shot_interval == 0:
            cv2.imwrite(str(shots_dir/f"shot_{frame_idx:05d}.jpg"), composed)

        pct = int(100*frame_idx/total_frames)
        bar = "█"*(pct//3)+"░"*(34-pct//3)
        print(f"\r\033[93m  [{bar}] {pct}%  ETA:{eta:.1f}s  FPS:{fps:.1f}\033[0m", end="", flush=True)

    print(); cap.release(); writer.release()

    print(f"\033[92m[6/6] Compressing...\033[0m")
    try:
        r = subprocess.run(["ffmpeg","-y","-i",tmp_path,"-vcodec","libx264",
                            "-crf","23","-preset","fast",output_path], capture_output=True)
        if r.returncode != 0:
            import shutil; shutil.copy(tmp_path, output_path)
    except FileNotFoundError:
        import shutil; shutil.copy(tmp_path, output_path)

    print(f"\033[92m[✓] Output → {output_path}\033[0m")
    print(f"\033[92m[✓] Screenshots → {shots_dir}/\033[0m")
    print(f"\033[94m[i] Size: {os.path.getsize(output_path)/1e6:.2f} MB\033[0m")

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()
    if len(sys.argv) < 2:
        print("\033[96mUsage:\033[0m")
        print("  python main.py video.mp4")
        print("  python main.py video.mp4 'BICEP CURL'")
        print("  python main.py video.mp4 'HAMMER CURL'")
        print("  python main.py video.mp4 'SHOULDER PRESS'")
        sys.exit(1)

    video  = sys.argv[1]
    ex_arg = " ".join(sys.argv[2:]).upper() if len(sys.argv)>2 else "BICEP CURL"
    if ex_arg not in EXERCISES:
        print(f"\033[91m[WARN] Unknown exercise '{ex_arg}'. Using BICEP CURL.\033[0m")
        ex_arg = "BICEP CURL"

    process_video(video, ex_arg)