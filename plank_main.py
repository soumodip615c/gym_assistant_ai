import sys
import os
import subprocess
import importlib
import time
import math
from pathlib import Path

# ─── AUTO INSTALLER ──────────────────────────────────────────────────────────
REQUIRED = [
    ("ultralytics", "ultralytics"),
    ("cv2",         "opencv-python"),
    ("numpy",       "numpy"),
    ("torch",       "torch"),
    ("tqdm",        "tqdm"),
]

def print_banner():
    print("\033[96m")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       AI PLANK FORM CHECKER & HOLD TIMER                    ║")
    print("║       YOLOv8 Pose · Body Alignment · Cinematic HUD          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\033[0m")

def install_step(step, total, name):
    bar_len = 30
    fill = int(bar_len * step / total)
    bar  = "█" * fill + "░" * (bar_len - fill)
    pct  = int(100 * step / total)
    print(f"\r\033[93m[{step}/{total}] {name:<35} [{bar}] {pct}%\033[0m",
          end="", flush=True)

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
            install_step(i + 1, len(missing), f"Installing {pkg}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
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
print("\033[93m[MODEL] Loading YOLOv8 pose model (auto-downloads ~6MB)...\033[0m")
model = YOLO("yolov8n-pose.pt")
print("\033[92m[MODEL] Ready.\033[0m")

# ─── KEYPOINT INDICES (COCO 17-point) ────────────────────────────────────────
NOSE=0
L_EYE=1;  R_EYE=2
L_EAR=3;  R_EAR=4
L_SHOULDER=5;  R_SHOULDER=6
L_ELBOW=7;     R_ELBOW=8
L_WRIST=9;     R_WRIST=10
L_HIP=11;      R_HIP=12
L_KNEE=13;     R_KNEE=14
L_ANKLE=15;    R_ANKLE=16

SKELETON_PAIRS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),  (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),  (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),        (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),        (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER),     (NOSE, R_SHOULDER),
]

# ─── COLORS ──────────────────────────────────────────────────────────────────
C = {
    "good":    (0,  255, 120),
    "warn":    (0,  190, 255),
    "bad":     (60,  50, 255),
    "accent":  (0,  200, 255),
    "timer":   (0,  255, 120),
    "trail":   (255, 140,  0),
    "white":   (220, 220, 220),
    "dim":     (100, 100, 130),
}

# ─── PLANK JOINT CHECKS ──────────────────────────────────────────────────────
PLANK_CHECKS = [
    # (display_name, joint_A, joint_B(vertex), joint_C, good_range, warn_range)
    ("Body Line",   L_SHOULDER, L_HIP,      L_ANKLE,    (160,180), (145,180)),
    ("Hip Level",   L_SHOULDER, L_HIP,      L_KNEE,     (160,180), (148,180)),
    ("Neck",        NOSE,       L_SHOULDER, L_HIP,      (155,180), (138,180)),
    ("L Elbow",     L_SHOULDER, L_ELBOW,    L_WRIST,    (80, 100), (70, 115)),
    ("R Elbow",     R_SHOULDER, R_ELBOW,    R_WRIST,    (80, 100), (70, 115)),
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))

def get_kp(kps, idx):
    if idx >= len(kps): return None
    kp = kps[idx]
    return (float(kp[0]), float(kp[1])) if float(kp[2]) > 0.25 else None

def status_color(status):
    return C["good"] if status == "GOOD" else C["warn"] if status == "WARN" else C["bad"]

def glow_line(img, p1, p2, color, thick=2, layers=3):
    for i in range(layers, 0, -1):
        gc = tuple(min(255, int(c * 0.2 * i)) for c in color)
        cv2.line(img, p1, p2, gc, thick + i * 2)
    cv2.line(img, p1, p2, color, thick)

def glow_circle(img, center, r, color, layers=2):
    for i in range(layers, 0, -1):
        gc = tuple(min(255, int(c * 0.25)) for c in color)
        cv2.circle(img, center, r + i * 2, gc, 2)
    cv2.circle(img, center, r, color, -1)

def txt(img, text, pos, color=None, scale=0.5, thick=1):
    color = color or C["accent"]
    x, y = pos
    cv2.putText(img, text, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1)
    cv2.putText(img, text, (x,  y),   cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thick)

# ─── PLANK TRACKER ───────────────────────────────────────────────────────────
class PlankTracker:
    def __init__(self):
        self.timers      = {}           # pid -> start time (resets on bad form)
        self.form_hist   = defaultdict(lambda: deque(maxlen=30))
        self.id_map      = {}           # pid -> (cx, cy)
        self.next_id     = 0
        self.milestones  = defaultdict(set)  # pid -> set of hit milestones

    def match_id(self, cx, cy):
        best_id, best_d = None, 120
        for pid, (px, py) in self.id_map.items():
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best_d, best_id = d, pid
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        self.id_map[best_id] = (cx, cy)
        return best_id

    def update_timer(self, pid, form_score):
        if pid not in self.timers:
            self.timers[pid] = time.time()
        if form_score < 35:
            self.timers[pid] = time.time()   # bad form resets hold
        return int(time.time() - self.timers[pid])

    def smooth_score(self, pid):
        h = list(self.form_hist[pid])
        return int(np.mean(h)) if h else 100

# ─── DRAW FUNCTIONS ──────────────────────────────────────────────────────────
def draw_skeleton(frame, kps, color):
    h, w = frame.shape[:2]
    valid = {}
    for idx in range(min(len(kps), 17)):
        pt = get_kp(kps, idx)
        if pt and 0 <= int(pt[0]) < w and 0 <= int(pt[1]) < h:
            valid[idx] = (int(pt[0]), int(pt[1]))
    for a, b in SKELETON_PAIRS:
        if a in valid and b in valid:
            glow_line(frame, valid[a], valid[b], color, 2, 2)
    for idx, pt in valid.items():
        glow_circle(frame, pt, 4, color)
    return valid

def draw_alignment_line(frame, kps):
    ls = get_kp(kps, L_SHOULDER)
    lh = get_kp(kps, L_HIP)
    la = get_kp(kps, L_ANKLE)
    if not (ls and lh and la):
        return None, None
    angle = calc_angle(ls, lh, la)
    col   = C["good"] if angle >= 160 else C["warn"] if angle >= 145 else C["bad"]
    glow_line(frame, (int(ls[0]),int(ls[1])), (int(la[0]),int(la[1])), col, 2, 3)
    mid = (int((ls[0]+la[0])/2) + 10, int((ls[1]+la[1])/2))
    label = "ALIGNED ✓" if angle>=160 else ("SLIGHT SAG" if angle>=145 else "HIP SAG !")
    txt(frame, label, mid, col, 0.45, 1)
    return angle, col

def draw_angle_checks(frame, kps, panel_x, panel_y):
    results = []
    for name, ai, bi, ci, good, warn in PLANK_CHECKS:
        pa = get_kp(kps, ai)
        pb = get_kp(kps, bi)
        pc = get_kp(kps, ci)
        if pa and pb and pc:
            angle = calc_angle(pa, pb, pc)
            if good[0] <= angle <= good[1]:
                status = "GOOD"
            elif warn[0] <= angle <= warn[1]:
                status = "WARN"
            else:
                status = "BAD"
            results.append((name, angle, status))
            # arc at joint B
            bx, by = int(pb[0]), int(pb[1])
            arc_col = status_color(status)
            cv2.ellipse(frame, (bx,by), (26,26), 0, 0, int(min(angle,180)),
                        tuple(c//4 for c in arc_col), 2)
            cv2.ellipse(frame, (bx,by), (26,26), 0, 0, int(min(angle,180)), arc_col, 1)
            txt(frame, f"{int(angle)}", (bx+28,by+5), arc_col, 0.32, 1)

    # side panel
    ph = 18 + len(results) * 22
    pw = 195
    panel = np.zeros((ph, pw, 3), np.uint8)
    cv2.rectangle(panel, (0,0),(pw-1,ph-1),(12,12,30),-1)
    cv2.rectangle(panel, (0,0),(pw-1,ph-1),C["accent"],1)
    for i,(name,angle,status) in enumerate(results):
        col = status_color(status)
        ry  = 14 + i*22
        bar = int((pw-95) * min(angle,180)/180)
        cv2.rectangle(panel,(90,ry-9),(90+bar,ry+1),tuple(c//3 for c in col),-1)
        cv2.putText(panel,f"{name:<12}",(4,ry),cv2.FONT_HERSHEY_SIMPLEX,0.32,(180,180,200),1)
        cv2.putText(panel,f"{int(angle):3d}",(82,ry),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)

    h, w = frame.shape[:2]
    py2  = min(panel_y, h - ph - 2)
    px2  = min(panel_x, w - pw - 2)
    roi  = frame[py2:py2+ph, px2:px2+pw]
    if roi.shape[:2] == (ph,pw):
        cv2.addWeighted(roi, 0.2, panel, 0.9, 0, roi)

    return results

def draw_hold_timer(frame, pid, hold_secs, form_score, pos):
    x, y = pos
    mins = hold_secs // 60
    secs = hold_secs  % 60
    time_str = f"{mins:02d}:{secs:02d}"
    bw, bh = 130, 72
    bg = np.zeros((bh, bw, 3), np.uint8)
    cv2.rectangle(bg,(0,0),(bw-1,bh-1),(10,10,28),-1)
    cv2.rectangle(bg,(0,0),(bw-1,bh-1),C["accent"],1)
    h, w = frame.shape[:2]
    ey = min(y+bh, h); ex = min(x+bw, w)
    roi = frame[y:ey, x:ex]
    bg_crop = bg[:ey-y, :ex-x]
    if roi.shape == bg_crop.shape:
        cv2.addWeighted(roi, 0.25, bg_crop, 0.9, 0, roi)
    col = C["good"] if hold_secs < 60 else (C["accent"] if hold_secs < 120 else (255,160,0))
    txt(frame, "HOLD TIME", (x+16, y+18), C["accent"], 0.42, 1)
    txt(frame, time_str,    (x+12, y+56), col, 1.1, 3)
    # quality dot
    dot_col = C["good"] if form_score>=70 else C["warn"] if form_score>=45 else C["bad"]
    cv2.circle(frame, (x+bw-10, y+10), 6, dot_col, -1)

def draw_form_bar(frame, score, pos, width=130):
    x, y = pos
    col  = C["good"] if score>=70 else C["warn"] if score>=45 else C["bad"]
    cv2.rectangle(frame,(x,y),(x+width,y+13),(25,25,45),-1)
    cv2.rectangle(frame,(x,y),(x+int(width*score/100),y+13),col,-1)
    cv2.rectangle(frame,(x,y),(x+width,y+13),(70,70,100),1)
    txt(frame, f"FORM: {score}%", (x, y-6), col, 0.38, 1)

def draw_warnings(frame, warnings_list):
    h = frame.shape[0]
    for i, w_msg in enumerate(warnings_list[:4]):
        txt(frame, f"  {w_msg}", (14, h-90+i*20), C["bad"], 0.48, 1)

def draw_milestone(frame, hold_secs, pid, tracker):
    milestones = {30:"30 SECS!", 60:"1 MINUTE!", 90:"90 SECS!", 120:"2 MINUTES!", 180:"3 MINS! BEAST MODE!"}
    h, w = frame.shape[:2]
    for ms, label in milestones.items():
        if hold_secs >= ms and ms not in tracker.milestones[pid]:
            tracker.milestones[pid].add(ms)
        if ms in tracker.milestones[pid] and hold_secs >= ms:
            # show badge for 3 seconds after hitting milestone
            txt(frame, f"🏆 {label}", (w//2 - 100, h//2 - 30),
                (0,255,255), 1.0, 3)
            break

# ─── COMPOSE FRAME ───────────────────────────────────────────────────────────
def compose_frame(frame, results, tracker, frame_idx, total_frames, fps):
    h, w   = frame.shape[:2]
    output = frame.copy()

    # subtle scanline
    scan = np.zeros_like(output)
    for row in range(0, h, 4):
        scan[row] = 0
    cv2.addWeighted(output, 0.93, scan, 0.07, 0, output)

    result   = results[0]
    n_people = 0

    if result.boxes is not None and result.keypoints is not None:
        boxes   = result.boxes.xyxy.cpu().numpy()
        kps_all = result.keypoints.data.cpu().numpy()
        n_people = len(boxes)

        for i, (box, kps) in enumerate(zip(boxes, kps_all)):
            x1,y1,x2,y2 = map(int, box)
            cx = (x1+x2)/2;  cy = (y1+y2)/2
            pid = tracker.match_id(cx, cy)

            # ── joint checks ──
            check_results = draw_angle_checks(output, kps, x2+10, y1)
            warnings_list = []
            scores = []
            for cname, cangle, cstatus in check_results:
                if cstatus == "GOOD":   scores.append(100)
                elif cstatus == "WARN": scores.append(60);  warnings_list.append(f"⚠ {cname}: {int(cangle)}° — adjust")
                else:                   scores.append(15);  warnings_list.append(f"✖ {cname}: {int(cangle)}° — INJURY RISK")

            frame_score = int(np.mean(scores)) if scores else 80
            tracker.form_hist[pid].append(frame_score)
            smooth = tracker.smooth_score(pid)

            # ── hold timer ──
            hold = tracker.update_timer(pid, smooth)

            # ── skeleton ──
            sk_col = C["good"] if smooth>=70 else C["warn"] if smooth>=45 else C["bad"]
            draw_skeleton(output, kps, sk_col)

            # ── alignment spine line ──
            draw_alignment_line(output, kps)

            # ── bounding box ──
            glow_line(output,(x1,y1),(x2,y1),sk_col,2,2)
            glow_line(output,(x1,y2),(x2,y2),sk_col,2,2)
            glow_line(output,(x1,y1),(x1,y2),sk_col,2,2)
            glow_line(output,(x2,y1),(x2,y2),sk_col,2,2)

            # badge
            bw_badge = 90
            ov = output.copy()
            cv2.rectangle(ov,(x1,y1-22),(x1+bw_badge,y1),(0,0,0),-1)
            cv2.addWeighted(ov,0.5,output,0.5,0,output)
            cv2.rectangle(output,(x1,y1-22),(x1+bw_badge,y1),C["accent"],1)
            txt(output,"PLANK",(x1+6,y1-6),C["accent"],0.45,1)
            txt(output,f"ID:{pid}",(x1,y1-28),C["dim"],0.35,1)

            # ── timer & form bar ──
            tx = max(x1, 4)
            ty = min(y2+10, h-80)
            draw_hold_timer(output, pid, hold, smooth, (tx, ty))
            draw_form_bar(output, smooth, (tx+138, ty+30), 130)

            # ── tip ──
            txt(output,"Keep straight · No hip sag · Neutral neck · Elbows 90°",
                (x1, y2+100),(180,220,255),0.33,1)

            # ── warnings ──
            draw_warnings(output, warnings_list)

            # ── milestone ──
            draw_milestone(output, hold, pid, tracker)

    # ── HUD HEADER ───────────────────────────────────────────────────────────
    progress = frame_idx / max(total_frames,1)
    cv2.rectangle(output,(0,0),(w,42),(0,0,0),-1)
    cv2.addWeighted(output[0:42],0.5,np.zeros((42,w,3),np.uint8),0.5,0,output[0:42])
    txt(output,"▸ AI PLANK FORM CHECKER & HOLD TIMER",(10,28),C["accent"],0.6,2)
    txt(output,f"FPS:{fps:.0f}  FRAME:{frame_idx}/{total_frames}  DEVICE:{DEVICE.upper()}",
        (w-340,28),(180,255,180),0.45)

    # stats
    txt(output,f"PEOPLE IN FRAME: {n_people}",(10,60),(200,255,200),0.5)

    # progress bar
    cv2.rectangle(output,(0,h-6),(w,h),(20,20,40),-1)
    cv2.rectangle(output,(0,h-6),(int(w*progress),h),C["accent"],-1)

    # watermark
    txt(output,"tubakhxn | github.com/tubakhxn",(w-268,h-18),(60,60,100),0.35)

    return output

# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────
def process_video(video_path):
    print(f"\033[92m[5/6] Processing: {video_path}\033[0m")
    base_dir = Path(__file__).parent
    import shutil
    out_dir = base_dir / "output_dumbbell"
    if out_dir.exists():
     shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    shots_dir = base_dir / "screenshots_dumbbell"
    if shots_dir.exists():
     shutil.rmtree(shots_dir)
     shots_dir.mkdir(exist_ok=True) 
    output_path = str(base_dir / "output_plank_ai.mp4")
    tmp_path = str(out_dir / "tmp_plank.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\033[91m[ERROR] Cannot open: {video_path}\033[0m"); sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer  = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), orig_fps, (fw,fh))
    tracker = PlankTracker()

    frame_idx     = 0
    t_start       = time.time()
    shot_interval = max(1, total_frames // 10)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        elapsed = time.time() - t_start
        fps     = frame_idx / elapsed if elapsed > 0 else orig_fps
        eta     = (total_frames - frame_idx) / fps if fps > 0 else 0

        results  = model(frame, verbose=False, device=DEVICE, half=USE_FP16, classes=[0])
        composed = compose_frame(frame, results, tracker, frame_idx, total_frames, fps)
        writer.write(composed)

        if frame_idx % shot_interval == 0:
            cv2.imwrite(str(shots_dir / f"shot_{frame_idx:05d}.jpg"), composed)

        pct = int(100 * frame_idx / total_frames)
        bar = "█"*(pct//3) + "░"*(34-pct//3)
        print(f"\r\033[93m  [{bar}] {pct}%  ETA:{eta:.1f}s  FPS:{fps:.1f}\033[0m",
              end="", flush=True)

    print()
    cap.release()
    writer.release()

    print(f"\033[92m[6/6] Compressing output...\033[0m")
    try:
        ret = subprocess.run(
            ["ffmpeg","-y","-i",tmp_path,"-vcodec","libx264","-crf","23","-preset","fast",output_path],
            capture_output=True)
        if ret.returncode != 0:
            import shutil; shutil.copy(tmp_path, output_path)
    except FileNotFoundError:
        import shutil; shutil.copy(tmp_path, output_path)

    print(f"\033[92m[✓] Output  → {output_path}\033[0m")
    print(f"\033[92m[✓] Shots   → {shots_dir}/\033[0m")
    print(f"\033[94m[i] Size: {os.path.getsize(output_path)/1e6:.2f} MB\033[0m")

# ─── ENTRY ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()
    if len(sys.argv) < 2:
        print("\033[96mUsage: python main.py video.mp4\033[0m")
        sys.exit(1)
    process_video(sys.argv[1])

