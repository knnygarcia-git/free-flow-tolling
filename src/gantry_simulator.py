"""
Visual simulator of a Free Flow toll gantry: detection, tracking, plate reading (OCR), RFID tag
reading, vehicle classification and data generation, in a computer-vision dashboard layout.

THE SCENE IS SYNTHETIC: vehicles, plates and readings come from a stochastic model. What is real
is the pipeline logic, the same as on a gantry:

    camera -> detection -> tracking (ID) -> trigger line -> plate OCR
           -> RFID antenna reads the tag -> tag x plate fusion -> vehicle class -> billing event

Each pass becomes an event carrying the size in bytes of its evidence package (the same model as
src/data_volume.py). Events are written to reports/tables/simulated_events.csv.

Usage:
    python src/gantry_simulator.py               # 30 s scene, writes MP4 + GIF + PNG
    python src/gantry_simulator.py --seconds 60 --seed 7
"""
from __future__ import annotations

import argparse
import random
import string
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from data_volume import TRANSACTION_BYTES, EvidencePackage

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FONTS = ROOT / "assets" / "fonts"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- layout
W, H = 1280, 820
SCENE_W, SCENE_H = 900, 540
PANEL_X = SCENE_W + 12
FPS = 15

# ---------------------------------------------------------------- camera (simple pinhole)
HORIZON_Y = 95           # horizon line in the image
CAM_H = 6.0              # camera height on the gantry (m)
FOCAL = 900.0            # focal length in pixels (telephoto lens, typical for ANPR)
CX = SCENE_W * 0.47      # horizontal optical center
LANE_W = 3.6             # lane width (m)
LANES_X = [-LANE_W / 2, LANE_W / 2]    # center of each lane (m)
SHOULDER_X = LANE_W + 1.5
Z_MAX, Z_MIN = 120.0, 11.0
# Positions on the road (distance z to the camera filming the scene, in meters)
Z_GANTRY = 24.0          # the gantry: structure over the road; the laser scans exactly here
Z_TRIGGER = 34.0         # trigger line: the ANPR camera photographs the plate 10 m before the gantry
Z_RFID = (24.0, 40.0)    # RFID antenna read zone: 0 to 16 m before the gantry
BEAM_H = 6.6             # height of the gantry beam (m)
POST_X = (-LANE_W - 1.4, SHOULDER_X + 0.8)
# sensor colors (BGR) - the same in the scene, the side view and the legends
COLOR_ANPR = (60, 60, 235)
COLOR_RFID = (230, 160, 40)
COLOR_LASER = (40, 210, 250)
COLOR_CTX = (200, 120, 200)

# ---------------------------------------------------------------- colors (BGR)
YELLOW = (40, 200, 240)
WHITE = (255, 255, 255)
GREEN = (90, 200, 90)
RED = (70, 70, 230)
ORANGE_BGR = (58, 131, 232)
BLUE_BGR = (219, 111, 42)

CATEGORIES = {
    # name: (width m, height m, prob, ANTT class, axles, speed range km/h)
    "car":        (1.80, 1.45, 0.62, 1, 2, (85, 115)),
    "suv":         (1.95, 1.75, 0.16, 1, 2, (85, 112)),
    "truck_2axle": (2.50, 3.10, 0.09, 2, 2, (70, 88)),
    "truck_5axle": (2.60, 3.90, 0.08, 7, 5, (65, 82)),
    "bus":         (2.55, 3.40, 0.03, 4, 3, (75, 95)),
    "motorcycle":  (0.80, 1.40, 0.02, 9, 2, (85, 115)),
}
VEHICLE_COLORS = [(200, 200, 205), (40, 40, 45), (180, 60, 50), (60, 60, 160), (235, 235, 235),
              (120, 125, 130), (30, 90, 160), (60, 130, 70)]

P_TAG = 0.62             # share of vehicles with a tag (assumption)
P_OCR_ERROR = 0.06        # chance of a character being misread
CONFUSIONS = {"O": "0", "0": "O", "B": "8", "8": "B", "I": "1", "1": "I", "S": "5", "5": "S",
             "Z": "2", "2": "Z", "D": "0", "G": "6"}


def font(size: int, bold: bool = False):
    names = (["Poppins-SemiBold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold
             else ["Poppins-Regular.ttf", "DejaVuSans.ttf", "arial.ttf"])
    folders = [str(FONTS) + "/", "", "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/",
              "C:/Windows/Fonts/", "/Library/Fonts/", "/System/Library/Fonts/Supplemental/"]
    for n in names:
        for p in folders:
            try:
                return ImageFont.truetype(p + n, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


F_SM, F_MD, F_LG, F_XL = font(12), font(14), font(18, True), font(30, True)
F_BOX, F_MONO = font(13, True), font(12)


def project(x: float, z: float, h: float = 0.0) -> tuple[float, float]:
    """World (x lateral, z distance, h height) -> pixel."""
    return CX + FOCAL * x / z, HORIZON_Y + FOCAL * (CAM_H - h) / z


def ip(pt) -> tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def mercosur_plate(rng: random.Random) -> str:
    L = string.ascii_uppercase
    return (rng.choice(L) + rng.choice(L) + rng.choice(L) + rng.choice(string.digits)
            + rng.choice(L) + rng.choice(string.digits) + rng.choice(string.digits))


@dataclass
class Vehicle:
    vid: int
    kind: str
    lane: int
    z: float
    v: float                       # m/s
    plate: str
    has_tag: bool
    color: tuple
    trail: deque = field(default_factory=lambda: deque(maxlen=9))
    ocr: str | None = None
    conf: float | None = None
    tag_read: bool = False
    tag_id: str | None = None
    decision: str | None = None      # TAG / OCR / REVIEW
    event_sent: bool = False

    @property
    def dims(self):
        return CATEGORIES[self.kind][:2]

    @property
    def cat(self):
        return CATEGORIES[self.kind][3]

    @property
    def axles(self):
        return CATEGORIES[self.kind][4]

    def box(self):
        w, h = self.dims
        x = LANES_X[self.lane]
        xl, yb = project(x - w / 2, self.z)
        xr, _ = project(x + w / 2, self.z)
        scale = FOCAL / self.z
        return int(xl), int(yb - h * scale), int(xr), int(yb)


class Simulator:
    def __init__(self, seconds: int, seed: int):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.n_frames = seconds * FPS
        self.dt = 1 / FPS
        self.vehicles: list[Vehicle] = []
        self.next_id = 101
        self.events: list[dict] = []
        self.ev = EvidencePackage()
        self.bytes_total = 0.0
        self.flow_series: list[float] = []      # vehicles/min (10 s moving window)
        self.mb_series: list[float] = []
        self.heat = np.zeros((SCENE_H // 6, SCENE_W // 6), np.float32)
        self.intensities: list[str] = []
        # Demand profile over the scene (vehicles/s across both lanes), in phases
        t = np.linspace(0, 1, self.n_frames)
        self.rate = 0.3 + 1.3 * np.clip(np.sin(np.pi * t * 1.1) ** 2, 0, 1)
        self.background = None
        self.last_crop = None
        self.last_read = None
        self.state = "Low"
        self.gantry_layer = None
        self.activity = {}          # (sensor, lane) -> time of the last read
        self.t = 0.0
        self.sensor_count = {"anpr": 0, "rfid": 0, "laser": 0}

    # ------------------------------------------------------------ dynamics
    def spawn_arrivals(self, f: int):
        if self.rng.random() > self.rate[f] * self.dt:
            return
        lane = self.rng.choice([0, 1])
        same_lane = [v for v in self.vehicles if v.lane == lane]
        if same_lane and max(v.z for v in same_lane) > Z_MAX - 20:
            return
        kinds, weights = zip(*[(k, c[2]) for k, c in CATEGORIES.items()])
        kind = self.rng.choices(kinds, weights)[0]
        if lane == 1 and kind in ("truck_2axle", "truck_5axle", "bus") and self.rng.random() < .7:
            lane = 0   # heavy vehicles prefer the driver's right lane (left of the image)
        vmin, vmax = CATEGORIES[kind][5]
        self.vehicles.append(Vehicle(
            vid=self.next_id, kind=kind, lane=lane, z=Z_MAX,
            v=self.rng.uniform(vmin, vmax) / 3.6, plate=mercosur_plate(self.rng),
            has_tag=self.rng.random() < P_TAG, color=self.rng.choice(VEHICLE_COLORS)))
        self.next_id += 1

    def read_ocr(self, v: Vehicle):
        """Simulates OCR: confidence drops for small/dirty plates; sometimes swaps a character."""
        chars = list(v.plate)
        errors = 0
        for i, c in enumerate(chars):
            if c in CONFUSIONS and self.rng.random() < P_OCR_ERROR:
                chars[i] = CONFUSIONS[c]
                errors += 1
        base = self.rng.uniform(0.90, 0.995) if v.kind != "motorcycle" else self.rng.uniform(.6, .9)
        v.ocr = "".join(chars)
        v.conf = max(0.35, base - 0.18 * errors)

    def fuse(self, v: Vehicle):
        """Decision rule: a valid tag wins; without a tag, high-confidence OCR; otherwise, human review."""
        if v.tag_read:
            v.decision = "TAG"
        elif v.conf is not None and v.conf >= 0.90:
            v.decision = "OCR"
        else:
            v.decision = "REVIEW"

    def emit_event(self, v: Vehicle, t: float):
        size_bytes = self.ev.total + TRANSACTION_BYTES
        self.bytes_total += size_bytes
        self.events.append({
            "t_s": round(t, 2), "track_id": v.vid, "lane": v.lane + 1, "kind": v.kind,
            "antt_class": v.cat, "axles": v.axles, "speed_kmh": round(v.v * 3.6, 1),
            "true_plate": v.plate, "ocr_plate": v.ocr, "ocr_conf": round(v.conf, 3),
            "ocr_correct": v.ocr == v.plate, "has_tag": v.has_tag, "tag_read": v.tag_read,
            "tag_id": v.tag_id, "decision": v.decision, "event_bytes": int(size_bytes),
        })
        v.event_sent = True

    def step(self, f: int):
        t = f * self.dt
        self.t = t
        self.spawn_arrivals(f)
        for lane in (0, 1):
            queue = sorted([v for v in self.vehicles if v.lane == lane], key=lambda v: v.z)
            for i, v in enumerate(queue):
                speed = v.v
                if i > 0:   # do not drive through the vehicle ahead
                    gap = v.z - queue[i - 1].z
                    if gap < 18:
                        speed = min(speed, queue[i - 1].v)
                v.z -= speed * self.dt
                x1, y1, x2, y2 = v.box()
                v.trail.append(((x1 + x2) // 2, y2))
                if Z_RFID[0] <= v.z <= Z_RFID[1] and v.has_tag and not v.tag_read:
                    if self.rng.random() < 0.35:        # several read attempts inside the zone
                        v.tag_read = True
                        self.activity[("rfid", v.lane)] = t
                        self.sensor_count["rfid"] += 1
                        v.tag_id = "E2801160" + "".join(self.rng.choices("0123456789ABCDEF", k=8))
                if v.ocr is None and v.z <= Z_TRIGGER:
                    self.read_ocr(v)
                    self.last_read = v
                    self.activity[("anpr", v.lane)] = t
                    self.sensor_count["anpr"] += 1
                if v.ocr is not None and not v.event_sent and v.z <= Z_GANTRY:
                    # passed under the gantry: the laser measures the profile and closes the pass
                    self.activity[("laser", v.lane)] = t
                    self.sensor_count["laser"] += 1
                    self.fuse(v)
                    self.emit_event(v, t)
                a, b = max(0, x1 // 6), max(0, y1 // 6)
                self.heat[b:max(b, y2 // 6), a:max(a, x2 // 6)] += v.v * 0.08
        self.vehicles = [v for v in self.vehicles if v.z > Z_MIN]
        self.heat *= 0.93

        window = [e for e in self.events if e["t_s"] > t - 10]
        flow = len(window) * 6.0
        self.flow_series.append(flow)
        self.mb_series.append(sum(e["event_bytes"] for e in window) * 6 / 1e6)
        # hysteresis: only change state when the threshold is clearly crossed
        limits = {"Low": (None, 25), "Moderate": (15, 42), "High": (34, None)}
        lo, hi = limits[self.state]
        if hi is not None and flow > hi:
            self.state = "Moderate" if self.state == "Low" else "High"
        elif lo is not None and flow < lo:
            self.state = "Moderate" if self.state == "High" else "Low"
        self.intensities.append(self.state)

    # ------------------------------------------------------------ drawing
    def draw_road(self, img):
        if self.background is None:
            self.background = self._static_background()
        img[:] = self.background
        return img

    def _static_background(self):
        """Gantry and camera are fixed: the background is drawn only once."""
        img = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        img[:HORIZON_Y] = (205, 175, 140)                     # sky
        img[HORIZON_Y:] = (70, 120, 80)                       # vegetation
        noise = (self.np_rng.random((SCENE_H - HORIZON_Y, SCENE_W)) * 14).astype(np.int16)
        noise = cv2.GaussianBlur(noise.astype(np.float32), (5, 5), 0).astype(np.int16)
        img[HORIZON_Y:, :, 1] = np.clip(img[HORIZON_Y:, :, 1].astype(np.int16) + noise - 7, 0, 255)
        # asphalt (includes the shoulder on the right)
        xl0, _ = project(-LANE_W - 0.4, Z_MAX)
        xr0, _ = project(SHOULDER_X, Z_MAX)
        xl1, yb = project(-LANE_W - 0.4, Z_MIN)
        xr1, _ = project(SHOULDER_X, Z_MIN)
        pts = np.array([[xl0, HORIZON_Y + 1], [xr0, HORIZON_Y + 1], [xr1, yb], [xl1, yb]], np.int32)
        cv2.fillPoly(img, [pts], (78, 78, 82))
        # central median (left)
        med = np.array([[0, yb], [xl1, yb], [xl0, HORIZON_Y + 1], [0, HORIZON_Y + 30]], np.int32)
        cv2.fillPoly(img, [med], (60, 105, 70))
        # solid edge lines
        for x in (-LANE_W, LANE_W):
            p0, p1 = project(x, Z_MAX), project(x, Z_MIN)
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (235, 235, 235), 2,
                     cv2.LINE_AA)
        # dashed center line
        z = Z_MIN
        while z < Z_MAX:
            a, b = project(0, z), project(0, min(z + 4, Z_MAX))
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (235, 235, 235),
                     max(1, int(14 / z * 3)), cv2.LINE_AA)
            z += 12
        # --- area covered by each sensor, drawn on the asphalt and as "beams" coming out of the gantry
        overlay = img.copy()
        for lx in LANES_X:
            # RFID antenna: lobe covering 0 to 16 m before the gantry
            q = [project(lx - 1.7, Z_RFID[1]), project(lx + 1.7, Z_RFID[1]),
                 project(lx + 1.7, Z_RFID[0]), project(lx - 1.7, Z_RFID[0])]
            cv2.fillPoly(overlay, [np.array(q, np.int32)], COLOR_RFID)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        beams = img.copy()
        for lx in LANES_X:
            ant = project(lx, Z_GANTRY, BEAM_H - 0.5)
            q = [ant, project(lx - 1.7, Z_RFID[1]), project(lx + 1.7, Z_RFID[1])]
            cv2.fillPoly(beams, [np.array(q, np.int32)], COLOR_RFID)
            cam = project(lx - 0.7, Z_GANTRY, BEAM_H - 0.4)
            q = [cam, project(lx - 1.6, Z_TRIGGER + 1.2), project(lx + 1.6, Z_TRIGGER + 1.2),
                 project(lx + 1.6, Z_TRIGGER - 1.2), project(lx - 1.6, Z_TRIGGER - 1.2)]
            cv2.fillPoly(beams, [cv2.convexHull(np.array(q, np.int32))], COLOR_ANPR)
        cv2.addWeighted(beams, 0.16, img, 0.84, 0, img)
        # ANPR camera trigger line (10 m before the gantry)
        a, b = project(-LANE_W, Z_TRIGGER), project(LANE_W, Z_TRIGGER)
        cv2.line(img, ip(a), ip(b), COLOR_ANPR, 3, cv2.LINE_AA)
        # laser line, exactly under the gantry
        a, b = project(-LANE_W, Z_GANTRY), project(LANE_W, Z_GANTRY)
        cv2.line(img, ip(a), ip(b), COLOR_LASER, 3, cv2.LINE_AA)
        return img

    def _gantry_layer(self):
        """Gantry structure with its sensors, as a layer (image + mask) drawn in front of the
        vehicles that have not passed yet and behind the ones that already have."""
        img = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        z = Z_GANTRY
        steel, steel_dark = (150, 150, 155), (95, 95, 100)
        # posts
        for x in POST_X:
            b0, t0 = project(x - 0.25, z, 0), project(x + 0.25, z, BEAM_H + 0.6)
            cv2.rectangle(img, ip(t0), ip(b0), steel, -1)
            cv2.rectangle(img, ip(t0), ip(b0), steel_dark, 1)
            footing = (project(x - 0.6, z, 0), project(x + 0.6, z, 0.4))
            cv2.rectangle(img, ip((footing[0][0], footing[1][1])), ip((footing[1][0], footing[0][1])),
                          (120, 120, 125), -1)
        # truss beam
        xa, xb = POST_X[0] - 0.25, POST_X[1] + 0.25
        beam_top, base = BEAM_H + 0.6, BEAM_H
        cv2.line(img, ip(project(xa, z, beam_top)), ip(project(xb, z, beam_top)), steel, 3)
        cv2.line(img, ip(project(xa, z, base)), ip(project(xb, z, base)), steel, 3)
        xs = np.arange(xa, xb, 0.6)
        for i, x in enumerate(xs[:-1]):
            h1, h2 = (base, beam_top) if i % 2 == 0 else (beam_top, base)
            cv2.line(img, ip(project(x, z, h1)), ip(project(xs[i + 1], z, h2)), steel_dark, 1,
                     cv2.LINE_AA)
        # sensors hanging from the beam, one set per lane
        for lx in LANES_X:
            # ANPR camera (body + lens) and infrared illuminator
            c0, c1 = project(lx - 1.0, z, base), project(lx - 0.4, z, base - 0.35)
            cv2.rectangle(img, ip(c0), ip(c1), (40, 40, 45), -1)
            cv2.rectangle(img, ip(c0), ip(c1), COLOR_ANPR, 2)
            cv2.circle(img, ip(project(lx - 0.7, z, base - 0.18)), 3, (200, 200, 210), -1)
            # RFID antenna (flat panel)
            a0, a1 = project(lx - 0.25, z, base), project(lx + 0.25, z, base - 0.55)
            cv2.rectangle(img, ip(a0), ip(a1), (235, 235, 235), -1)
            cv2.rectangle(img, ip(a0), ip(a1), COLOR_RFID, 2)
            # laser scanner
            l0, l1 = project(lx + 0.45, z, base), project(lx + 0.85, z, base - 0.25)
            cv2.rectangle(img, ip(l0), ip(l1), (30, 30, 30), -1)
            cv2.rectangle(img, ip(l0), ip(l1), COLOR_LASER, 2)
        # context camera on top of the shoulder post
        k0, k1 = project(POST_X[1] - 0.9, z, BEAM_H + 0.9), project(POST_X[1] - 0.2, z, BEAM_H + 0.5)
        cv2.rectangle(img, ip(k0), ip(k1), (40, 40, 45), -1)
        cv2.rectangle(img, ip(k0), ip(k1), COLOR_CTX, 2)
        mask = img.any(axis=2)
        return img, mask

    def sensor_positions(self) -> dict:
        """Pixel position of each sensor (for labels and activity lights)."""
        z, base = Z_GANTRY, BEAM_H
        pos = {}
        for i, lx in enumerate(LANES_X):
            pos[("anpr", i)] = ip(project(lx - 0.7, z, base - 0.18))
            pos[("rfid", i)] = ip(project(lx, z, base - 0.27))
            pos[("laser", i)] = ip(project(lx + 0.65, z, base - 0.12))
        pos[("ctx", 0)] = ip(project(POST_X[1] - 0.55, z, BEAM_H + 0.7))
        return pos

    def draw_activity(self, img):
        """Lights up the sensor that just read: IR flash, antenna and laser."""
        pos = self.sensor_positions()
        for (sensor, lane), t0 in self.activity.items():
            dt = self.t - t0
            if dt < 0 or dt > 0.45:
                continue
            strength = 1 - dt / 0.45
            x, y = pos[(sensor, lane)]
            color = {"anpr": (255, 255, 255), "rfid": (80, 230, 80), "laser": COLOR_LASER}[sensor]
            overlay = img.copy()
            cv2.circle(overlay, (x, y), int(8 + 14 * strength), color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.6 * strength, img, 1 - 0.6 * strength, 0, img)
            if sensor == "laser":
                lx = LANES_X[lane]
                a, b = project(lx - 1.8, Z_GANTRY), project(lx + 1.8, Z_GANTRY)
                fan = img.copy()
                cv2.fillPoly(fan, [np.array([(x, y), ip(a), ip(b)], np.int32)], COLOR_LASER)
                cv2.addWeighted(fan, 0.35 * strength, img, 1 - 0.35 * strength, 0, img)

    def draw_vehicle(self, img, v: Vehicle):
        x1, y1, x2, y2 = v.box()
        w, h = x2 - x1, y2 - y1
        if w < 3:
            return
        shadow = np.array([[x1 - w * .05, y2], [x2 + w * .05, y2],
                           [x2, y2 - h * .06], [x1, y2 - h * .06]], np.int32)
        cv2.fillPoly(img, [shadow], (40, 40, 40))
        heavy = v.kind in ("truck_2axle", "truck_5axle", "bus")
        cv2.rectangle(img, (x1, y1), (x2, y2), v.color, -1)
        # windshield
        top = y1 + int(h * (0.08 if heavy else 0.12))
        bot = y1 + int(h * (0.42 if heavy else 0.48))
        cv2.rectangle(img, (x1 + int(w * .1), top), (x2 - int(w * .1), bot), (70, 55, 45), -1)
        # headlights
        fy = y1 + int(h * .72)
        r = max(1, int(w * .06))
        cv2.circle(img, (x1 + int(w * .15), fy), r, (210, 240, 255), -1)
        cv2.circle(img, (x2 - int(w * .15), fy), r, (210, 240, 255), -1)
        # Mercosur plate (white with a blue band)
        pw, ph = max(4, int(w * (.38 if v.kind != "motorcycle" else .7))), max(2, int(w * .12))
        px, py = (x1 + x2) // 2 - pw // 2, y1 + int(h * .80)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + max(1, ph // 4)), (150, 60, 20), -1)
        if pw > 40:
            cv2.putText(img, v.plate, (px + 2, py + ph - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        pw / 150, (20, 20, 20), 1, cv2.LINE_AA)

    def draw_annotations(self, img, draw_ops):
        for v in sorted(self.vehicles, key=lambda v: -v.z):
            x1, y1, x2, y2 = v.box()
            if x2 - x1 < 8:
                continue
            color = YELLOW
            if v.decision == "TAG" or (v.decision is None and v.tag_read):
                color = GREEN
            elif v.decision == "REVIEW":
                color = RED
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            pts = list(v.trail)
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], YELLOW, 1, cv2.LINE_AA)
            label = f"#{v.vid}"
            if v.ocr:
                label += f" {v.ocr} {v.conf*100:.0f}%"
            draw_ops.append(("box", (x1, y1), label, color))
            if v.z < 36:
                sub = f"Class {v.cat} · {v.axles} axles · {v.v*3.6:.0f} km/h"
                sub += "  · TAG read" if v.tag_read else ""
                draw_ops.append(("sub", (x1, y2 + 2), sub, color))

    def plate_crop(self, v: Vehicle) -> np.ndarray:
        """Plate crop as the infrared ANPR camera hands it to the OCR: monochrome, with noise
        and blur proportional to distance/speed."""
        pw, ph = 170, 48
        img = Image.new("L", (pw, ph), 235)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, pw - 1, 11), fill=70)                  # blue band (dark in IR)
        d.text((pw // 2 - 18, 0), "BRASIL", font=font(9, True), fill=230)
        d.text((10, 13), v.plate, font=font(28, True), fill=25)
        d.rectangle((0, 0, pw - 1, ph - 1), outline=40, width=2)
        a = np.array(img, np.float32)
        blur = 1 + int((v.v * 3.6) / 60)
        a = cv2.GaussianBlur(a, (blur * 2 + 1, 1), 0)           # motion blur
        a += self.np_rng.normal(0, 10 + 25 * (1 - (v.conf or 1)), a.shape)
        a = np.clip(a, 0, 255).astype(np.uint8)
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

    def sensor_labels(self, d: ImageDraw.ImageDraw):
        """Numbered badges on the gantry sensors + legend."""
        pos = self.sensor_positions()
        marks = [("anpr", "1", COLOR_ANPR), ("rfid", "2", COLOR_RFID), ("laser", "3", COLOR_LASER)]
        for lane in (0, 1):
            for k, num, color in marks:
                x, y = pos[(k, lane)]
                self._badge(d, x, y - 26 - (int(num) - 1) % 2 * 0, num, color)
        x, y = pos[("ctx", 0)]
        self._badge(d, x, y - 24, "4", COLOR_CTX)
        # legend
        lx, ly = 12, 8
        d.rounded_rectangle((lx - 4, ly - 4, lx + 600, ly + 40), 6, fill=(20, 24, 28))
        legend = [("1", "ANPR camera + IR flash", COLOR_ANPR), ("2", "RFID antenna", COLOR_RFID),
                   ("3", "3D laser (axles)", COLOR_LASER), ("4", "Context camera", COLOR_CTX)]
        cx = lx
        for num, name, color in legend:
            self._badge(d, cx + 8, ly + 1, num, color, center=False)
            d.text((cx + 22, ly + 1), name, font=F_SM, fill=(235, 235, 235))
            cx += 22 + int(d.textlength(name, font=F_SM)) + 16
        d.text((lx, ly + 20), "Box:  green = by tag   yellow = by plate   red = human review",
               font=F_SM, fill=(200, 200, 200))
        # labels for the lines on the asphalt
        for z, txt, color in [(Z_TRIGGER, "camera trigger (10 m before)", COLOR_ANPR),
                            (Z_GANTRY, "laser under the gantry", COLOR_LASER)]:
            x, y = project(LANE_W + 0.2, z)
            rgb = (color[2], color[1], color[0])
            bb = d.textbbox((x + 6, y - 9), txt, font=F_SM)
            d.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=(20, 24, 28))
            d.text((x + 6, y - 9), txt, font=F_SM, fill=rgb)
        x, y = project(-LANE_W - 0.3, (Z_RFID[0] + Z_RFID[1]) / 2)
        txt = "RFID zone (0-16 m)"
        bb = d.textbbox((0, 0), txt, font=F_SM)
        d.rectangle((x - bb[2] - 10, y - 10, x - 2, y + 8), fill=(20, 24, 28))
        d.text((x - bb[2] - 6, y - 9), txt, font=F_SM, fill=(COLOR_RFID[2], COLOR_RFID[1], COLOR_RFID[0]))

    def side_view_texts(self, d):
        x0, y0, w, h = 12, SCENE_H + 8, SCENE_W - 24, 132
        ground = y0 + h - 26
        d_min, d_max = -60.0, 15.0
        px = lambda dist: int(x0 + 30 + (dist - d_min) / (d_max - d_min) * (w - 60))
        d.text((x0 + 8, y0 + 4), "Side view (diagram) · direction of traffic  >>", font=F_BOX,
               fill=(40, 40, 40))
        for dist in range(-60, 16, 10):
            lbl = "gantry" if dist == 0 else f"{dist:+d} m"
            bb = d.textbbox((0, 0), lbl, font=F_SM)
            d.text((px(dist) - bb[2] // 2, ground + 6), lbl, font=F_SM, fill=(90, 90, 90))
        notes = [(-16, -1, "2 RFID antenna: 0-16 m before", COLOR_RFID),
                 (-10, -2, "1 ANPR camera: plate at 10 m", COLOR_ANPR),
                 (0, -3, "3 laser: axles, under the gantry", COLOR_LASER)]
        cx = x0 + 8
        for dist, _, txt, color in notes:
            rgb = (int(color[2] * .8), int(color[1] * .8), int(color[0] * .8))
            d.rectangle((cx, y0 + 28, cx + 9, y0 + 37), fill=rgb)
            d.text((cx + 14, y0 + 24), txt, font=F_SM, fill=rgb)
            cx += 14 + int(d.textlength(txt, font=F_SM)) + 22

    @staticmethod
    def _badge(d, x, y, num, color, center=True):
        r = 8
        cx, cy = (x, y) if center else (x, y + 7)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(color[2], color[1], color[0]),
                  outline=(255, 255, 255), width=1)
        d.text((cx - 4, cy - 8), num, font=F_BOX, fill=(15, 15, 15))

    def draw_side_view(self, canvas):
        """Side diagram: where each sensor sits and which stretch of road it covers."""
        x0, y0, w, h = 12, SCENE_H + 8, SCENE_W - 24, 132
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (236, 238, 240), -1)
        ground = y0 + h - 26
        # scale: from 60 m before the gantry to 15 m after
        d_min, d_max = -60.0, 15.0
        def px(dist):   # dist < 0: before the gantry
            return int(x0 + 30 + (dist - d_min) / (d_max - d_min) * (w - 60))
        def py(height_m):
            return int(ground - height_m * 11)
        cv2.line(canvas, (x0 + 10, ground), (x0 + w - 10, ground), (90, 90, 95), 2)
        for dist in range(-60, 16, 10):
            cv2.line(canvas, (px(dist), ground), (px(dist), ground + 4), (120, 120, 120), 1)
        # zones on the ground
        rf0, rf1 = Z_GANTRY - Z_RFID[1], Z_GANTRY - Z_RFID[0]      # -16 .. 0
        cv2.rectangle(canvas, (px(rf0), ground - 3), (px(rf1), ground + 3), COLOR_RFID, -1)
        trig = Z_GANTRY - Z_TRIGGER                                  # -10
        gx = px(0)
        beam_top = py(BEAM_H)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.array([(gx, beam_top + 8), (px(rf0), ground), (px(rf1), ground)])],
                     COLOR_RFID)
        cv2.fillPoly(overlay, [np.array([(gx - 4, beam_top + 6), (px(trig - 1.5), ground),
                                       (px(trig + 1.5), ground)])], COLOR_ANPR)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.line(canvas, (gx + 3, beam_top + 6), (gx + 3, ground), COLOR_LASER, 2)
        # gantry
        cv2.rectangle(canvas, (gx - 3, beam_top - 6), (gx + 3, ground), (130, 130, 135), -1)
        cv2.rectangle(canvas, (gx - 10, beam_top - 8), (gx + 10, beam_top), (130, 130, 135), -1)
        cv2.circle(canvas, (gx - 6, beam_top + 5), 4, COLOR_ANPR, -1)
        cv2.circle(canvas, (gx, beam_top + 8), 4, COLOR_RFID, -1)
        cv2.circle(canvas, (gx + 6, beam_top + 5), 4, COLOR_LASER, -1)
        # vehicles (side view, lane 0 drawn over lane 1, slightly offset)
        for v in self.vehicles:
            dist = Z_GANTRY - v.z
            if not (d_min <= dist <= d_max):
                continue
            length = {"car": 4.5, "suv": 4.8, "truck_2axle": 8, "truck_5axle": 18, "bus": 13,
                    "motorcycle": 2}[v.kind]
            height_m = v.dims[1]
            x_front = px(dist)
            x_tail = px(dist - length)
            color = GREEN if v.decision == "TAG" or v.tag_read else (
                RED if v.decision == "REVIEW" else YELLOW)
            offset = 0 if v.lane == 0 else -3
            cv2.rectangle(canvas, (x_tail, py(height_m) + offset), (x_front, ground - 2 + offset), v.color, -1)
            cv2.rectangle(canvas, (x_tail, py(height_m) + offset), (x_front, ground - 2 + offset), color, 2)

    def render(self, f: int) -> np.ndarray:
        canvas = np.full((H, W, 3), 248, np.uint8)
        scene = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        self.draw_road(scene)
        if self.gantry_layer is None:
            self.gantry_layer = self._gantry_layer()
        # paint order: vehicles before the gantry -> gantry -> vehicles that already passed
        order = sorted(self.vehicles, key=lambda v: -v.z)
        for v in order:
            if v.z > Z_GANTRY:
                self.draw_vehicle(scene, v)
        scene[self.gantry_layer[1]] = self.gantry_layer[0][self.gantry_layer[1]]
        for v in order:
            if v.z <= Z_GANTRY:
                self.draw_vehicle(scene, v)
        self.draw_activity(scene)
        clean = scene.copy()
        ops: list = []
        self.draw_annotations(scene, ops)
        canvas[:SCENE_H, :SCENE_W] = scene

        # --- side panel: motion heat map over the grayscale scene
        pw = W - PANEL_X - 12
        heat = cv2.GaussianBlur(cv2.resize(self.heat, (pw, 150)), (0, 0), 4)
        heat = np.clip(heat / 12.0 * 255, 0, 255).astype(np.uint8)
        heat_img = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        background = cv2.cvtColor(cv2.cvtColor(cv2.resize(clean, (pw, 150)), cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR)
        background = cv2.addWeighted(background, 0.35, np.full_like(background, (90, 30, 30)), 0.65, 0)
        alpha = (heat.astype(np.float32) / 255)[..., None] ** 0.6
        canvas[0:150, PANEL_X:W - 12] = (heat_img * alpha + background * (1 - alpha)).astype(np.uint8)
        # scale bar
        grad = np.tile(np.linspace(0, 255, W - PANEL_X - 12).astype(np.uint8), (6, 1))
        canvas[156:162, PANEL_X:W - 12] = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

        # --- crop of the last plate read
        if self.last_read is not None and (
                self.last_crop is None or self.last_crop[1] is not self.last_read):
            self.last_crop = (self.plate_crop(self.last_read), self.last_read)
        last = None
        if self.last_crop is not None:
            rc, last = self.last_crop
            canvas[400:448, PANEL_X:PANEL_X + 170] = rc

        # --- flow chart
        gx0, gy0, gw, gh = PANEL_X + 34, 520, W - PANEL_X - 56, 150
        cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (238, 238, 238), -1)
        for k in range(1, 4):
            yy = gy0 + gh - int(gh * k / 4)
            cv2.line(canvas, (gx0, yy), (gx0 + gw, yy), (225, 225, 225), 1)
        series = self.flow_series
        if len(series) > 1:
            xs = np.linspace(gx0, gx0 + gw * len(series) / self.n_frames, len(series))
            ys = gy0 + gh - np.clip(np.array(series) / 80, 0, 1) * gh
            pts = np.stack([xs, ys], 1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (110, 80, 40), 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), 4, (110, 80, 40), -1)

        # --- intensity timeline (footer)
        tx0, ty0, tw, th = 12, 778, SCENE_W - 24, 26
        intensity_colors = {"Low": (225, 225, 215), "Moderate": (210, 220, 200), "High": (180, 200, 240)}
        seg_start = 0
        blocks = []
        for i in range(1, len(self.intensities) + 1):
            if i == len(self.intensities) or self.intensities[i] != self.intensities[seg_start]:
                a = tx0 + int(tw * seg_start / self.n_frames)
                b = tx0 + int(tw * i / self.n_frames)
                cv2.rectangle(canvas, (a, ty0), (b, ty0 + th), intensity_colors[self.intensities[seg_start]], -1)
                cv2.line(canvas, (b, ty0), (b, ty0 + th), WHITE, 2)
                blocks.append((a, b, self.intensities[seg_start]))
                seg_start = i
        cv2.rectangle(canvas, (tx0 + int(tw * len(self.intensities) / self.n_frames), ty0),
                      (tx0 + tw, ty0 + th), (236, 236, 236), -1)
        xm = tx0 + int(tw * f / self.n_frames)
        cv2.line(canvas, (xm, ty0 - 4), (xm, ty0 + th + 4), (60, 50, 40), 2)

        self.draw_side_view(canvas)

        # ---------------------------------------------------- text (PIL)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for kind, (x, y), txt, color in ops:
            fnt = F_BOX if kind == "box" else F_SM
            rgb = (color[2], color[1], color[0])
            if kind == "box":
                y = max(0, y - 17)
            bb = d.textbbox((x, y), txt, font=fnt)
            d.rectangle((bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1), fill=(20, 20, 20))
            d.text((x, y), txt, font=fnt, fill=rgb)
        t = f / FPS
        clock = f"00:{int(t):02d}"
        bb = d.textbbox((0, 0), clock, font=F_XL)
        d.rounded_rectangle((W - 12 - (bb[2] + 24), 14, W - 22, 60), 8, fill=(25, 25, 30))
        d.text((W - 22 - bb[2] - 12, 18), clock, font=F_XL, fill=(255, 255, 255))
        d.text((PANEL_X, 166), "0 km/h", font=F_SM, fill=(90, 90, 90))
        d.text((W - 70, 166), "120 km/h", font=F_SM, fill=(90, 90, 90))
        d.text((14, SCENE_H - 22), "SYNTHETIC SCENE · Brazilian gantry (2 lanes + shoulder)",
               font=F_SM, fill=(255, 255, 255))
        self.sensor_labels(d)
        self.side_view_texts(d)

        intensity = self.intensities[-1]
        d.text((PANEL_X, 190), intensity, font=font(26, True), fill=(30, 40, 60))
        d.text((PANEL_X, 224), "flow over the last 10 s", font=F_SM, fill=(110, 110, 110))
        d.line((PANEL_X, 244, W - 12, 244), fill=(225, 225, 225))
        col2 = PANEL_X + 180
        d.text((PANEL_X, 252), "Vehicles in frame", font=F_SM, fill=(90, 90, 90))
        d.text((col2, 252), "Flow (veh/min)", font=F_SM, fill=(90, 90, 90))
        visible = sum(1 for v in self.vehicles if v.z < Z_MAX - 5)
        d.text((PANEL_X, 270), f"{visible}", font=font(26, True), fill=(20, 20, 20))
        d.text((col2, 270), f"{self.flow_series[-1]:.0f}", font=font(26, True), fill=(20, 20, 20))
        d.text((PANEL_X, 310), "Data generated", font=F_SM, fill=(90, 90, 90))
        d.text((col2, 310), "Rate (MB/min)", font=F_SM, fill=(90, 90, 90))
        d.text((PANEL_X, 326), f"{self.bytes_total/1e6:,.1f} MB", font=font(22, True),
               fill=(20, 20, 20))
        d.text((col2, 326), f"{self.mb_series[-1]:.1f}", font=font(22, True), fill=(20, 20, 20))
        n = len(self.events)
        if n:
            tag = sum(e["decision"] == "TAG" for e in self.events) / n
            ocr = sum(e["decision"] == "OCR" for e in self.events) / n
            rev = 1 - tag - ocr
            d.text((PANEL_X, 360), f"Passes: {n}   TAG {tag:.0%} · OCR {ocr:.0%} · review {rev:.0%}",
                   font=F_SM, fill=(60, 60, 60))
        d.text((PANEL_X, 382), "Last plate crop (IR)", font=F_SM, fill=(90, 90, 90))
        if last is not None:
            d.text((PANEL_X + 180, 402), f"{last.ocr}", font=font(18, True), fill=(20, 20, 20))
            d.text((PANEL_X + 180, 426), f"conf. {last.conf*100:.0f}%", font=F_SM,
                   fill=(90, 90, 90))
        d.text((PANEL_X, 462), "Last event sent", font=F_SM, fill=(90, 90, 90))
        if self.events:
            e = self.events[-1]
            d.text((PANEL_X, 478), f"#{e['track_id']} {e['ocr_plate']} Class {e['antt_class']} "
                   f"{e['decision']} {e['event_bytes']/1e3:.0f} KB", font=F_MONO, fill=(40, 40, 40))
        # chart axes
        for k, val in enumerate([0, 20, 40, 60, 80]):
            d.text((PANEL_X, gy0 + gh - int(gh * k / 4) - 7), f"{val}", font=F_SM,
                   fill=(110, 110, 110))
        d.text((PANEL_X, gy0 - 18), "vehicles/min", font=F_SM, fill=(90, 90, 90))
        d.text((gx0, gy0 + gh + 4), "0 s", font=F_SM, fill=(110, 110, 110))
        d.text((gx0 + gw - 26, gy0 + gh + 4), f"{self.n_frames // FPS} s", font=F_SM,
               fill=(110, 110, 110))
        # sensor status
        d.text((PANEL_X, 712), "Reads per sensor in this scene", font=F_SM, fill=(90, 90, 90))
        items = [("ANPR camera", "anpr", COLOR_ANPR), ("RFID antenna", "rfid", COLOR_RFID),
                 ("3D laser", "laser", COLOR_LASER)]
        for i, (name, k, color) in enumerate(items):
            y = 732 + i * 24
            active = any(s == k and 0 <= self.t - t0 < 0.45 for (s, _), t0 in self.activity.items())
            rgb = (color[2], color[1], color[0])
            d.rectangle((PANEL_X, y + 2, PANEL_X + 12, y + 14), fill=rgb if active else None,
                        outline=rgb, width=2)
            d.text((PANEL_X + 20, y), name, font=F_MD, fill=(40, 40, 40))
            d.text((PANEL_X + 160, y), f"{self.sensor_count[k]:>3d} reads", font=F_MONO,
                   fill=(60, 60, 60))
            if active:
                d.text((PANEL_X + 270, y), "reading", font=F_BOX, fill=(30, 150, 60))
        # footer
        d.text((12, 690), "Pipeline:  camera  >  detection  >  tracking  >  trigger  >  plate OCR  >  RFID read  >  fusion  >  class  >  event", font=F_MD, fill=(40, 40, 40))
        d.text((12, 714), "Each pass stores ~965 KB of evidence (2 ANPR photos, crops, "
               "context photo, laser profile, JSON) and sends ~2 KB to the back office.",
               font=F_SM, fill=(90, 90, 90))
        d.text((12, 750), "Estimated traffic intensity", font=F_LG, fill=(30, 30, 30))
        d.text((SCENE_W - 110, 752), f"{t:.1f} / {self.n_frames / FPS:.0f} s", font=F_SM,
               fill=(90, 90, 90))
        for a, b, name in blocks:
            if b - a > 70:
                d.text((a + 8, ty0 + 5), name, font=F_SM, fill=(40, 40, 40))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    sim = Simulator(args.seconds, args.seed)
    # warm the scene up for 6 s so it starts with traffic on the road
    warmup = 6 * FPS
    sim_warm = sim
    for f in range(warmup):
        sim_warm.step(0)
    sim.events.clear()
    sim.bytes_total = 0.0
    sim.flow_series.clear()
    sim.mb_series.clear()
    sim.intensities.clear()
    sim.sensor_count = {k: 0 for k in sim.sensor_count}
    for v in sim.vehicles:
        v.event_sent = v.event_sent or v.z < Z_GANTRY

    frames, density = [], []
    for f in range(sim.n_frames):
        sim.step(f)
        frames.append(sim.render(f))
        density.append(sum(1 for v in sim.vehicles if v.z < 45) + 0.1 * len(sim.vehicles))

    import imageio.v2 as imageio
    mp4 = FIG / "gantry_simulation.mp4"
    with imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8) as w:
        for fr in frames:
            w.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    # highlight frame: the one with the most nearby vehicles after the first third
    third = len(frames) // 3
    highlight = third + int(np.argmax(density[third:]))
    cv2.imwrite(str(FIG / "gantry_simulation_frame.png"), frames[highlight])
    if not args.no_gif:
        gif_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(fr, (800, round(800 * H / W)),
                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                      for fr in frames[::2]]
        gif_frames[0].save(FIG / "gantry_simulation.gif", save_all=True,
                           append_images=gif_frames[1:], duration=int(2000 / FPS), loop=0,
                           optimize=True)

    ev = pd.DataFrame(sim.events)
    ev.to_csv(TAB / "simulated_events.csv", index=False)
    print(f"Simulated passes: {len(ev)} in {args.seconds} s")
    if len(ev):
        print(ev.decision.value_counts(normalize=True).round(3).to_string())
        print(f"OCR correct: {ev.ocr_correct.mean():.1%} | "
              f"OCR errors caught by the tag: "
              f"{((~ev.ocr_correct) & (ev.decision == 'TAG')).sum()}")
        print(f"Data generated: {ev.event_bytes.sum()/1e6:.1f} MB "
              f"({ev.event_bytes.sum()/1e6/args.seconds*60:.1f} MB/min)")
    print(f"Video: {mp4}")


if __name__ == "__main__":
    main()
