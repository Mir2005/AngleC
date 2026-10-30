import cv2
import numpy as np
import math
import csv

try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from matplotlib import font_manager as _font_manager
    MATPLOTLIB_FONT_MANAGER_AVAILABLE = True
except ImportError:
    _font_manager = None
    MATPLOTLIB_FONT_MANAGER_AVAILABLE = False


# ================================================================
#                         MECHATRIX AngleC
#            Real-Time Sessile-Drop Contact Angle Measurement
# ================================================================
# Version      : v1.0
#
# Concept and system design: Mir Md Minhaajuddin
# Coding assistance: ChatGPT and Claude
#
# Software:
#   AngleC — Contact Angle Measurement & Analysis Software
#
# Hardware:
#   MECHATRIX Model Alpha
#
# Copyright © 2026 Mir Md Minhaajuddin
# ================================================================
#
# FINAL WORKFLOW
# ------------------------------------------------------------
# 1. Start camera and position the droplet.
# 2. Press C:
#       - capture the current frame
#       - save RAW image with the user name
#       - freeze the camera
# 3. On the frozen frame:
#       - click two points to define the baseline
#       - click 5-6 (or more) points on the OUTER droplet surface
#       - the program fits a circle and an ellipse to those
#         manually selected surface points
# 4. The baseline is independent of both fits. It only intersects
#    the fitted curves. Contact angle is obtained from the tangent
#    of each fitted curve at its baseline intersection.
# 5. Press E:
#       - save analyzed PNG
#       - save CSV-formatted TXT file
# 6. Press R:
#       - clear the current measurement
#       - ask for a NEW name
#       - return to live camera mode
# 7. Press ESC at any time to stop the program. Q is also retained as a quit key.
#
# Geometry colors:
#   Yellow = user baseline
#   Red    = fitted circle
#   Green  = fitted ellipse
#   Blue   = selected droplet surface points
#   Deep blue = tangents / angle arcs
#   White  = text / UI
# ============================================================


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30


# ============================================================
# MANUAL TRACE SETTINGS
# ============================================================

MIN_SURFACE_POINTS = 5
RECOMMENDED_SURFACE_POINTS = 6
MAX_SURFACE_POINTS = 30

# Magnifier/loupe.
LOUPE_SIZE = 150
LOUPE_ZOOM = 5
LOUPE_MARGIN = 12

# Export band: never exceed 20% of the complete image.
TEXT_BAND_MAX_FRACTION = 0.20

# Geometry drawing.
CURVE_SAMPLES = 720
CURVE_THICKNESS = 1
TANGENT_LENGTH = 48
ANGLE_ARC_RADIUS = 24
POINT_RADIUS = 4
BASELINE_THICKNESS = 1


# ============================================================
# COLORS -- OpenCV BGR
# ============================================================

YELLOW = (0, 220, 255)       # baseline
RED = (0, 0, 230)             # fitted circle
GREEN = (0, 190, 0)           # fitted ellipse
BLUE = (230, 90, 0)           # selected surface points
WHITE = (255, 255, 255)       # text / loupe border
ANGLE_COLOR = YELLOW           # same yellow as the baseline for tangents + angle arcs
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (70, 70, 70)


# ============================================================
# GLOBAL STATE
# ============================================================

IMAGE_BASENAME = "contact_angle"

frozen = False
frozen_frame = None

# Baseline: two user-selected endpoints.
baseline_x1 = None
baseline_y1 = None
baseline_x2 = None
baseline_y2 = None

# Surface points selected by the user.
surface_points = []

# Current fits.
last_ellipse = None
last_circle = None

# Mouse position for loupe.
mouse_x = None
mouse_y = None


# ============================================================
# INPUT / NAME HANDLING
# ============================================================

def sanitize_name(raw_name):
    raw_name = raw_name.strip()

    if raw_name.lower().endswith(".png"):
        raw_name = raw_name[:-4]

    # Keep a simple filename-safe name without using os/pathlib.
    safe_name = "".join(
        c for c in raw_name
        if c.isalnum() or c in ("_", "-")
    )

    return safe_name if safe_name else "contact_angle"


def ask_for_image_name():
    while True:
        print("==============================================")
        print("_______________MECHATRIX AngleC________________")
        print("Real-Time Sessile-Drop Contact Angle Measurement")
        print("Designed & Developed by: Mir Md Minhaajuddin")
        print("==============================================")
        raw_name = input(
            "Enter a base name for this measurement (e.g. W32): "
        ).strip()

        name = sanitize_name(raw_name)

        if name:
            return name


IMAGE_BASENAME = ask_for_image_name()


# ============================================================
# FONT HELPERS
# ============================================================

def find_times_font(bold=False, size=24):
    """
    Try Times New Roman first, then free TeX Gyre Termes.
    No OS-specific font-path lookup is used.
    """

    if not PIL_AVAILABLE or not MATPLOTLIB_FONT_MANAGER_AVAILABLE:
        return None

    weight = "bold" if bold else "normal"

    for family in ("Times New Roman", "TeX Gyre Termes"):
        try:
            props = _font_manager.FontProperties(
                family=family,
                weight=weight
            )
            path = _font_manager.findfont(
                props,
                fallback_to_default=False
            )
            if path:
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue

    return None


def put_text_pil(img_bgr, text, xy, font, fill=(0, 0, 0)):
    if PIL_AVAILABLE and font is not None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(xy, text, font=font, fill=fill)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    scale = 0.6
    thickness = 1

    cv2.putText(
        img_bgr,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        fill,
        thickness,
        cv2.LINE_AA
    )
    return img_bgr


def text_bbox_pil(text, font):
    if PIL_AVAILABLE and font is not None:
        dummy = Image.new("RGB", (10, 10), "white")
        draw = ImageDraw.Draw(dummy)
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]

    return cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        1
    )[0]


# ============================================================
# BASELINE GEOMETRY
# ============================================================

def baseline_ready():
    return (
        baseline_x1 is not None
        and baseline_y1 is not None
        and baseline_x2 is not None
        and baseline_y2 is not None
    )


def baseline_line_parameters():
    """Return p0 and unit direction for the user baseline."""

    if not baseline_ready():
        return None

    p0 = np.array(
        [float(baseline_x1), float(baseline_y1)],
        dtype=float
    )

    p1 = np.array(
        [float(baseline_x2), float(baseline_y2)],
        dtype=float
    )

    d = p1 - p0
    length = np.linalg.norm(d)

    if length < 1e-9:
        return None

    d /= length

    return p0, d, length


def point_to_baseline_coordinates(point):
    """Return longitudinal coordinate s and signed normal coordinate n."""

    params = baseline_line_parameters()
    if params is None:
        return None

    p0, d, _ = params
    p = np.asarray(point, dtype=float)

    s = float(np.dot(p - p0, d))
    nvec = np.array([-d[1], d[0]])
    n = float(np.dot(p - p0, nvec))

    return s, n


def baseline_x_at_y(y):
    """Return baseline x at a given y for a non-vertical baseline."""

    if not baseline_ready():
        return None

    x1, y1 = float(baseline_x1), float(baseline_y1)
    x2, y2 = float(baseline_x2), float(baseline_y2)

    dy = y2 - y1

    if abs(dy) < 1e-12:
        return None

    return x1 + (float(y) - y1) * (x2 - x1) / dy


def baseline_intersection_with_horizontal_curve_point(point):
    """
    Not used for fitting. Kept as a small geometry helper for
    near-horizontal baselines.
    """

    return point


def draw_baseline(image, preview=False):
    if baseline_x1 is None or baseline_y1 is None:
        return image

    p1 = (int(round(baseline_x1)), int(round(baseline_y1)))

    if baseline_x2 is None or baseline_y2 is None:
        cv2.circle(
            image,
            p1,
            POINT_RADIUS,
            YELLOW,
            1,
            cv2.LINE_AA
        )
        return image

    p2 = (int(round(baseline_x2)), int(round(baseline_y2)))

    cv2.line(
        image,
        p1,
        p2,
        YELLOW,
        BASELINE_THICKNESS,
        cv2.LINE_AA
    )

    cv2.circle(image, p1, 3, YELLOW, 1, cv2.LINE_AA)
    cv2.circle(image, p2, 3, YELLOW, 1, cv2.LINE_AA)

    return image


# ============================================================
# SURFACE POINTS
# ============================================================

def valid_surface_points():
    if len(surface_points) < MIN_SURFACE_POINTS:
        return None

    return np.asarray(surface_points, dtype=np.float64)


def draw_surface_points(image):
    for i, point in enumerate(surface_points, start=1):
        x, y = int(round(point[0])), int(round(point[1]))

        # Thin surface-point marker
        cv2.circle(
            image,
            (x, y),
            3,
            BLUE,
            1,
            cv2.LINE_AA
        )

        # Small grey point number — no shadow/outline
        label = str(i)
        tx, ty = x + 7, y - 7

        cv2.putText(
            image,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            GRAY,
            1,
            cv2.LINE_AA
        )

    return image


# ============================================================
# FITTING HELPERS
# ============================================================

def choose_upward_tangent(dx, dy):
    tangent = np.array([float(dx), float(dy)], dtype=float)

    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return None

    tangent /= norm

    # OpenCV image coordinates: upward = negative y.
    if tangent[1] > 0:
        tangent = -tangent

    return tangent


def angle_to_baseline(tangent, baseline_direction, liquid_side_sign):
    """
    Return the acute/obtuse liquid-side angle between the upward
    fitted-curve tangent and the baseline ray pointing inward from
    the relevant contact point.

    liquid_side_sign:
      +1 = baseline direction toward the right
      -1 = baseline direction toward the left
    """

    tangent = np.asarray(tangent, dtype=float)
    tangent /= max(np.linalg.norm(tangent), 1e-12)

    d = np.asarray(baseline_direction, dtype=float)
    d /= max(np.linalg.norm(d), 1e-12)

    ray = d * float(liquid_side_sign)

    cosine = float(np.dot(tangent, ray))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return math.degrees(math.acos(cosine))


def line_curve_intersections_from_parametric(
    point_at_t,
    tangent_at_t,
    baseline_p0,
    baseline_d,
    samples=2000
):
    """
    Numerically find intersections between a parametric curve and
    the user-defined baseline.

    For each curve sample, signed cross-product distance to the
    baseline is evaluated. Sign changes locate crossings. The
    result is refined by linear interpolation in parameter space.
    """

    ts = np.linspace(0.0, 2.0 * math.pi, samples + 1)
    values = []
    points = []

    def signed_value(p):
        # Cross(d, p-p0) in 2D.
        q = np.asarray(p, dtype=float) - baseline_p0
        return float(
            baseline_d[0] * q[1]
            - baseline_d[1] * q[0]
        )

    for t in ts:
        p = point_at_t(t)
        points.append(np.asarray(p, dtype=float))
        values.append(signed_value(p))

    values = np.asarray(values, dtype=float)
    points = np.asarray(points, dtype=float)

    roots = []

    for i in range(len(ts) - 1):
        f1 = values[i]
        f2 = values[i + 1]

        if abs(f1) < 1e-8:
            roots.append(float(ts[i]))
            continue

        if f1 * f2 < 0:
            denom = f1 - f2
            alpha = f1 / denom if abs(denom) > 1e-12 else 0.5
            t_root = ts[i] + alpha * (ts[i + 1] - ts[i])
            roots.append(float(t_root))

    # Deduplicate roots.
    unique = []
    for t in roots:
        if not unique or abs(t - unique[-1]) > 1e-4:
            unique.append(t)

    intersections = []

    for t in unique:
        p = point_at_t(t)
        tangent = tangent_at_t(t)
        if tangent is None:
            continue

        # Only keep intersections close to the selected baseline segment.
        q = np.asarray(p, dtype=float) - baseline_p0
        s = float(np.dot(q, baseline_d))
        baseline_length = np.linalg.norm(
            np.asarray([baseline_x2, baseline_y2], dtype=float)
            - baseline_p0
        )

        if -0.05 * baseline_length <= s <= 1.05 * baseline_length:
            intersections.append((float(s), np.asarray(p), tangent))

    intersections.sort(key=lambda item: item[0])

    # Deduplicate spatially.
    unique_intersections = []
    for item in intersections:
        if not unique_intersections:
            unique_intersections.append(item)
        else:
            if np.linalg.norm(
                item[1] - unique_intersections[-1][1]
            ) > 1.0:
                unique_intersections.append(item)

    return unique_intersections


def select_left_right_intersections(intersections):
    if len(intersections) < 2:
        return None

    # The two extreme intersections are the left/right contact points
    # along the user-defined baseline direction.
    left = intersections[0]
    right = intersections[-1]

    if left[0] > right[0]:
        left, right = right, left

    return left, right


# ============================================================
# ELLIPSE FIT
# ============================================================

def ellipse_point(cx, cy, a, b, theta, t):
    c = math.cos(theta)
    s = math.sin(theta)
    ct = math.cos(t)
    st = math.sin(t)

    x = cx + a * ct * c - b * st * s
    y = cy + a * ct * s + b * st * c

    return np.array([x, y], dtype=float)


def ellipse_tangent(cx, cy, a, b, theta, t):
    c = math.cos(theta)
    s = math.sin(theta)
    st = math.sin(t)
    ct = math.cos(t)

    dxdt = -a * st * c - b * ct * s
    dydt = -a * st * s + b * ct * c

    return choose_upward_tangent(dxdt, dydt)


def ellipse_eccentricity(a, b):
    major = max(abs(a), abs(b))
    minor = min(abs(a), abs(b))

    if major < 1e-12:
        return 0.0

    return math.sqrt(
        max(0.0, 1.0 - (minor * minor) / (major * major))
    )


def ellipse_geometry_from_fit(cx, cy, a, b, angle_deg):
    if not baseline_ready():
        return None

    theta = math.radians(angle_deg)

    baseline_params = baseline_line_parameters()
    if baseline_params is None:
        return None

    baseline_p0, baseline_d, _ = baseline_params

    def point_at_t(t):
        return ellipse_point(cx, cy, a, b, theta, t)

    def tangent_at_t(t):
        return ellipse_tangent(cx, cy, a, b, theta, t)

    intersections = line_curve_intersections_from_parametric(
        point_at_t,
        tangent_at_t,
        baseline_p0,
        baseline_d
    )

    selected = select_left_right_intersections(intersections)
    if selected is None:
        return None

    left = selected[0]
    right = selected[1]

    left_point = left[1]
    right_point = right[1]

    left_tangent = left[2]
    right_tangent = right[2]

    left_angle = angle_to_baseline(
        left_tangent,
        baseline_d,
        +1
    )

    right_angle = angle_to_baseline(
        right_tangent,
        baseline_d,
        -1
    )

    return {
        "left_point": left_point,
        "right_point": right_point,
        "left_x": float(left_point[0]),
        "right_x": float(right_point[0]),
        "left_tangent": left_tangent,
        "right_tangent": right_tangent,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "mean_angle": (left_angle + right_angle) / 2.0,
        "ellipse_center": (float(cx), float(cy)),
        "ellipse_axes": (float(a), float(b)),
        "ellipse_angle_deg": float(angle_deg),
        "eccentricity": ellipse_eccentricity(a, b),
        "intersection_count": len(intersections),
    }


def fit_ellipse(points):
    if points is None or len(points) < MIN_SURFACE_POINTS:
        return None

    if not baseline_ready():
        return None

    try:
        (cx, cy), (width, height), angle_deg = cv2.fitEllipse(
            points.astype(np.float32)
        )
    except cv2.error:
        return None

    a = float(width) / 2.0
    b = float(height) / 2.0

    if a < 1e-6 or b < 1e-6:
        return None

    return ellipse_geometry_from_fit(
        cx,
        cy,
        a,
        b,
        angle_deg
    )


def refine_ellipse(points, initial):
    """
    Optional light geometric refinement for the selected points.
    This remains a geometric ellipse fit, not Young-Laplace/ADSA.
    """

    if not SCIPY_AVAILABLE or points is None or initial is None:
        return initial

    cx0, cy0 = initial["ellipse_center"]
    a0, b0 = initial["ellipse_axes"]
    theta0 = math.radians(initial["ellipse_angle_deg"])

    p0 = np.array(
        [cx0, cy0, a0, b0, theta0],
        dtype=float
    )

    pts = np.asarray(points, dtype=float)

    def residuals(params):
        cx, cy, a, b, theta = params

        a = max(abs(a), 1e-3)
        b = max(abs(b), 1e-3)

        xr = pts[:, 0] - cx
        yr = pts[:, 1] - cy

        c = math.cos(theta)
        s = math.sin(theta)

        u = (xr * c + yr * s) / a
        v = (-xr * s + yr * c) / b

        F = u * u + v * v - 1.0

        dFdx = (
            2.0 * u * c / a
            - 2.0 * v * s / b
        )

        dFdy = (
            2.0 * u * s / a
            + 2.0 * v * c / b
        )

        grad = np.sqrt(
            dFdx * dFdx + dFdy * dFdy
        ) + 1e-9

        return F / grad

    try:
        result = least_squares(
            residuals,
            p0,
            method="lm",
            max_nfev=200
        )

        cx, cy, a, b, theta = result.x

        refined = ellipse_geometry_from_fit(
            cx,
            cy,
            abs(a),
            abs(b),
            math.degrees(theta)
        )

        return refined if refined is not None else initial

    except Exception:
        return initial


# ============================================================
# CIRCLE FIT
# ============================================================

def fit_circle(points):
    if points is None or len(points) < 3:
        return None

    if not baseline_ready():
        return None

    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    # Algebraic least-squares initial solution.
    A = np.column_stack([
        x,
        y,
        np.ones_like(x)
    ])
    rhs = -(x * x + y * y)

    try:
        sol, _, rank, _ = np.linalg.lstsq(
            A,
            rhs,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    if rank < 3:
        return None

    D, E, F = sol

    cx = -D / 2.0
    cy = -E / 2.0

    r2 = cx * cx + cy * cy - F

    if r2 <= 0:
        return None

    radius = math.sqrt(r2)

    # Optional geometric refinement using the same manually selected points.
    if SCIPY_AVAILABLE and len(points) >= 5:
        def residual(p):
            ccx, ccy, rr = p
            rr = abs(rr)
            return np.sqrt(
                (x - ccx) ** 2
                + (y - ccy) ** 2
            ) - rr

        try:
            result = least_squares(
                residual,
                np.array([cx, cy, radius], dtype=float),
                max_nfev=100
            )
            cx, cy, radius = result.x
            radius = abs(radius)
        except Exception:
            pass

    baseline_params = baseline_line_parameters()
    if baseline_params is None:
        return None

    baseline_p0, baseline_d, baseline_length = baseline_params

    # Solve |p0 + s*d - center|^2 = r^2.
    center = np.array([cx, cy], dtype=float)
    w = baseline_p0 - center

    B = 2.0 * float(np.dot(w, baseline_d))
    C = float(np.dot(w, w) - radius * radius)
    discriminant = B * B - 4.0 * C

    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(max(0.0, discriminant))

    s1 = (-B - sqrt_disc) / 2.0
    s2 = (-B + sqrt_disc) / 2.0

    candidates = sorted([s1, s2])

    # Contact points should lie on/near the selected baseline segment.
    if (
        candidates[-1] < -0.05 * baseline_length
        or candidates[0] > 1.05 * baseline_length
    ):
        return None

    left_s = candidates[0]
    right_s = candidates[-1]

    left_point = baseline_p0 + left_s * baseline_d
    right_point = baseline_p0 + right_s * baseline_d

    def circle_tangent(point):
        radius_vector = point - center
        tangent = np.array([
            -radius_vector[1],
            radius_vector[0]
        ], dtype=float)
        return choose_upward_tangent(
            tangent[0],
            tangent[1]
        )

    left_tangent = circle_tangent(left_point)
    right_tangent = circle_tangent(right_point)

    if left_tangent is None or right_tangent is None:
        return None

    left_angle = angle_to_baseline(
        left_tangent,
        baseline_d,
        +1
    )

    right_angle = angle_to_baseline(
        right_tangent,
        baseline_d,
        -1
    )

    return {
        "left_point": left_point,
        "right_point": right_point,
        "left_x": float(left_point[0]),
        "right_x": float(right_point[0]),
        "left_tangent": left_tangent,
        "right_tangent": right_tangent,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "mean_angle": (left_angle + right_angle) / 2.0,
        "circle_center": (float(cx), float(cy)),
        "radius": float(radius),
    }


# ============================================================
# FIT ALL
# ============================================================

def calculate_fits():
    global last_ellipse, last_circle

    points = valid_surface_points()

    if points is None or not baseline_ready():
        last_ellipse = None
        last_circle = None
        return None, None

    ellipse = fit_ellipse(points)
    circle = fit_circle(points)

    last_ellipse = ellipse
    last_circle = circle

    return ellipse, circle


# ============================================================
# CURVE DRAWING
# ============================================================

def draw_full_ellipse(image, measurement, color=GREEN, thickness=2):
    if measurement is None:
        return image

    h, w = image.shape[:2]
    cx, cy = measurement["ellipse_center"]
    a, b = measurement["ellipse_axes"]
    theta = math.radians(measurement["ellipse_angle_deg"])

    pts = []

    for t in np.linspace(
        0.0,
        2.0 * math.pi,
        CURVE_SAMPLES
    ):
        p = ellipse_point(
            cx,
            cy,
            a,
            b,
            theta,
            t
        )

        x, y = int(round(p[0])), int(round(p[1]))

        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))

    if len(pts) >= 2:
        cv2.polylines(
            image,
            [np.asarray(pts, dtype=np.int32)],
            True,
            color,
            thickness,
            cv2.LINE_AA
        )

    return image


def draw_full_circle(image, measurement, color=RED, thickness=2):
    if measurement is None:
        return image

    cx, cy = measurement["circle_center"]
    r = measurement["radius"]

    cv2.ellipse(
        image,
        (int(round(cx)), int(round(cy))),
        (int(round(r)), int(round(r))),
        0,
        0,
        360,
        color,
        thickness,
        cv2.LINE_AA
    )

    return image


def draw_tangent(image, point, tangent, length=TANGENT_LENGTH):
    if point is None or tangent is None:
        return image

    p = np.asarray(point, dtype=float)
    t = np.asarray(tangent, dtype=float)
    t /= max(np.linalg.norm(t), 1e-12)

    p1 = p - t * length
    p2 = p + t * length

    h, w = image.shape[:2]

    p1 = (
        int(round(p1[0])),
        int(round(p1[1]))
    )
    p2 = (
        int(round(p2[0])),
        int(round(p2[1]))
    )

    # Clip visually by OpenCV's line drawing region naturally.
    cv2.line(
        image,
        p1,
        p2,
        ANGLE_COLOR,
        1,
        cv2.LINE_AA
    )

    return image


def draw_angle_arc(image, point, tangent, side):
    if point is None or tangent is None:
        return image

    center = np.asarray(point, dtype=float)
    t = np.asarray(tangent, dtype=float)
    t /= max(np.linalg.norm(t), 1e-12)

    baseline_params = baseline_line_parameters()
    if baseline_params is None:
        return image

    _, baseline_d, _ = baseline_params

    if side == "left":
        baseline_ray = baseline_d
    else:
        baseline_ray = -baseline_d

    a1 = math.atan2(
        baseline_ray[1],
        baseline_ray[0]
    )

    a2 = math.atan2(
        t[1],
        t[0]
    )

    # Shortest angular path.
    delta = (
        (a2 - a1 + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )

    angles = np.linspace(
        a1,
        a1 + delta,
        45
    )

    pts = []

    for a in angles:
        p = center + ANGLE_ARC_RADIUS * np.array([
            math.cos(a),
            math.sin(a)
        ])
        pts.append((int(round(p[0])), int(round(p[1]))))

    if len(pts) >= 2:
        cv2.polylines(
            image,
            [np.asarray(pts, dtype=np.int32)],
            False,
            ANGLE_COLOR,
            1,
            cv2.LINE_AA
        )

    cv2.circle(
        image,
        (int(round(center[0])), int(round(center[1]))),
        3,
        ANGLE_COLOR,
        -1,
        cv2.LINE_AA
    )

    return image


def draw_angle_text(image, measurement, prefix):
    if measurement is None:
        return image

    left = measurement["left_point"]
    right = measurement["right_point"]

    # Place compact angle labels slightly inside the droplet.
    lp = (
        int(round(left[0] + 12)),
        int(round(left[1] - 12))
    )
    rp = (
        int(round(right[0] - 75)),
        int(round(right[1] - 12))
    )

    cv2.putText(
        image,
        f"{prefix} L {measurement['left_angle']:.1f} deg",
        lp,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"{prefix} R {measurement['right_angle']:.1f} deg",
        rp,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )

    return image


def draw_measurement_overlay(image, ellipse_measurement, circle_measurement):
    """Draw the complete interactive measurement geometry.

    The display is intentionally incremental: the baseline and every
    clicked surface point are always drawn immediately. Once enough
    surface points exist, the fitted circle/ellipse and their contact
    geometry are drawn as well.
    """
    out = image.copy()

    # Baseline is independent of both curve fits.
    draw_baseline(out)

    # Always show every manually selected surface point immediately.
    draw_surface_points(out)

    # Once enough points are available, show both complete fitted curves.
    draw_full_circle(out, circle_measurement, RED, CURVE_THICKNESS)
    draw_full_ellipse(out, ellipse_measurement, GREEN, CURVE_THICKNESS)

    # Contact geometry is based on the ellipse, which is the primary
    # reported fitted-surface result.
    if ellipse_measurement is not None:
        draw_tangent(
            out,
            ellipse_measurement["left_point"],
            ellipse_measurement["left_tangent"]
        )
        draw_tangent(
            out,
            ellipse_measurement["right_point"],
            ellipse_measurement["right_tangent"]
        )

        draw_angle_arc(
            out,
            ellipse_measurement["left_point"],
            ellipse_measurement["left_tangent"],
            "left"
        )
        draw_angle_arc(
            out,
            ellipse_measurement["right_point"],
            ellipse_measurement["right_tangent"],
            "right"
        )

    return out


# ============================================================
# MAGNIFIER / LOUPE
# ============================================================

def draw_magnifier(image):
    if mouse_x is None or mouse_y is None:
        return image

    h, w = image.shape[:2]

    half = max(
        2,
        LOUPE_SIZE // (2 * LOUPE_ZOOM)
    )

    x1 = max(0, mouse_x - half)
    y1 = max(0, mouse_y - half)
    x2 = min(w, mouse_x + half + 1)
    y2 = min(h, mouse_y + half + 1)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return image

    loupe = cv2.resize(
        crop,
        (LOUPE_SIZE, LOUPE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    px = mouse_x + LOUPE_MARGIN
    py = mouse_y + LOUPE_MARGIN

    if px + LOUPE_SIZE > w:
        px = mouse_x - LOUPE_SIZE - LOUPE_MARGIN

    if py + LOUPE_SIZE > h:
        py = mouse_y - LOUPE_SIZE - LOUPE_MARGIN

    px = max(0, px)
    py = max(0, py)

    # Ensure the loupe stays inside the image.
    if px + LOUPE_SIZE > w:
        px = max(0, w - LOUPE_SIZE)
    if py + LOUPE_SIZE > h:
        py = max(0, h - LOUPE_SIZE)

    image[
        py:py + LOUPE_SIZE,
        px:px + LOUPE_SIZE
    ] = loupe

    cv2.rectangle(
        image,
        (px, py),
        (px + LOUPE_SIZE - 1, py + LOUPE_SIZE - 1),
        WHITE,
        1,
        cv2.LINE_AA
    )

    cx = px + LOUPE_SIZE // 2
    cy = py + LOUPE_SIZE // 2

    cv2.line(
        image,
        (cx - 8, cy),
        (cx + 8, cy),
        RED,
        1,
        cv2.LINE_AA
    )
    cv2.line(
        image,
        (cx, cy - 8),
        (cx, cy + 8),
        RED,
        1,
        cv2.LINE_AA
    )

    return image


# ============================================================
# RAW IMAGE WATERMARK
# ============================================================

def draw_raw_watermark(frame):
    image = frame.copy()
    h, w = image.shape[:2]

    font = find_times_font(
        bold=True,
        size=max(22, h // 22)
    )

    text = IMAGE_BASENAME
    margin = 14

    if PIL_AVAILABLE and font is not None:
        tw, th = text_bbox_pil(text, font)

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (
                max(0, w - tw - 2 * margin),
                margin
            ),
            (
                w - margin // 2,
                margin + th + margin // 2
            ),
            BLACK,
            -1
        )

        image = cv2.addWeighted(
            overlay,
            0.45,
            image,
            0.55,
            0
        )

        image = put_text_pil(
            image,
            text,
            (
                w - tw - margin,
                margin // 2
            ),
            font,
            fill=(255, 255, 255)
        )

        return image

    # OpenCV fallback.
    font_scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    cv2.putText(
        image,
        text,
        (w - tw - 12, th + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA
    )

    return image


def capture_raw(frame):
    if frame is None:
        print("Nothing to capture.")
        return

    raw = draw_raw_watermark(frame)
    filename = f"{IMAGE_BASENAME}.png"

    if cv2.imwrite(filename, raw):
        print(f"RAW image saved: {filename}")
    else:
        print(f"ERROR: Could not save {filename}")


# ============================================================
# EXPORT BAND
# ============================================================

def draw_analyzed_export(
    photo,
    ellipse_measurement,
    circle_measurement
):
    """Create the final image with a readable horizontal data panel."""
    h, w = photo.shape[:2]

    # Keep the white panel within the original 20% maximum.
    band_h = max(1, min(int(h * TEXT_BAND_MAX_FRACTION), int(h * 0.25)))
    photo_h = h - band_h

    # All geometry remains directly on the photograph.
    overlay = draw_measurement_overlay(
        photo,
        ellipse_measurement,
        circle_measurement
    )
    photo_part = overlay[:photo_h].copy()

    band = np.full((band_h, w, 3), 255, dtype=np.uint8)

    left_margin = max(10, int(w * 0.035))
    right_margin = max(10, int(w * 0.025))
    name_width = int(w * 0.18)
    stats_x = left_margin + name_width
    stats_w = w - stats_x - right_margin
    col_w = stats_w / 3.0

    name_font = find_times_font(bold=False, size=max(22, int(band_h * 0.38)))
    data_font = find_times_font(bold=False, size=max(12, int(band_h * 0.15)))
    data_bold_font = find_times_font(bold=True, size=max(13, int(band_h * 0.16)))
    status_font = find_times_font(bold=True, size=max(10, int(band_h * 0.15)))

    # If the requested Times-style font is unavailable, PIL/OpenCV
    # fallbacks still produce a readable panel.
    def draw_centered(text, center_x, y, font, bold=False):
        nonlocal band
        tw, th = text_bbox_pil(text, font)
        band = put_text_pil(
            band,
            text,
            (int(center_x - tw / 2), int(y)),
            font,
            fill=(0, 0, 0)
        )
        return th

    # Name on the left.
    nw, nh = text_bbox_pil(IMAGE_BASENAME, name_font)
    band = put_text_pil(
        band,
        IMAGE_BASENAME,
        (left_margin, max(1, (band_h - nh) // 2)),
        name_font,
        fill=(0, 0, 0)
    )

    status = "DROPLET DETECTED" if ellipse_measurement is not None else "INSUFFICIENT FIT"
    sw, sh = text_bbox_pil(status, status_font)
    band = put_text_pil(
        band,
        status,
        (int(stats_x + (stats_w - sw) / 2), 2),
        status_font,
        fill=(0, 0, 0)
    )

    if ellipse_measurement is None:
        return np.vstack([photo_part, band])

    theta_c = (
        circle_measurement["mean_angle"]
        if circle_measurement is not None
        else None
    )

    row1 = [
        f"Left Angle: {ellipse_measurement['left_angle']:.1f} deg",
        f"Right Angle: {ellipse_measurement['right_angle']:.1f} deg",
        f"Mean Angle: {ellipse_measurement['mean_angle']:.1f} deg",
    ]

    row2 = [
        f"ThetaE: {ellipse_measurement['mean_angle']:.1f} deg",
        f"ThetaC: {theta_c:.1f} deg" if theta_c is not None else "ThetaC: N/A",
        f"Base / Length: {abs(ellipse_measurement['right_x'] - ellipse_measurement['left_x']):.1f} px",
    ]

    # Two clean horizontal rows. Only Mean Angle is bold, matching the
    # earlier lab-style requirement while making every value readable.
    row1_y = int(band_h * 0.32)
    row2_y = int(band_h * 0.64)

    for i, text in enumerate(row1):
        center_x = stats_x + (i + 0.5) * col_w
        font = data_bold_font if i == 2 else data_font
        draw_centered(text, center_x, row1_y, font)

    for i, text in enumerate(row2):
        center_x = stats_x + (i + 0.5) * col_w
        draw_centered(text, center_x, row2_y, data_font)

    return np.vstack([photo_part, band])


# ============================================================
# CSV-FORMATTED TXT EXPORT
# ============================================================

def export_txt(
    filename,
    ellipse_measurement,
    circle_measurement,
    baseline_length_px,
    point_count
):
    """
    The extension is .txt deliberately, but the content is valid
    comma-separated CSV-style text. It can be copied directly into
    a CSV/Excel workflow.
    """

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "name",
            IMAGE_BASENAME
        ])

        writer.writerow([
            "left_angle_deg",
            f"{ellipse_measurement['left_angle']:.4f}"
        ])

        writer.writerow([
            "right_angle_deg",
            f"{ellipse_measurement['right_angle']:.4f}"
        ])

        writer.writerow([
            "mean_angle_deg",
            f"{ellipse_measurement['mean_angle']:.4f}"
        ])

        writer.writerow([
            "ThetaE_deg",
            f"{ellipse_measurement['mean_angle']:.4f}"
        ])

        writer.writerow([
            "ThetaC_deg",
            (
                f"{circle_measurement['mean_angle']:.4f}"
                if circle_measurement is not None
                else "N/A"
            )
        ])

        writer.writerow([
            "base_length_px",
            f"{baseline_length_px:.4f}"
        ])

        writer.writerow([
            "surface_points",
            point_count
        ])

        writer.writerow([
            "ellipse_eccentricity",
            f"{ellipse_measurement['eccentricity']:.6f}"
        ])

        writer.writerow([
            "circle_radius_px",
            (
                f"{circle_measurement['radius']:.4f}"
                if circle_measurement is not None
                else "N/A"
            )
        ])

        writer.writerow([
            "baseline_point_1",
            f"{baseline_x1:.4f}",
            f"{baseline_y1:.4f}"
        ])

        writer.writerow([
            "baseline_point_2",
            f"{baseline_x2:.4f}",
            f"{baseline_y2:.4f}"
        ])

        # Also store the manually selected surface points so the
        # measurement can be reconstructed/checked later.
        for i, p in enumerate(surface_points, start=1):
            writer.writerow([
                f"surface_point_{i}",
                f"{p[0]:.4f}",
                f"{p[1]:.4f}"
            ])


# ============================================================
# EXPORT ANALYZED IMAGE + TXT
# ============================================================

def export_analyzed():
    if not frozen or frozen_frame is None:
        print("Nothing is frozen. Press C first.")
        return

    if not baseline_ready():
        print("Select the two baseline points first.")
        return

    if len(surface_points) < MIN_SURFACE_POINTS:
        print(
            f"Select at least {MIN_SURFACE_POINTS} "
            "droplet-surface points first."
        )
        return

    ellipse, circle = calculate_fits()

    if ellipse is None:
        print(
            "Could not obtain a valid ellipse/baseline intersection. "
            "Check the baseline and surface points."
        )
        return

    # Refit once on export using the same user-selected points.
    final_ellipse = refine_ellipse(
        valid_surface_points(),
        ellipse
    )

    # Recalculate circle from the same points.
    final_circle = fit_circle(
        valid_surface_points()
    )

    # Keep global state synchronized with the final export values.
    global last_ellipse, last_circle
    last_ellipse = final_ellipse
    last_circle = final_circle

    base_length_px = abs(
        final_ellipse["right_x"]
        - final_ellipse["left_x"]
    )

    analyzed = draw_analyzed_export(
        frozen_frame,
        final_ellipse,
        final_circle
    )

    image_filename = f"{IMAGE_BASENAME}_analyzed.png"
    txt_filename = f"{IMAGE_BASENAME}_analyzed.txt"

    ok = cv2.imwrite(image_filename, analyzed)

    if not ok:
        print(f"ERROR: Could not save {image_filename}")
        return

    export_txt(
        txt_filename,
        final_ellipse,
        final_circle,
        base_length_px,
        len(surface_points)
    )

    print()
    print("==============================================")
    print(" ANALYSIS EXPORTED")
    print("==============================================")
    print(f"Image : {image_filename}")
    print(f"TXT   : {txt_filename}  (CSV-formatted content)")
    print()
    print(f"Surface points = {len(surface_points)}")
    print(f"Left Angle  = {final_ellipse['left_angle']:.3f} deg")
    print(f"Right Angle = {final_ellipse['right_angle']:.3f} deg")
    print(f"Mean Angle  = {final_ellipse['mean_angle']:.3f} deg")

    if final_circle is not None:
        print(f"ThetaC      = {final_circle['mean_angle']:.3f} deg")

    print(f"Base / Length = {base_length_px:.3f} px")
    print("==============================================")
    print()


# ============================================================
# RESET
# ============================================================

def reset_measurement():
    global frozen
    global frozen_frame
    global baseline_x1, baseline_y1
    global baseline_x2, baseline_y2
    global surface_points
    global last_ellipse, last_circle

    frozen = False
    frozen_frame = None

    baseline_x1 = None
    baseline_y1 = None
    baseline_x2 = None
    baseline_y2 = None

    surface_points = []

    last_ellipse = None
    last_circle = None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    global baseline_x1, baseline_y1
    global baseline_x2, baseline_y2
    global surface_points

    mouse_x = x
    mouse_y = y

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Surface tracing is allowed only after the camera has been
    # frozen with C.
    if not frozen:
        print("Press C first to freeze the frame before tracing.")
        return

    # --------------------------------------------------------
    # Step 1: baseline point 1
    # --------------------------------------------------------
    if baseline_x1 is None:
        baseline_x1 = int(x)
        baseline_y1 = int(y)

        print(
            f"Baseline point 1: ({x}, {y})"
        )
        print(
            "Now click baseline point 2."
        )
        return

    # --------------------------------------------------------
    # Step 2: baseline point 2
    # --------------------------------------------------------
    if baseline_x2 is None:
        baseline_x2 = int(x)
        baseline_y2 = int(y)

        baseline_length = math.hypot(
            baseline_x2 - baseline_x1,
            baseline_y2 - baseline_y1
        )

        if baseline_length < 5:
            print("Baseline is too short. Select point 2 again.")
            baseline_x2 = None
            baseline_y2 = None
            return

        print(
            f"Baseline point 2: ({x}, {y})"
        )
        print(
            f"Baseline length = {baseline_length:.1f} px"
        )
        print(
            f"Now click {RECOMMENDED_SURFACE_POINTS} points "
            "on the outer droplet surface."
        )
        return

    # --------------------------------------------------------
    # Step 3: manual droplet outer-surface points
    # --------------------------------------------------------
    if len(surface_points) >= MAX_SURFACE_POINTS:
        print(
            f"Maximum of {MAX_SURFACE_POINTS} surface points reached."
        )
        return

    surface_points.append(
        np.array([float(x), float(y)], dtype=float)
    )

    print(
        f"Surface point {len(surface_points)}: ({x}, {y})"
    )
    print("  Point and current geometry are shown immediately in the window.")

    if len(surface_points) >= MIN_SURFACE_POINTS:
        ellipse, circle = calculate_fits()

        if ellipse is not None:
            print(
                f"  ThetaE = {ellipse['mean_angle']:.2f} deg"
            )

        if circle is not None:
            print(
                f"  ThetaC = {circle['mean_angle']:.2f} deg"
            )


def clear_surface_points():
    global surface_points
    global last_ellipse, last_circle

    surface_points = []
    last_ellipse = None
    last_circle = None

    print("All droplet-surface points cleared.")


def undo_last_surface_point():
    global surface_points
    global last_ellipse, last_circle

    if not surface_points:
        print("No surface point to undo.")
        return

    removed = surface_points.pop()
    last_ellipse = None
    last_circle = None

    print(
        f"Removed surface point: "
        f"({removed[0]:.1f}, {removed[1]:.1f})"
    )

    if len(surface_points) >= MIN_SURFACE_POINTS:
        calculate_fits()


# ============================================================
# LIVE / FROZEN STATUS
# ============================================================

def draw_status(display):
    h, w = display.shape[:2]

    if not frozen:
        lines = [
            "LIVE  |  C = capture + freeze  |  R = new measurement  |  ESC = stop"
        ]
    elif not baseline_ready():
        lines = [
            "FROZEN  |  Step 1: click two baseline points",
            "E = export after tracing  |  R = new measurement  |  ESC = stop"
        ]
    elif len(surface_points) < MIN_SURFACE_POINTS:
        lines = [
            f"FROZEN  |  Step 2: click droplet outer surface points ({len(surface_points)}/{RECOMMENDED_SURFACE_POINTS})",
            "5 minimum  |  U = undo  |  X = clear  |  E = export  |  R = new  |  ESC = stop"
        ]
    else:
        lines = [
            f"FROZEN  |  {len(surface_points)} surface points | U = undo | X = clear | E = export | R = new | ESC = stop"
        ]

    y = 22

    for line in lines:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            WHITE,
            1,
            cv2.LINE_AA
        )
        y += 21

    # Name.
    cv2.putText(
        display,
        f"Name: {IMAGE_BASENAME}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA
    )

    # Live/frozen measurement preview.
    if last_ellipse is not None:
        cv2.putText(
            display,
            f"ThetaE = {last_ellipse['mean_angle']:.2f} deg",
            (w - 185, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )

    if last_circle is not None:
        cv2.putText(
            display,
            f"ThetaC = {last_circle['mean_angle']:.2f} deg",
            (w - 185, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )

    return display


# ============================================================
# MAIN PROGRAM
# ============================================================
print()
print(f"Name: {IMAGE_BASENAME}")
print()
print("Workflow:")
print("  C -> save RAW image + freeze")
print("  2 baseline clicks")
print("  5-6 surface clicks")
print("  E -> analyzed PNG + CSV-formatted TXT")
print("  R -> ask for NEW name + return to live")
print("  Q -> quit")
print()
print("Opening USB webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print()
    print("ERROR: Webcam could not be opened.")
    print("Try changing CAMERA_INDEX from 0 to 1, 2, etc.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

WINDOW_NAME = "Real-Time Contact Angle Measurement"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)

print()
print("Move the mouse over the image for the precision loupe.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # LIVE MODE: continuously read camera.
    # FROZEN MODE: deliberately do NOT read camera.
    # --------------------------------------------------------

    if not frozen:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        display = frame.copy()
        draw_status(display)

    else:
        frame = frozen_frame.copy()
        display = frame.copy()

        # The only contour shown is the user's manually selected
        # surface points and the curves fitted from those points.
        # IMPORTANT: draw_measurement_overlay() returns a new image,
        # so its returned frame must be assigned back to display.
        # This makes every click, baseline, fitted curve, tangent and
        # angle marker visible immediately during live interaction.
        display = draw_measurement_overlay(
            display,
            last_ellipse,
            last_circle
        )

        draw_status(display)

    # Loupe is UI-only and is never saved.
    display = draw_magnifier(display)

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    key = cv2.waitKey(20 if frozen else 1) & 0xFF

    # --------------------------------------------------------
    # ESC = stop program immediately
    # --------------------------------------------------------
    if key == 27:
        print("ESC pressed. Closing camera and stopping program...")
        break

    # --------------------------------------------------------
    # C = capture RAW + freeze
    # --------------------------------------------------------
    elif key == ord("c"):
        if not frozen:
            frozen_frame = frame.copy()
            frozen = True

            # Save the exact frame that was frozen.
            capture_raw(frozen_frame)

            print()
            print("Frame frozen.")
            print("Step 1: click two points for the baseline.")
            print()
        else:
            # If already frozen, C simply saves the same raw frame
            # again under the current name. It does not pull a new frame.
            capture_raw(frozen_frame)

    # --------------------------------------------------------
    # E = export analyzed image + TXT
    # --------------------------------------------------------
    elif key == ord("e"):
        export_analyzed()

    # --------------------------------------------------------
    # R = reset + ASK FOR NEW NAME
    # --------------------------------------------------------
    elif key == ord("r"):
        reset_measurement()

        print()
        print("----------------------------------------------")
        print("NEW MEASUREMENT")
        print("----------------------------------------------")

        IMAGE_BASENAME = ask_for_image_name()

        print(f"New name: {IMAGE_BASENAME}")
        print("Camera is live again.")
        print("Press C when the next droplet is ready.")
        print()

    # --------------------------------------------------------
    # U = undo last surface point
    # --------------------------------------------------------
    elif key == ord("u"):
        if frozen and baseline_ready():
            undo_last_surface_point()
        else:
            print("U is available after the baseline is defined.")

    # --------------------------------------------------------
    # X = clear all surface points
    # --------------------------------------------------------
    elif key == ord("x"):
        if frozen and baseline_ready():
            clear_surface_points()
        else:
            print("X is available after the baseline is defined.")

    # --------------------------------------------------------
    # Q = quit
    # --------------------------------------------------------
    elif key == ord("q"):
        print("Closing camera...")
        break


cap.release()
cv2.destroyAllWindows()

print()
print("Contact-angle measurement program closed.")
print("Thank you for using MECHATRIX AngleC.")
print("Concept and system design: Mir Md Minhaajuddin")
print("Coding assistance: ChatGPT and Claude")
print("Copyright © 2026 Mir Md Minhaajuddin. All rights reserved.")
