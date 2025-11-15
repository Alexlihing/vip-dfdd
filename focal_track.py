# Alex Li
# VIP - A-Camera
# Fall 2025

# focal_track.py - calibration and visualization helpers for focal tracking
# Requires opencv-python, numpy, scipy, matplotlib

# High-level Focal Split / DfDD pipeline
# 1) Load calibration dataset and extract image pairs + depths
#    - load_calib_pkl(), _extract_pair_from_rec(), _extract_depth_from_rec(), _pair_list_of_img_loc()
#
# 2) Preprocess and (optionally) align each calibration image pair
#    - _ensure_gray_np(), preprocess(), align_homography()
#
# 3) Compute differential-defocus ratio and confidence mask per pair
#    - ratio_fs() or ratio_ft(), spatial_gradients(), laplacian(), aggregate(), confidence()
#
# 4) Collect reliable (r, Z) samples from all calibration frames
#    - collect_r_and_Z_from_dataset()
#
# 5) Fit the depth model Z ≈ A / (B + r) (or Padé[1/1]) from (r, Z)
#    - calibrate_ab() and depth_from_ratio()
#      (optionally: fit_pade11(), depth_from_ratio_pade11())
#
# 6) Visualize calibration quality with a 2D histogram of Z_true vs Z_est
#    - histogram_from_arrays(), plot_depth_histogram(), overlay_bin_median()
#
# 7) For a new image pair, compute ratio + confidence in the same way
#    - depth_for_pair(): imread_gray(), preprocess(), align_homography(),
#      ratio_fs() / ratio_ft(), confidence()
#
# 8) Convert ratio map to a sparse depth map using calibrated A, B and save visualizations
#    - depth_from_ratio(), normalize_to_uint8(), colorize_depth_cv2(),
#      save_confidence_bw(), depth_for_pair() (gray/color/confidence PNGs)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import pickle
import itertools

LOC_TO_Z_OFFSET = 0.4       # meters
LOC_TO_Z_SCALE  = 4e-7      # meters per step

def _ensure_gray_np(img_any) -> np.ndarray:
    """Accepts a path or array (RGB/BGR/gray) and returns a float64 grayscale image scaled to 0..255."""
    if isinstance(img_any, str):
        arr = cv2.imread(img_any, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(img_any)
    else:
        arr = np.asarray(img_any)

    # Normalize any 3-channel input before converting to grayscale
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            mx = float(np.nanmax(a)) if a.size else 1.0
            if mx <= 1.01:  # looks like 0..1
                a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
            else:
                a = np.clip(a, 0, 255).astype(np.uint8)
        else:
            a = arr
        try:
            gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        except Exception:
            gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float64)

    # Already-grayscale arrays only need dtype/scale cleanup
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            mx = float(np.nanmax(a)) if a.size else 1.0
            if mx <= 1.01:
                a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
            else:
                a = np.clip(a, 0, 255).astype(np.uint8)
            return a.astype(np.float64)
        return arr.astype(np.float64)

    raise ValueError(f"Unexpected image shape: {arr.shape}")

def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float64)

def preprocess(img: np.ndarray, illum_ksize: int = 21, denoise_sigma: float = 1.5) -> np.ndarray:
    bg = cv2.boxFilter(img, ddepth=-1, ksize=(illum_ksize, illum_ksize))
    flat = img - bg
    if denoise_sigma and denoise_sigma > 0:
        flat = cv2.GaussianBlur(flat, (0, 0), denoise_sigma)
    return flat

def _make_feature_detector():
    try:
        return cv2.SIFT_create(), "SIFT", cv2.NORM_L2
    except Exception:
        orb = cv2.ORB_create(nfeatures=4000)
        return orb, "ORB", cv2.NORM_HAMMING

def align_homography(I_ref: np.ndarray, I_mov: np.ndarray):
    """Align I_mov to I_ref via SIFT/ORB + RANSAC homography; return warped image & valid overlap mask."""
    h, w = I_ref.shape
    det, name, norm_type = _make_feature_detector()
    k1, d1 = det.detectAndCompute(I_ref.astype(np.uint8), None)
    k2, d2 = det.detectAndCompute(I_mov.astype(np.uint8), None)

    def _warp_and_mask(img, H):
        warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ones = np.ones_like(img, dtype=np.float32)
        valid = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        valid_overlap = valid > 0.5
        return warped, H, valid_overlap

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return _warp_and_mask(I_mov, np.eye(3))

    if name == "SIFT":
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = matcher.knnMatch(d2, d1, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        good = matcher.match(d2, d1)

    if len(good) < 8:
        return _warp_and_mask(I_mov, np.eye(3))

    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        H = np.eye(3)
    return _warp_and_mask(I_mov, H)

# Derivative helpers and focal-ratio estimators -----

def spatial_gradients(I: np.ndarray):
    """Compute Sobel gradients used throughout the ratio calculations."""
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy

def laplacian(I: np.ndarray):
    """Simple Laplacian wrapper to keep the filter parameters consistent."""
    return cv2.Laplacian(I, cv2.CV_64F, ksize=3)

def aggregate(img: np.ndarray, win: int = 21) -> np.ndarray:
    """Uniform filter aggregation to calm noise before ratio evaluation."""
    if win <= 1:
        return img
    return uniform_filter(img, size=win, mode='nearest')

def ratio_fs(I1: np.ndarray, I2: np.ndarray, agg_win: int = 21):
    """Compute the focus sweep (FS) ratio."""
    Iavg = 0.5 * (I1 + I2)
    Lap = laplacian(Iavg)
    Is  = I1 - I2
    num_s = aggregate(Is,  agg_win)
    den_s = aggregate(Lap, agg_win)
    eps = 1e-8
    r = num_s / (den_s + eps)
    return r, Is, Lap

def ratio_ft(I1: np.ndarray, I2: np.ndarray, d: float = 0.0, agg_win: int = 21):
    """Compute the focus tracking (FT) ratio with the optional affine term d."""
    Iavg = 0.5 * (I1 + I2)
    Lap = laplacian(Iavg)
    Is  = I1 - I2
    Ix, Iy = spatial_gradients(Iavg)
    h, w = Iavg.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx - w / 2.0
    yy = yy - h / 2.0
    magnif = xx * Ix + yy * Iy
    num = d * magnif + Is
    den = Lap
    num_s = aggregate(num, agg_win)
    den_s = aggregate(den, agg_win)
    eps = 1e-8
    r = num_s / (den_s + eps)
    return r, Is, Lap

def confidence(Is: np.ndarray, Lap: np.ndarray, drop_percent: float = 40.0) -> np.ndarray:
    """Binary mask that keeps only pixels with strong differences and curvature."""
    C = Is**2
    tC = np.percentile(C, drop_percent)
    tL = np.percentile(np.abs(Lap), drop_percent)
    return (C > tC) & (np.abs(Lap) > tL)

IMG_KEYS_2 = [("I1","I2"), ("img1","img2"), ("im1","im2"), ("left","right")]
IMG_KEYS_LIST = ["images","imgs","img","Imgs","Img"]

# Keys that already encode depth in meters
DEPTH_KEYS = ["Z", "depth", "z_true", "z", "Ztrue", "Z_true"]

def _is_img_array(x):
    try:
        a = np.asarray(x)
    except Exception:
        return False
    return a.ndim in (2,3) and a.size > 0

def _as_gray_np(x):
    if isinstance(x, str) or _is_img_array(x):
        return _ensure_gray_np(x)
    raise ValueError("Not an image-like object or path")

def _extract_pair_from_rec(rec):
    # Prefer the obvious pair keys when they exist
    for k1, k2 in IMG_KEYS_2:
        if k1 in rec and k2 in rec:
            return _as_gray_np(rec[k1]), _as_gray_np(rec[k2])
    # Fall back to list-style containers that bundle the frames
    for k in IMG_KEYS_LIST:
        if k in rec:
            v = rec[k]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return _as_gray_np(v[0]), _as_gray_np(v[1])
            if _is_img_array(v):
                a = np.asarray(v)
                if a.ndim == 3 and a.shape[2] == 2:
                    return _as_gray_np(a[...,0]), _as_gray_np(a[...,1])
                if a.ndim == 3 and a.shape[0] == 2:
                    return _as_gray_np(a[0]), _as_gray_np(a[1])
    return None

def _extract_depth_from_rec(rec):
    """
    Pull a depth value in meters from a record.
    - 'Loc'/'loc' are motor steps and must be converted.
    - Otherwise look for keys that already represent meters.
    """
    # Motor steps still need a conversion before use
    if "Loc" in rec or "loc" in rec:
        loc_val = rec.get("Loc", rec.get("loc"))
        try:
            loc_val = float(np.mean(loc_val)) if np.ndim(loc_val) > 0 else float(loc_val)
        except Exception:
            return None
        Z_true = LOC_TO_Z_OFFSET + LOC_TO_Z_SCALE * loc_val
        return float(Z_true)

    # Otherwise check the whitelist of keys that store depth directly
    for k in DEPTH_KEYS:
        if k in rec:
            z = rec[k]
            try:
                return float(np.mean(z)) if np.ndim(z) > 0 else float(z)
            except Exception:
                continue

    return None

def _pair_list_of_img_loc(records_list, out):
    """
    Walk a list such as [{'Img': ..., 'Loc': ...}, ...],
    pair entries (0,1), (2,3), ... and convert Loc (steps) to meters.
    """
    items = [r for r in records_list if isinstance(r, dict) and ('Img' in r) and ('Loc' in r)]
    if len(items) < 2:
        return

    for i in range(0, len(items)-1, 2):
        I1 = items[i]['Img']
        I2 = items[i+1]['Img']

        # Convert recorded stage steps into meters
        loc_i   = float(np.mean(items[i]['Loc']))      # handles scalar or array input
        Z_true  = LOC_TO_Z_OFFSET + LOC_TO_Z_SCALE * loc_i

        out.append({
            "images": (_ensure_gray_np(I1), _ensure_gray_np(I2)),
            "Z": float(Z_true)
        })

def load_calib_pkl(pkl_path: str):
    """Traverse a calibration PKL and emit a list of {'images': (I1,I2), 'Z': depth_meters} entries."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    collected = []

    def walk(obj):
        """Depth-first search through mixed dict/list structures."""
        if isinstance(obj, dict):
            pair = _extract_pair_from_rec(obj)
            z    = _extract_depth_from_rec(obj)
            if pair is not None and z is not None:
                I1, I2 = pair
                collected.append({
                    "images": (_ensure_gray_np(I1), _ensure_gray_np(I2)),
                    "Z": float(np.mean(z))
                })
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            if all(isinstance(it, dict) for it in obj):
                has_img = sum(1 for it in obj if 'Img' in it) >= 2
                has_loc = any('Loc' in it or 'loc' in it for it in obj)
                if has_img and has_loc:
                    _pair_list_of_img_loc(obj, collected)
            for v in obj:
                walk(v)

    walk(data)

    if not collected:
        raise RuntimeError("No valid calibration records found in PKL.")
    return collected

# Collect (r, Z) samples from the dataset with filtering

def collect_r_and_Z_from_dataset(
    pkl_path: str,
    use_ft: bool = False,
    d: float = 0.0,
    do_align: bool = True,
    agg_win: int = 21,
    drop_percent: float = 40.0,
):
    # Pull normalized image pairs with known depths
    samples = load_calib_pkl(pkl_path)

    # Quick stats help catch some PKL issues early
    Z_list = [float(rec["Z"]) for rec in samples if np.isfinite(rec.get("Z", np.nan))]
    if not Z_list:
        raise RuntimeError("No valid Z values found in calibration data.")
    Z_arr = np.array(Z_list, dtype=np.float64)
    print("Z_true min/max/median (m):", Z_arr.min(), Z_arr.max(), np.median(Z_arr))

    all_r, all_Z = [], []
    min_Z = 1e-6

    for idx, rec in enumerate(samples):
        if idx % 25 == 0:
            print(f"[calib] processing {idx+1}/{len(samples)}", flush=True)

        Z_true = float(rec["Z"])
        if not np.isfinite(Z_true) or Z_true <= min_Z:
            continue

        I1 = preprocess(_ensure_gray_np(rec["images"][0]))
        I2 = preprocess(_ensure_gray_np(rec["images"][1]))

        if do_align:
            I2, _, valid_overlap = align_homography(I1, I2)
        else:
            valid_overlap = np.ones_like(I1, dtype=bool)

        if use_ft:
            r, Is, Lap = ratio_ft(I1, I2, d=d, agg_win=agg_win)
        else:
            r, Is, Lap = ratio_fs(I1, I2, agg_win=agg_win)

        # Keep confident pixels that also survive the geometric overlap mask
        m = confidence(Is, Lap, drop_percent=drop_percent) & valid_overlap
        rv = r[m]
        if rv.size == 0:
            continue

        rv = rv[np.isfinite(rv)]
        if rv.size == 0:
            continue

        Zv = np.full_like(rv, fill_value=Z_true, dtype=np.float64)
        all_r.append(rv)
        all_Z.append(Zv)

    if not all_r:
        raise RuntimeError("No valid confident pixels collected (after filtering).")

    r_values = np.concatenate(all_r, axis=0)
    Z_values = np.concatenate(all_Z, axis=0)
    return r_values, Z_values

# ============================================================
# Fit calibration curves (rational and Padé[1/1])
# ============================================================

def calibrate_ab(r_values, Z_values, clip_percentiles=(1.0, 99.0)):
    """Fit the simple r ≈ a*(1/Z) - b model with percentile clipping."""
    r = np.asarray(r_values, float).ravel()
    Z = np.asarray(Z_values, float).ravel()
    min_Z = 1e-6
    m0 = np.isfinite(r) & np.isfinite(Z) & (Z > min_Z)
    r, Z = r[m0], Z[m0]
    if r.size < 10:
        raise RuntimeError("Not enough samples for calibration.")

    x = 1.0 / Z
    p_lo, p_hi = clip_percentiles
    r_lo, r_hi = np.percentile(r, [p_lo, p_hi])
    x_lo, x_hi = np.percentile(x, [p_lo, p_hi])
    m1 = (r >= r_lo) & (r <= r_hi) & (x >= x_lo) & (x <= x_hi)
    r, x = r[m1], x[m1]
    if r.size < 10:
        raise RuntimeError("Too few samples after clipping; relax clip_percentiles.")

    A = np.vstack([x, np.ones_like(x)]).T
    try:
        sol, *_ = np.linalg.lstsq(A, r, rcond=None)
    except np.linalg.LinAlgError:
        sol = np.polyfit(x, r, 1)  # fallback slope/intercept fit
    a = float(sol[0])
    b = -float(sol[1])  # store intercept as -b so r ≈ a*x - b
    return a, b

def fit_pade11(r_values, Z_values, clip=(1.0, 99.0)):
    """Fit the Padé[1/1] rational model Z ≈ (p0 + p1*r) / (1 + q1*r)."""
    r = np.asarray(r_values, float).ravel()
    Z = np.asarray(Z_values, float).ravel()
    ok = np.isfinite(r) & np.isfinite(Z) & (Z > 1e-6)
    r, Z = r[ok], Z[ok]
    if r.size < 10:
        raise RuntimeError("Not enough samples for Padé[1/1].")

    rlo,rhi = np.percentile(r,[clip[0],clip[1]])
    Zlo,Zhi = np.percentile(Z,[clip[0],clip[1]])
    m = (r>=rlo)&(r<=rhi)&(Z>=Zlo)&(Z<=Zhi)
    r, Z = r[m], Z[m]
    if r.size < 10:
        raise RuntimeError("Too few samples after clipping for Padé[1/1].")

    # Derived from: Z ≈ (p0 + p1 r) / (1 + q1 r)  =>  Z - p0 - p1 r + q1 Z r ≈ 0
    A = np.column_stack([np.ones_like(r), r, -(Z*r)])
    sol, *_ = np.linalg.lstsq(A, Z, rcond=None)
    p0, p1, q1 = sol
    return float(p0), float(p1), float(q1)

def depth_from_ratio(r: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate depth from an FS ratio using the a/b calibration."""
    eps = 1e-8
    return a / (b + r + eps)

def depth_from_ratio_pade11(r: np.ndarray, p0: float, p1: float, q1: float) -> np.ndarray:
    """Evaluate depth with the Padé[1/1] mapping."""
    return (p0 + p1*r) / (1.0 + q1*r + 1e-12)

# Depth histogram plotting helpers ---------------------

def overlay_bin_median(ax, Z_true, Z_est, rng, bins=60, color="orange", lw=2):
    """Overlay a ridge that tracks the median estimate per true-depth bin."""
    Zt = np.asarray(Z_true).ravel()
    Ze = np.asarray(Z_est).ravel()
    m  = np.isfinite(Zt) & np.isfinite(Ze)
    Zt, Ze = Zt[m], Ze[m]
    edges = np.linspace(rng[0], rng[1], bins+1)
    centers, meds = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        sel = (Zt >= lo) & (Zt < hi)
        if np.any(sel):
            centers.append(0.5*(lo+hi))
            meds.append(np.median(Ze[sel]))
    if centers:
        ax.plot(centers, meds, color=color, linewidth=lw)

def plot_depth_histogram(
    Z_est: np.ndarray,
    Z_true: np.ndarray,
    bins: int = 40,
    rng: tuple[float, float] | None = None,
    normalize_cols: bool = True,
    pathname: str = "depth_vs_true.png",
    title: str | None = None,
    show_fit_line: bool = True,
    show_ridge: bool = False,
):
    """Heatmap showing estimated vs true depth with optional ridge/fit overlays."""
    Zt = np.asarray(Z_true, float).ravel()
    Ze = np.asarray(Z_est,  float).ravel()
    valid = np.isfinite(Zt) & np.isfinite(Ze)
    Zt, Ze = Zt[valid], Ze[valid]
    if Zt.size == 0:
        print("[plot_depth_histogram] No valid samples.")
        return

    if rng is None:
        lo, hi = np.percentile(Zt, [1, 99])  # clamp to observed depth span
        pad = 0.02 * (hi - lo) if hi > lo else 0.05
        rng = (lo - pad, hi + pad)

    heatmap, xedges, yedges = np.histogram2d(Zt, Ze, bins=bins, range=[rng, rng])
    heatmap = heatmap.T
    if normalize_cols:
        col_sums = heatmap.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        heatmap = heatmap / col_sums

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap, extent=extent, origin="lower", aspect="equal", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Identity line shows how far estimates drift from the ideal
    ax.plot([rng[0], rng[1]], [rng[0], rng[1]], color="white", alpha=0.65, linewidth=2)

    # Light-weight linear fit for a quick global sanity check
    if show_fit_line and Zt.size > 10:
        try:
            m, c = np.polyfit(Zt, Ze, 1)
            xs = np.array([rng[0], rng[1]], dtype=float)
            ys = m*xs + c
            ax.plot(xs, ys, color="orange", alpha=0.9, linewidth=2)
        except Exception:
            pass

    # Ridge highlights the median estimate within each true-depth bin
    if show_ridge:
        overlay_bin_median(ax, Zt, Ze, rng, bins=60, color="orange", lw=2)

    ax.set_xlabel("True Depth")
    ax.set_ylabel("Estimated Depth")
    if title:
        ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig(pathname)
    plt.close(fig)

def histogram_from_arrays(
    r_values: np.ndarray,
    Z_values: np.ndarray,
    params: dict,
    pathname: str = "calib_depth_vs_true.png",
    rng: tuple[float, float] | None = None,
    normalize_cols: bool = True,
    title: str | None = "Estimated vs True Depth (DfDD calibration)",
    mapping: str = "rational",  # or "pade11"
):
    """Convenience wrapper that maps r-values to Z and plots the histogram."""
    if mapping == "rational":
        Z_est = depth_from_ratio(r_values, params["a"], params["b"])
    else:
        Z_est = depth_from_ratio_pade11(r_values, params["p0"], params["p1"], params["q1"])
    plot_depth_histogram(
        Z_est=Z_est, Z_true=Z_values, bins=40, rng=rng,
        normalize_cols=normalize_cols, pathname=pathname, title=title,
        show_fit_line=True, show_ridge=False
    )

def pick_d_by_grid(pkl_path, d_vals, agg_win=21, drop_percent=40.0, use_pade=True, do_align=False):
    """Try a list of d values and keep the one with the best R^2 on calibration data."""
    best = None
    for d in d_vals:
        r_all, Z_all = collect_r_and_Z_from_dataset(
            pkl_path, use_ft=True, d=d, do_align=do_align, agg_win=agg_win, drop_percent=drop_percent
        )
        if use_pade:
            p0,p1,q1 = fit_pade11(r_all, Z_all)
            Z_hat = depth_from_ratio_pade11(r_all, p0,p1,q1)
        else:
            a,b = calibrate_ab(r_all, Z_all)
            Z_hat = depth_from_ratio(r_all, a, b)

        Z = np.asarray(Z_all, float).ravel()
        Z_hat = np.asarray(Z_hat, float).ravel()
        ok = np.isfinite(Z) & np.isfinite(Z_hat)
        if not np.any(ok): 
            continue
        ss_res = np.sum((Z[ok] - Z_hat[ok])**2)
        ss_tot = np.sum((Z[ok] - np.mean(Z[ok]))**2) + 1e-12
        R2 = 1 - ss_res/ss_tot
        if (best is None) or (R2 > best[0]):
            if use_pade: best = (R2, d, {"p0":p0,"p1":p1,"q1":q1}, "pade11")
            else:        best = (R2, d, {"a":a,"b":b}, "rational")
    return best  # tuple: (R2, d_best, params, mapping_tag)

# Visualization helpers for a single image pair ------

def colorize_depth_cv2(depth: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Map sparse depth values to a color image while keeping invalid pixels black."""
    norm = np.zeros_like(depth, dtype=np.uint8)
    valid = depth[mask]
    if valid.size > 0:
        order = np.argsort(valid, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.linspace(0, 255, num=valid.size, endpoint=True)
        norm_vals = ranks.astype(np.uint8)
        norm[mask] = norm_vals
    color = cv2.applyColorMap(norm, colormap)
    color[~mask] = (0, 0, 0)
    return color

def _valid_bbox(mask: np.ndarray):
    """Tight bounding box around valid pixels."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)

def normalize_to_uint8(arr: np.ndarray, mask: np.ndarray | None = None,
                       clip_percentiles: tuple[float, float] | None = (2.0, 98.0)) -> np.ndarray:
    """Scale an array to 0..255 after optional percentile clipping."""
    a = arr.copy()
    if mask is not None:
        a = np.where(mask, a, np.nan)
    valid = np.isfinite(a)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.uint8)
    if clip_percentiles is not None:
        pmin, pmax = clip_percentiles
        try:
            lo = np.nanpercentile(a[valid], pmin)
            hi = np.nanpercentile(a[valid], pmax)
            if hi > lo:
                a = np.clip(a, lo, hi)
        except Exception:
            pass
    vmin = np.nanmin(a[valid]); vmax = np.nanmax(a[valid])
    if vmax <= vmin + 1e-12:
        out = np.zeros_like(arr, dtype=np.float64)
    else:
        out = (a - vmin) / (vmax - vmin)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def save_confidence_bw(Is: np.ndarray, path_png: str,
                       clip: tuple[float, float] = (2.0, 98.0),
                       gamma: float = 1.3,
                       use_clahe: bool = True,
                       invert: bool = False):
    """Debug helper that writes a contrasty confidence map."""
    C_vis = np.abs(Is)
    gray8 = normalize_to_uint8(C_vis, mask=None, clip_percentiles=clip)
    if gamma and gamma != 1.0:
        x = gray8.astype(np.float32) / 255.0
        x = np.power(x, 1.0 / float(gamma))
        gray8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray8 = clahe.apply(gray8)
    if invert:
        gray8 = 255 - gray8
    cv2.imwrite(path_png, gray8)

def depth_for_pair(img1_path: str, img2_path: str,
                   use_ft: bool = False, d: float = 0.0,
                   do_align: bool = True,
                   agg_win: int = 21,
                   have_cal: bool = False, a: float = 1.0, b: float = 0.0,
                   drop_percent: float = 40.0,
                   gray_path: str = "depth_gray.png",
                   color_path: str = "depth_colored.png",
                   conf_path: str = "confidence.png",
                   color_engine: str = "plt",
                   color_cmap: str = "viridis",
                   invert_color: bool = False,
                   color_lightness: float = 0.25):
    """Process a single stereo pair, returning sparse depth and mask while saving visualizations."""
    I1 = preprocess(imread_gray(img1_path))
    I2 = preprocess(imread_gray(img2_path))
    H0, W0 = I1.shape  # remember original resolution for later upsampling

    # Align the moving image so the ratio computation focuses on defocus, not parallax
    if do_align:
        I2, _, valid_overlap = align_homography(I1, I2)
    else:
        valid_overlap = np.ones_like(I1, dtype=bool)

    if use_ft:
        r, Is, Lap = ratio_ft(I1, I2, d=d, agg_win=agg_win)
    else:
        r, Is, Lap = ratio_fs(I1, I2, agg_win=agg_win)

    # Keep only pixels that are both confident and covered after warping
    mask = confidence(Is, Lap, drop_percent=drop_percent) & valid_overlap

    if have_cal:
        Z = depth_from_ratio(r, a, b)
        Z_sparse = np.where(mask, Z, np.nan)
        vis = Z_sparse
    else:
        Z_sparse = np.where(mask, r, np.nan)
        vis = Z_sparse

    # Visualizations operate on the masked values.
    gray8_full  = normalize_to_uint8(vis, mask=mask)
    color_full  = colorize_depth_cv2(Z_sparse, mask, colormap=cv2.COLORMAP_TURBO)
    conf_full   = normalize_to_uint8(np.abs(Is), mask=None)

    y0, y1, x0, x1 = _valid_bbox(valid_overlap)
    gray_crop  = gray8_full[y0:y1, x0:x1]
    color_crop = color_full[y0:y1, x0:x1]
    conf_crop  = conf_full[y0:y1, x0:x1]

    # Expand back out to the original image size for easier comparison
    gray_out  = cv2.resize(gray_crop,  (W0, H0), interpolation=cv2.INTER_NEAREST)
    color_out = cv2.resize(color_crop, (W0, H0), interpolation=cv2.INTER_NEAREST)
    conf_out  = cv2.resize(conf_crop,  (W0, H0), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(gray_path, gray_out)
    cv2.imwrite(color_path, color_out)
    cv2.imwrite(conf_path, conf_out)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.imshow(gray_out, cmap='gray'); plt.title('Depth (grayscale)'); plt.axis('off')
    ax_col = plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(color_out, cv2.COLOR_BGR2RGB)); plt.title('Depth (colored)'); plt.axis('off')
    try:
        import matplotlib as mpl
        vis_vals = np.array(vis); vis_finite = vis_vals[np.isfinite(vis_vals)]
        vmin, vmax = (float(np.nanmin(vis_finite)), float(np.nanmax(vis_finite))) if vis_finite.size > 0 else (0.0, 1.0)
        cmap = plt.get_cmap('viridis'); norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        plt.colorbar(sm, ax=ax_col, fraction=0.046, pad=0.04, label=('Depth' if have_cal else 'Ratio r'))
    except Exception:
        pass
    plt.subplot(1, 3, 3); plt.imshow(conf_out, cmap='gray'); plt.title('Confidence'); plt.axis('off')
    plt.tight_layout(); plt.show()

    return Z_sparse, mask



# Script
if __name__ == "__main__":
    CALIB_PKL = r"C:\Users\alexl\vip\calibration_dataset.pkl"

    USE_FT   = False        # toggle to use the FT ratio (FS is default)
    SEARCH_D = False        # enable to grid-search d; only makes sense when USE_FT is True
    D_GRID   = np.linspace(-0.002, 0.002, 9)  # search grid for d; adjust to your rig
    DO_ALIGN = False        # skip the expensive homography step if pairs are already aligned
    AGG_WIN  = 21
    DROP_PCT = 40.0
    MAPPING  = "rational"   # can switch to "pade11" to fit Padé coefficients

    if USE_FT and SEARCH_D:
        print("[calib] grid-searching d...")
        best = pick_d_by_grid(CALIB_PKL, D_GRID, agg_win=AGG_WIN, drop_percent=DROP_PCT,
                              use_pade=(MAPPING=="pade11"), do_align=DO_ALIGN)
        if best is None:
            raise RuntimeError("Grid search failed to find a valid d.")
        R2, d_best, params_use, mapping_use = best
        print(f"[calib] d_best={d_best:.6g}, R2={R2:.4f}, mapping={mapping_use}, params={params_use}")
        d_use = d_best
    else:
        d_use = 0.0
        # Single pass: collect ratios, fit parameters, and keep the mapping handy
        r_vals, Z_vals = collect_r_and_Z_from_dataset(
            CALIB_PKL, use_ft=USE_FT, d=d_use, do_align=DO_ALIGN, agg_win=AGG_WIN, drop_percent=DROP_PCT
        )
        if MAPPING == "rational":
            a_hat, b_hat = calibrate_ab(r_vals, Z_vals)
            params_use = {"a": a_hat, "b": b_hat}
            mapping_use = "rational"
            print(f"Calibrated parameters (rational): A={a_hat:.6g}, B={b_hat:.6g}")
        else:
            p0,p1,q1 = fit_pade11(r_vals, Z_vals)
            params_use = {"p0": p0, "p1": p1, "q1": q1}
            mapping_use = "pade11"
            print(f"Calibrated parameters (Padé[1/1]): p0={p0:.6g}, p1={p1:.6g}, q1={q1:.6g}")

        # Visualize how the model tracks the observed true-depth range
        validZ = Z_vals[np.isfinite(Z_vals)]
        zmin, zmax = np.percentile(validZ, [1, 99])
        pad = 0.02 * (zmax - zmin) if zmax > zmin else 0.0
        rng = (zmin - pad, zmax + pad)

        histogram_from_arrays(
            r_vals, Z_vals, params_use,
            pathname="calib_depth_vs_true.png",
            rng=rng, normalize_cols=True,
            title="Estimated vs True Depth (DfDD calibration)",
            mapping=mapping_use
        )

    print("Saved: calib_depth_vs_true.png")

    IMG_1 = r"C:\Users\alexl\vip\Example_CVPR_img0.png"
    IMG_2 = r"C:\Users\alexl\vip\Example_CVPR_aligned.png"

    if mapping_use == "rational":
        have_cal = True
        a_use = params_use["a"]
        b_use = params_use["b"]
    else:
        have_cal = False
        a_use = 1.0
        b_use = 0.0

    Z_sparse, mask = depth_for_pair(
        IMG_1, IMG_2,
        use_ft=USE_FT, d=d_use,
        do_align=True, agg_win=AGG_WIN,
        have_cal=have_cal, a=a_use, b=b_use,
        drop_percent=DROP_PCT,
        gray_path="depth_gray.png",
        color_path="depth_colored.png",
        conf_path="confidence.png",
        color_engine="cv2",
        color_cmap="magma",
    )
    print("Saved: depth_gray.png, depth_colored.png, confidence.png")
    print("Saving to:", os.getcwd())
