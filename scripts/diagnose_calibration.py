"""Per-run diagnostic for the PnLCalib field-registration stage.

Reads ``<run>/field_registration/{calibration_metadata.json, homographies.pkl}``
and reports the distributions that decide whether per-frame solves are
limited by detection (keypoint/line count) or by the solver (mode/ransac
choice, rep_err, focal-length stability).

fx is recovered from the saved H via MVG Algorithm 8.2 since it is not
serialized to metadata. cx, cy are forced to image center inside the
solver — that matches how PnLCalib estimates them, so reading fx back is
straightforward.

Outputs:
- stdout summary (counts + percentile tables)
- ``<run>/field_registration/diagnostic.png`` 6-panel dashboard

Usage:
    python scripts/diagnose_calibration.py <run_dir> [--no-plot]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def estimate_fx_from_H(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Recover fx from a ground-plane H assuming square pixels and PP=(cx,cy).

    Mirrors ``estimate_calibration_matrix_from_plane_homography`` in
    pnlcalib_orig but solves only for fx — square-pixel + centered-PP
    constraints reduce the system to one variable.

    Returns None if the recovery fails (cholesky breakdown / non-positive).
    """
    H = np.asarray(H, dtype=np.float64).reshape(9)
    A = np.zeros((5, 6))
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    A[1, 2] = -1.0
    A[2, 3] = cy / cx if cx != 0 else 0.0
    A[2, 4] = -1.0
    A[3, 0] = H[0] * H[1]
    A[3, 1] = H[0] * H[4] + H[1] * H[3]
    A[3, 2] = H[3] * H[4]
    A[3, 3] = H[0] * H[7] + H[1] * H[6]
    A[3, 4] = H[3] * H[7] + H[4] * H[6]
    A[3, 5] = H[6] * H[7]
    A[4, 0] = H[0] * H[0] - H[1] * H[1]
    A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
    A[4, 2] = H[3] * H[3] - H[4] * H[4]
    A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
    A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
    A[4, 5] = H[6] * H[6] - H[7] * H[7]
    _, _, vh = np.linalg.svd(A)
    w = vh[-1]
    if abs(w[5]) < 1e-12:
        return None
    W = np.array([
        [w[0] / w[5], w[1] / w[5], w[3] / w[5]],
        [w[1] / w[5], w[2] / w[5], w[4] / w[5]],
        [w[3] / w[5], w[4] / w[5], 1.0],
    ])
    try:
        Ktinv = np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        return None
    K = np.linalg.inv(Ktinv.T)
    K = K / K[2, 2]
    fx = float(K[0, 0])
    if not np.isfinite(fx) or fx <= 0:
        return None
    return fx


def camera_pos_from_H(H: np.ndarray, fx: float, cx: float, cy: float) -> np.ndarray | None:
    """Recover camera position (3D) from H given fx, cx, cy."""
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    try:
        hprim = np.linalg.inv(K) @ H
    except np.linalg.LinAlgError:
        return None
    n0 = np.linalg.norm(hprim[:, 0])
    n1 = np.linalg.norm(hprim[:, 1])
    if n0 < 1e-9 or n1 < 1e-9:
        return None
    lam1 = 1.0 / n0
    lam2 = 1.0 / n1
    lam3 = float(np.sqrt(lam1 * lam2))
    r0 = hprim[:, 0] * lam1
    r1 = hprim[:, 1] * lam2
    r2 = np.cross(r0, r1)
    R = np.column_stack([r0, r1, r2])
    u, _, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0:
        u[:, 2] *= -1
        R = u @ vh
    t = hprim[:, 2] * lam3
    return -R.T @ t


def is_good_camera(fx: float | None, pos: np.ndarray | None) -> tuple[bool, str]:
    """Mirror of pnlcalib.camera.is_good_camera (focal + position bounds)."""
    if fx is None:
        return False, "fx-recovery-failed"
    if not (10 <= fx <= 20000):
        return False, f"fx-out-of-range({fx:.0f})"
    if pos is None:
        return False, "pos-recovery-failed"
    if not (-250 < pos[0] < 250 and -250 < pos[1] < 250):
        return False, "pos-xy-out-of-range"
    if not (0 < abs(pos[2]) < 100):
        return False, "pos-z-out-of-range"
    return True, "ok"


def percentiles(arr: list[float], qs=(5, 25, 50, 75, 95)) -> dict[int, float]:
    if not arr:
        return {q: float("nan") for q in qs}
    a = np.asarray(arr, dtype=np.float64)
    return {q: float(np.percentile(a, q)) for q in qs}


def fmt_pct(d: dict[int, float]) -> str:
    return "  ".join(f"p{q}={v:.2f}" for q, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    fr = args.run_dir / "field_registration"
    meta = json.load(open(fr / "calibration_metadata.json"))
    homs: dict[int, np.ndarray] = pickle.load(open(fr / "homographies.pkl", "rb"))

    vinfo = meta["video_info"]
    W, H = int(vinfo["width"]), int(vinfo["height"])
    cx, cy = W / 2.0, H / 2.0
    frames = meta["frames"]
    n_total = len(frames)

    # ---- Bucket frames by saved status -----------------------------------
    calibrated = []          # solver returned a result, kept by sanity gate
    rejected_by_sanity = []  # solver returned a result, killed by gate
    failed_solver = []       # solver returned None
    interpolated = []        # gap-filling chained an H from a neighbour

    for fidx_str, v in frames.items():
        fidx = int(fidx_str)
        if not v.get("calibrated"):
            if v.get("rejected") == "geometry_sanity":
                rejected_by_sanity.append((fidx, v))
            else:
                failed_solver.append((fidx, v))
            continue
        if v.get("interpolated"):
            interpolated.append((fidx, v))
        else:
            calibrated.append((fidx, v))

    print(f"=== {args.run_dir.name} ===")
    print(f"Video: {Path(vinfo['path']).name}  {W}x{H}  fps={vinfo['fps']}  process_fps={vinfo.get('process_fps')}")
    print(f"Sampled frames: {n_total}")
    print(f"  calibrated (solver):       {len(calibrated):4d}  ({len(calibrated)/n_total:.1%})")
    print(f"  interpolated (gap-fill):   {len(interpolated):4d}  ({len(interpolated)/n_total:.1%})")
    print(f"  rejected by sanity gate:   {len(rejected_by_sanity):4d}  ({len(rejected_by_sanity)/n_total:.1%})")
    print(f"  solver failed (no result): {len(failed_solver):4d}  ({len(failed_solver)/n_total:.1%})")
    print()

    # ---- Failure breakdown using new detection_info fields (when present)
    # Keys: num_keypoints_detected (raw HRNet), num_keypoints_solver
    # (post-outlier-filter, post-ID-map), num_non_ground, num_lines_*
    if failed_solver and "num_keypoints_solver" in failed_solver[0][1]:
        det_kp = [v["num_keypoints_detected"] for _, v in failed_solver]
        slv_kp = [v["num_keypoints_solver"] for _, v in failed_solver]
        ng = [v["num_non_ground"] for _, v in failed_solver]
        det_ln = [v["num_lines_detected"] for _, v in failed_solver]
        slv_ln = [v["num_lines_solver"] for _, v in failed_solver]
        print("Failed-solver frames — detection counts:")
        print(f"  num_keypoints_detected  {fmt_pct(percentiles(det_kp))}")
        print(f"  num_keypoints_solver    {fmt_pct(percentiles(slv_kp))}")
        print(f"  num_non_ground          {fmt_pct(percentiles(ng))}")
        print(f"  num_lines_detected      {fmt_pct(percentiles(det_ln))}")
        print(f"  num_lines_solver        {fmt_pct(percentiles(slv_ln))}")

        # Buckets by likely cause
        buckets = Counter()
        for _, v in failed_solver:
            kpd = v["num_keypoints_detected"]
            kps = v["num_keypoints_solver"]
            if kpd == 0:
                buckets["zero-keypoints (HRNet)"] += 1
            elif kpd < 4:
                buckets["det<4 (HRNet under-recall)"] += 1
            elif kps < 4:
                buckets["det>=4 but solver<4 (ID-map / outlier filter dropped)"] += 1
            else:
                buckets["det>=4 & solver>=4 (Zhang refused)"] += 1
        print("Failed-solver root cause buckets:")
        for reason, n in buckets.most_common():
            print(f"  {reason:50s} {n:4d}  ({n/len(failed_solver):.1%})")
        print()

    # ---- Per-frame detection counts (calibrated-only — solver doesn't
    # report counts for failed frames in the current schema) ---------------
    nkp = [v["num_keypoints"] for _, v in calibrated]
    nln = [v["num_lines"] for _, v in calibrated]
    rep_err = [v["reprojection_error"] for _, v in calibrated]
    inliers = [v["inliers"] for _, v in calibrated]
    total_pts = [v["total_points"] for _, v in calibrated]

    print("Detection counts (calibrated frames only):")
    print(f"  num_keypoints  {fmt_pct(percentiles(nkp))}")
    print(f"  num_lines      {fmt_pct(percentiles(nln))}")
    print(f"  inliers        {fmt_pct(percentiles(inliers))}")
    print(f"  total_points   {fmt_pct(percentiles(total_pts))}")

    rej_err = [v["reprojection_error"] for _, v in rejected_by_sanity if "reprojection_error" in v]
    print()
    print("Reprojection error:")
    print(f"  calibrated  {fmt_pct(percentiles(rep_err))}")
    if rej_err:
        print(f"  rejected    {fmt_pct(percentiles(rej_err))}")

    # ---- Mode / ransac selection --------------------------------------
    mode_ct = Counter(v.get("mode") for _, v in calibrated)
    ransac_ct = Counter(v.get("ransac") for _, v in calibrated)
    print()
    print("Solver-selected mode:")
    for mode, n in mode_ct.most_common():
        print(f"  {mode:14s}  {n:4d}  ({n/max(1,len(calibrated)):.1%})")
    print("Solver-selected ransac threshold:")
    for r, n in sorted(ransac_ct.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"  {str(r):>6s}  {n:4d}  ({n/max(1,len(calibrated)):.1%})")

    # ---- Recover fx, pos from H, run is_good_camera ----------------------
    fx_series: list[tuple[int, float]] = []   # (frame_idx, fx)
    good_ct = Counter()
    fx_recovery_failed = 0
    for fidx, _ in calibrated:
        Hm = homs.get(fidx)
        if Hm is None:
            fx_recovery_failed += 1
            continue
        fx = estimate_fx_from_H(Hm, cx, cy)
        pos = camera_pos_from_H(Hm, fx, cx, cy) if fx is not None else None
        ok, reason = is_good_camera(fx, pos)
        good_ct[reason] += 1
        if fx is not None:
            fx_series.append((fidx, fx))

    fx_vals = [fx for _, fx in fx_series]
    print()
    print("Recovered intrinsics (calibrated frames):")
    if fx_vals:
        print(f"  fx          {fmt_pct(percentiles(fx_vals))}")
        # fx temporal variability
        fx_series.sort()
        if len(fx_series) >= 2:
            diffs = [abs(fx_series[i][1] - fx_series[i - 1][1]) for i in range(1, len(fx_series))]
            print(f"  |Δfx|       {fmt_pct(percentiles(diffs))}  (frame-to-frame)")
            # Relative jump
            rel = [
                abs(fx_series[i][1] - fx_series[i - 1][1])
                / max(1.0, 0.5 * (fx_series[i][1] + fx_series[i - 1][1]))
                for i in range(1, len(fx_series))
            ]
            print(f"  |Δfx|/fx    {fmt_pct(percentiles(rel))}")
    print(f"  is_good_camera passes: {good_ct.get('ok', 0)}/{len(calibrated)}")
    if good_ct:
        for reason, n in good_ct.most_common():
            if reason == "ok":
                continue
            print(f"    {reason}: {n}")
    print()

    # ---- Bottleneck verdict ----------------------------------------------
    print("--- Verdict ---")
    if not calibrated:
        print("  No calibrated frames — solver/detection completely broken.")
    else:
        kp_med = float(np.median(nkp))
        ln_med = float(np.median(nln))
        ground_pct = mode_ct.get("ground_plane", 0) / len(calibrated)
        rejected_pct = len(rejected_by_sanity) / max(1, n_total)
        # Heuristic verdicts
        if kp_med < 6:
            print(f"  [DETECTION]   median keypoints/frame is low ({kp_med:.0f}); HRNet recall is the bottleneck.")
        if ln_med < 2:
            print(f"  [DETECTION]   median lines/frame is low ({ln_med:.0f}); line head underused.")
        if ground_pct > 0.3:
            print(f"  [GEOMETRY]    {ground_pct:.0%} of frames fall back to ground_plane mode (no usable NON_GROUND points → fx ill-conditioned).")
        if rejected_pct > 0.05:
            print(f"  [SANITY]      {rejected_pct:.0%} of frames killed by fx-prior sanity gate; review intrinsics.fx and factor bounds.")
        if fx_vals:
            rel_p95 = np.percentile(
                [abs(fx_series[i][1] - fx_series[i-1][1]) / max(1.0, 0.5 * (fx_series[i][1] + fx_series[i-1][1])) for i in range(1, len(fx_series))],
                95,
            ) if len(fx_series) >= 2 else 0
            if rel_p95 > 0.20:
                print(f"  [STABILITY]   p95 |Δfx|/fx = {rel_p95:.0%} → fx jitters across frames; consider time-domain smoothing.")
        print(f"  Calibration rate: {len(calibrated)/n_total:.1%} (interpolated +{len(interpolated)/n_total:.1%}).")

    # ---- Plot dashboard --------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        ax = axes[0, 0]
        ax.hist(nkp, bins=range(0, max(nkp) + 2) if nkp else [0, 1])
        ax.set_title(f"num_keypoints (calibrated, n={len(nkp)})")
        ax.set_xlabel("keypoints"); ax.set_ylabel("frames")

        ax = axes[0, 1]
        ax.hist(nln, bins=range(0, max(nln) + 2) if nln else [0, 1])
        ax.set_title(f"num_lines (calibrated, n={len(nln)})")
        ax.set_xlabel("lines"); ax.set_ylabel("frames")

        ax = axes[0, 2]
        ax.hist(rep_err, bins=30)
        ax.set_title("reprojection_error (px)")
        ax.set_xlabel("rep_err"); ax.set_ylabel("frames")
        ax.axvline(5.0, color="r", linestyle="--", alpha=0.5, label="th=5px")
        ax.legend()

        ax = axes[1, 0]
        modes = list(mode_ct.keys())
        ax.bar([str(m) for m in modes], [mode_ct[m] for m in modes])
        ax.set_title("mode selected"); ax.set_ylabel("frames")

        ax = axes[1, 1]
        ranks = sorted(ransac_ct.keys(), key=lambda x: (x is None, x))
        ax.bar([str(r) for r in ranks], [ransac_ct[r] for r in ranks])
        ax.set_title("ransac threshold selected"); ax.set_ylabel("frames")

        ax = axes[1, 2]
        if fx_series:
            xs = [f for f, _ in fx_series]
            ys = [v for _, v in fx_series]
            ax.scatter(xs, ys, s=8)
            ax.set_title("fx(t)  recovered from H")
            ax.set_xlabel("frame_idx"); ax.set_ylabel("fx (px)")
        else:
            ax.text(0.5, 0.5, "no fx recovered", ha="center", va="center")
            ax.set_title("fx(t)")

        plt.tight_layout()
        out = fr / "diagnostic.png"
        fig.savefig(out, dpi=110)
        print(f"\nDashboard saved: {out}")


if __name__ == "__main__":
    main()
