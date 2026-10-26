# -*- coding: utf-8 -*-
"""Simulated partial dental scans -> global (FPFH+RANSAC) + fine (ICP) registration.

Input : public CC BY 4.0 3D model "Permanent Dentition" (DundeeDental, Sketchfab).
Output: results/metrics.json, results/summary.md, results/figure.png

Synthetic experiment on a public model -- NOT a clinical intraoral scan.
Units are the model's own (normalized) units, not millimetres; errors are reported
as a fraction of the model's bounding-box diagonal.

(A) scan -> reference model: each partial scan is placed at an unknown pose and
    registered back to the full model. Error is measured against the known
    ground-truth pose (point RMSE + rotation error).
(B) scan -> scan stitching: consecutive scans differ by a small unknown motion
    (random walk); adjacent scans are registered pairwise by ICP and chained, the way
    multiple hand-held scans are merged. Accumulated drift is measured per scan.
"""
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "DENTITION_GLB",
    r"C:\coding\github-repository\cover_letter\references\3d_models\osstem_teeth"
    r"\permanent_dentition\permanent_dentition.glb",
)
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
reg = o3d.pipelines.registration

N_VIEWS = 8
NOISE_LEVELS = [0.002, 0.005, 0.010]   # Gaussian sigma as a fraction of bbox diagonal
SEEDS = [0, 1, 2]
DROPOUT = 0.20                          # fraction of visible points removed
OUTLIER_FRAC = 0.01                     # uniform outliers added (fraction of scan points)
FOV_FRAC = 0.28                         # scan patch radius as a fraction of bbox diagonal
MAX_ROT_DEG = 180.0                     # unknown initial pose
MAX_TRANS_FRAC = 0.5                    # of bbox diagonal
VOXEL_FRAC = 0.01
RANSAC_ITERS = 1_000_000
RANSAC_CONF = 0.9999
RESTARTS = 5                            # multi-hypothesis: keep the best ICP fitness
SUCCESS_ROT_DEG = 5.0
SUCCESS_RMSE_FRAC = 0.01                # ground-truth point RMSE < 1% of bbox diagonal
N_STITCH = 16                           # (B) consecutive scans swept along one side of the arch
STITCH_MAX_ROT_DEG = 5.0                # (B) per-step motion between consecutive scans (about scan centroid)
STITCH_MAX_TRANS_FRAC = 0.02
STITCH_ICP_SCALES = (4.0, 2.0, 0.8)     # coarse-to-fine ICP correspondence distance, x voxel
STITCH_SEEDS = [0, 1, 2, 3, 4]

METHODS = ("icp_only", "ransac1+icp", f"ransac{RESTARTS}+icp", f"ransac{RESTARTS}+icp_unoriented")


# ---------------------------------------------------------------- model / geometry
def load_model():
    scene = trimesh.load(MODEL_PATH, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        parts.append(g)
    mesh = trimesh.util.concatenate(parts)
    centroids = np.array([p.vertices.mean(0) for p in parts])
    o3m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
    )
    o3m.compute_vertex_normals()
    return o3m, centroids


def arch_frame(centroids):
    """PCA of tooth centroids: smallest-variance axis ~ occlusal (vertical) direction."""
    c = centroids.mean(0)
    _, _, vt = np.linalg.svd(centroids - c)
    return c, vt[0], vt[1], vt[2]


def random_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(rng.uniform(0, max_deg)))


def pose_errors(T_est, T_true, pts_gt):
    """Rotation error (deg) and ground-truth point RMSE (model units)."""
    R_err = T_est[:3, :3] @ T_true[:3, :3].T
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))))
    P = np.c_[pts_gt, np.ones(len(pts_gt))]
    d = (P @ T_est.T - P @ T_true.T)[:, :3]
    return rot, float(np.sqrt(np.mean(np.sum(d ** 2, 1))))


# ---------------------------------------------------------------- scan simulation
def make_view_targets(dense, center, ax_w, ax_d, ax_v, n):
    """n targets ordered along the arch (by angle around the arch center)."""
    pts = np.asarray(dense.points)
    rel = pts - center
    ang = np.arctan2(rel @ ax_d, rel @ ax_w)
    targets = []
    for a in np.linspace(ang.min(), ang.max(), n):
        idx = np.argsort(np.abs(ang - a))[:2000]
        targets.append(pts[idx].mean(0))
    return np.array(targets)


def camera_for(target, center, ax_v, diag, view_idx):
    radial = target - center
    radial -= ax_v * (radial @ ax_v)
    radial /= np.linalg.norm(radial) + 1e-9
    vsign = 1.0 if view_idx % 2 == 0 else -1.0          # alternate occlusal sides
    view_dir = radial * 0.6 + ax_v * 0.8 * vsign
    return target + view_dir / np.linalg.norm(view_dir) * diag * 0.8


def simulate_scan(dense, target, center, ax_v, diag, view_idx, rng, noise, voxel):
    """Visible points from a scanner pose, limited FOV, dropout, noise, outliers.
    Returns (oriented-normal scan, unoriented copy); both share the same points."""
    cam = camera_for(target, center, ax_v, diag, view_idx)
    _, idx = dense.hidden_point_removal(cam, diag * 100)
    vp = np.asarray(dense.select_by_index(idx).points)
    vp = vp[np.linalg.norm(vp - target, axis=1) < diag * FOV_FRAC]
    vp = vp[rng.random(len(vp)) > DROPOUT]
    vp = vp + rng.normal(scale=noise * diag, size=vp.shape)
    n_out = int(len(vp) * OUTLIER_FRAC)
    vp = np.vstack([vp, rng.uniform(vp.min(0), vp.max(0), size=(n_out, 3))])

    oriented = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vp))
    oriented.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    oriented.orient_normals_towards_camera_location(cam)   # the scanner knows its own viewpoint
    unoriented = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vp))
    return oriented, unoriented


# ---------------------------------------------------------------- registration
def features(pc, voxel):
    d = pc.voxel_down_sample(voxel)
    if not d.has_normals():
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return d, reg.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


def icp_refine(src, tgt, voxel, init):
    s = src.voxel_down_sample(voxel * 0.5)
    t = tgt.voxel_down_sample(voxel * 0.5)
    for p in (s, t):
        if not p.has_normals():
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel, max_nn=30))
    return reg.registration_icp(s, t, voxel * 0.8, init, reg.TransformationEstimationPointToPlane(),
                                reg.ICPConvergenceCriteria(max_iteration=100))


def global_register(src, tgt, src_feat, tgt_feat, voxel, restarts, seed_base):
    """FPFH+RANSAC (restarts hypotheses) -> ICP; keep the hypothesis with the best ICP fitness."""
    s_down, s_f = src_feat
    t_down, t_f = tgt_feat
    dist = voxel * 1.5
    best = None
    for r in range(restarts):
        o3d.utility.random.seed(seed_base + r)
        ransac = reg.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_f, t_f, False, dist, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist)],
            reg.RANSACConvergenceCriteria(RANSAC_ITERS, RANSAC_CONF),
        )
        icp = icp_refine(src, tgt, voxel, ransac.transformation)
        if best is None or icp.fitness > best.fitness:
            best = icp
    return best


def icp_multiscale(src, tgt, voxel, init):
    """Coarse-to-fine point-to-plane ICP (larger correspondence distance first)."""
    s = src.voxel_down_sample(voxel * 0.5)
    t = tgt.voxel_down_sample(voxel * 0.5)
    for p in (s, t):
        if not p.has_normals():
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel, max_nn=30))
    T = init
    res = None
    for sc in STITCH_ICP_SCALES:
        res = reg.registration_icp(s, t, voxel * sc, T, reg.TransformationEstimationPointToPlane(),
                                   reg.ICPConvergenceCriteria(max_iteration=60))
        T = res.transformation
    return res


def chamfer(a, b):
    da = np.asarray(a.compute_point_cloud_distance(b))
    db = np.asarray(b.compute_point_cloud_distance(a))
    return float(da.mean() + db.mean()) / 2


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    o3d.utility.random.seed(0)                      # deterministic point sampling
    mesh, centroids = load_model()
    center, ax_w, ax_d, ax_v = arch_frame(centroids)
    ext = mesh.get_max_bound() - mesh.get_min_bound()
    diag = float(np.linalg.norm(ext))
    voxel = diag * VOXEL_FRAC
    dense = mesh.sample_points_uniformly(200_000)
    model_pc = mesh.sample_points_uniformly(60_000)             # carries outward mesh normals
    model_unoriented = o3d.geometry.PointCloud(model_pc.points)  # ablation: normals re-estimated
    tgt_feat = features(model_pc, voxel)
    tgt_feat_unor = features(model_unoriented, voxel)
    targets = make_view_targets(dense, center, ax_w, ax_d, ax_v, N_VIEWS)

    stitch_only = "--stitch-only" in sys.argv       # reuse saved (A) results, redo (B) + figure
    rows, scans_by_setting = [], {}
    for noise in NOISE_LEVELS:
        for seed in SEEDS:
            if stitch_only and (noise, seed) != (NOISE_LEVELS[0], 0):
                continue
            rng = np.random.default_rng(1000 * seed + int(noise * 1e4))
            scans = []
            for k, tgt in enumerate(targets):
                scan, scan_unor = simulate_scan(dense, tgt, center, ax_v, diag, k, rng, noise, voxel)
                T_rand = np.eye(4)
                T_rand[:3, :3] = random_rotation(rng, MAX_ROT_DEG)
                T_rand[:3, 3] = rng.uniform(-1, 1, 3) * MAX_TRANS_FRAC * diag
                moved = o3d.geometry.PointCloud(scan).transform(T_rand)
                moved_unor = o3d.geometry.PointCloud(scan_unor).transform(T_rand)
                T_true = np.linalg.inv(T_rand)
                gt_pts = np.asarray(scan.points)
                scans.append((scan, moved, T_true))
                if stitch_only:
                    continue

                src_feat = features(moved, voxel)
                sb = 100_000 * seed + 100 * k
                results = {}
                tt = time.time(); results["icp_only"] = icp_refine(moved, model_pc, voxel, np.eye(4)); t_icp = time.time() - tt
                tt = time.time(); results["ransac1+icp"] = global_register(moved, model_pc, src_feat, tgt_feat, voxel, 1, sb); t1 = time.time() - tt
                tt = time.time(); results[METHODS[2]] = global_register(moved, model_pc, src_feat, tgt_feat, voxel, RESTARTS, sb); t5 = time.time() - tt
                results[METHODS[3]] = global_register(moved_unor, model_unoriented, features(moved_unor, voxel),
                                                      tgt_feat_unor, voxel, RESTARTS, sb)
                times = {"icp_only": t_icp, "ransac1+icp": t1, METHODS[2]: t5, METHODS[3]: None}
                for m in METHODS:
                    res = results[m]
                    rot, rmse = pose_errors(res.transformation, T_true, gt_pts)
                    rows.append({
                        "noise_frac": noise, "seed": seed, "view": k, "method": m,
                        "n_points": len(scan.points), "fitness": float(res.fitness),
                        "inlier_rmse": float(res.inlier_rmse), "rot_err_deg": rot,
                        "gt_rmse": rmse, "gt_rmse_frac": rmse / diag, "time_s": times[m],
                        "success": bool(rot < SUCCESS_ROT_DEG and rmse / diag < SUCCESS_RMSE_FRAC),
                    })
            scans_by_setting[(noise, seed)] = scans
            print(f"noise={noise} seed={seed} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- (B) scan-to-scan stitching: consecutive scans differ by a SMALL unknown motion
    # (random walk, each step <= STITCH_MAX_ROT_DEG / STITCH_MAX_TRANS_FRAC), as between
    # consecutive frames of a hand-held scan. Pairwise ICP from identity, chained; scan 0 = anchor.
    # Two strategies: "pairwise" registers scan k to scan k-1 only; "to_map" registers scan k to the
    # union of all scans stitched so far (initialised from the previous pose).
    def run_stitch(base, T_world, mode):
        moved = [o3d.geometry.PointCloud(s).transform(T) for s, T in zip(base, T_world)]
        poses, chain = [np.eye(4)], []
        stitched = o3d.geometry.PointCloud(moved[0])
        for k in range(1, len(base)):
            if mode == "pairwise":
                icp = icp_multiscale(moved[k], moved[k - 1], voxel, np.eye(4))
                T_k = poses[k - 1] @ icp.transformation
            else:
                icp = icp_multiscale(moved[k], stitched.voxel_down_sample(voxel * 0.5), voxel, poses[k - 1])
                T_k = icp.transformation
            poses.append(T_k)
            stitched += o3d.geometry.PointCloud(moved[k]).transform(T_k)
            rot, rmse = pose_errors(T_k, np.linalg.inv(T_world[k]), np.asarray(base[k].points))
            chain.append({"view": k, "pair_fitness": float(icp.fitness),
                          "cum_rot_err_deg": rot, "cum_rmse_frac": rmse / diag})
        gt_union = o3d.geometry.PointCloud()
        for b in base:
            gt_union += b
        ch = chamfer(stitched.voxel_down_sample(voxel * 0.5), gt_union.voxel_down_sample(voxel * 0.5))
        return {"chain": chain, "chamfer_frac": ch / diag}

    stitching = {}
    targets_b = make_view_targets(dense, center, ax_w, ax_d, ax_v, N_STITCH)
    for noise_b in (NOISE_LEVELS[0], NOISE_LEVELS[1]):
        for mode in ("pairwise", "to_map"):
            runs = []
            for s_seed in STITCH_SEEDS:
                rng_s = np.random.default_rng(777 + 31 * s_seed + int(noise_b * 1e4))
                # view_idx=0 for every scan -> all scans look from the same side, so neighbours overlap
                base = [simulate_scan(dense, t, center, ax_v, diag, 0, rng_s, noise_b, voxel)[0] for t in targets_b]
                rng_b = np.random.default_rng(424242 + s_seed)
                T_world = [np.eye(4)]
                for k in range(1, len(base)):
                    c = np.asarray(base[k].points).mean(0)
                    R = random_rotation(rng_b, STITCH_MAX_ROT_DEG)
                    dT = np.eye(4)
                    dT[:3, :3] = R                          # small rotation about the scan's own centroid
                    dT[:3, 3] = c - R @ c + rng_b.uniform(-1, 1, 3) * STITCH_MAX_TRANS_FRAC * diag
                    T_world.append(T_world[-1] @ dT)
                r = run_stitch(base, T_world, mode)
                r["seed"] = s_seed
                runs.append(r)
            finals = [r["chain"][-1]["cum_rmse_frac"] for r in runs]
            stitching[f"{noise_b}|{mode}"] = {
                "noise_frac": noise_b, "mode": mode, "runs": runs,
                "final_rmse_frac_median": float(np.median(finals)),
                "final_rmse_frac_min": float(np.min(finals)), "final_rmse_frac_max": float(np.max(finals)),
                "chamfer_frac_median": float(np.median([r["chamfer_frac"] for r in runs])),
                "median_curve": [float(np.median([r["chain"][i]["cum_rmse_frac"] for r in runs]))
                                 for i in range(N_STITCH - 1)],
                "min_curve": [float(np.min([r["chain"][i]["cum_rmse_frac"] for r in runs])) for i in range(N_STITCH - 1)],
                "max_curve": [float(np.max([r["chain"][i]["cum_rmse_frac"] for r in runs])) for i in range(N_STITCH - 1)],
            }
            print(f"stitch noise={noise_b} mode={mode}: final RMSE %diag median "
                  f"{np.median(finals)*100:.2f} (min {np.min(finals)*100:.2f}, max {np.max(finals)*100:.2f})", flush=True)

    runtime_a = round(time.time() - t0, 1)
    if stitch_only:
        with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as f:
            prev = json.load(f)
        rows = prev["rows"]
        runtime_a = prev["meta"].get("runtime_a_s", prev["meta"].get("runtime_s"))

    # ---- aggregate
    def agg(method, noise):
        sub = [r for r in rows if r["method"] == method and r["noise_frac"] == noise]
        ok = [r for r in sub if r["success"]]
        med = lambda key, xs: float(np.median([r[key] for r in xs])) if xs else None
        tsub = [r["time_s"] for r in sub if r["time_s"] is not None]
        return {"n": len(sub), "n_success": len(ok), "success_rate": len(ok) / len(sub),
                "median_rot_err_deg_success": med("rot_err_deg", ok),
                "median_gt_rmse_frac_success": med("gt_rmse_frac", ok),
                "median_fitness_success": med("fitness", ok),
                "mean_time_s": float(np.mean(tsub)) if tsub else None}

    summary = {m: {str(n): agg(m, n) for n in NOISE_LEVELS} for m in METHODS}
    meta = {
        "model": "Permanent Dentition by DundeeDental (Sketchfab), CC BY 4.0",
        "bbox_extent_units": ext.round(4).tolist(), "bbox_diag_units": diag,
        "n_views": N_VIEWS, "noise_levels_frac": NOISE_LEVELS, "seeds": SEEDS,
        "dropout": DROPOUT, "outlier_frac": OUTLIER_FRAC, "fov_frac": FOV_FRAC,
        "max_rot_deg": MAX_ROT_DEG, "max_trans_frac": MAX_TRANS_FRAC, "voxel_units": voxel,
        "ransac_iters": RANSAC_ITERS, "ransac_conf": RANSAC_CONF, "restarts": RESTARTS,
        "success_criterion": f"rot < {SUCCESS_ROT_DEG} deg and GT point RMSE < {SUCCESS_RMSE_FRAC:.0%} of diag",
        "scan_points_mean": float(np.mean([r["n_points"] for r in rows if r["method"] == "icp_only"])),
        "n_stitch": N_STITCH, "stitch_step_max_rot_deg": STITCH_MAX_ROT_DEG,
        "stitch_step_max_trans_frac": STITCH_MAX_TRANS_FRAC, "stitch_icp_scales_x_voxel": list(STITCH_ICP_SCALES),
        "open3d": o3d.__version__, "runtime_a_s": runtime_a,
    }
    result = {"meta": meta, "summary": summary, "stitching": stitching, "rows": rows}
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fmt = lambda v, s="{:.3f}": "-" if v is None else s.format(v)
    lines = ["| method | noise σ (%diag) | success | median rot err (°) | median GT RMSE (%diag) | mean time (s) |",
             "|---|---|---|---|---|---|"]
    for m in METHODS:
        for n in NOISE_LEVELS:
            a = summary[m][str(n)]
            gtr = None if a["median_gt_rmse_frac_success"] is None else a["median_gt_rmse_frac_success"] * 100
            lines.append(f"| {m} | {n*100:.1f} | {a['n_success']}/{a['n']} ({a['success_rate']:.0%}) | "
                         f"{fmt(a['median_rot_err_deg_success'])} | {fmt(gtr)} | {fmt(a['mean_time_s'], '{:.2f}')} |")
    lines += ["", f"Stitching ({N_STITCH} scans swept along one side, coarse-to-fine ICP; per-step motion <= "
                  f"{STITCH_MAX_ROT_DEG:.0f}° / {STITCH_MAX_TRANS_FRAC:.0%} diag; {len(STITCH_SEEDS)} seeds):", "",
              "| noise σ (%diag) | strategy | final cum. GT RMSE (%diag): median [min, max] | "
              "RMSE after 7 scans (%diag), median | stitched-vs-GT Chamfer (%diag), median |",
              "|---|---|---|---|---|"]
    for s in stitching.values():
        lines.append(f"| {s['noise_frac']*100:.1f} | {s['mode']} | {s['final_rmse_frac_median']*100:.3f} "
                     f"[{s['final_rmse_frac_min']*100:.3f}, {s['final_rmse_frac_max']*100:.3f}] | "
                     f"{s['median_curve'][6]*100:.3f} | {s['chamfer_frac_median']*100:.3f} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    make_figure(model_pc, scans_by_setting[(NOISE_LEVELS[0], 0)], summary, stitching, diag, voxel, tgt_feat)
    print(f"total {time.time()-t0:.0f}s")


def make_figure(model_pc, scans, summary, stitching, diag, voxel, tgt_feat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    M = np.asarray(model_pc.points)
    M = M[rng.choice(len(M), 6000, replace=False)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(scans)))
    sub = lambda P, n: P[rng.choice(len(P), min(n, len(P)), replace=False)]

    fig = plt.figure(figsize=(16, 7.2))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(*M.T, s=0.3, c="lightgray")
    for (scan, _, _), c in zip(scans, colors):
        ax.scatter(*sub(np.asarray(scan.points), 1500).T, s=0.6, color=c)
    ax.set_title("(a) 8 simulated partial scans (colored)\non the reference model (gray)")

    scan0, moved0, T_true0 = scans[3]
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.scatter(*M.T, s=0.3, c="lightgray")
    ax.scatter(*sub(np.asarray(moved0.points), 2500).T, s=0.6, c="crimson")
    ax.set_title("(b) one scan at an unknown pose\n(random rotation up to 180°)")

    # same hypothesis seeds as the reported trial (noise level 0, seed 0, view 3)
    best = global_register(moved0, model_pc, features(moved0, voxel), tgt_feat, voxel, RESTARTS, 100 * 3)
    aligned = o3d.geometry.PointCloud(moved0).transform(best.transformation)
    rot, rmse = pose_errors(best.transformation, T_true0, np.asarray(scan0.points))
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.scatter(*M.T, s=0.3, c="lightgray")
    ax.scatter(*sub(np.asarray(aligned.points), 2500).T, s=0.6, c="crimson")
    ax.set_title(f"(c) after FPFH+RANSAC -> ICP\nrot err {rot:.2f}°, GT RMSE {rmse/diag*100:.2f}% diag")
    for a in fig.axes:
        a.set_axis_off()
        a.view_init(elev=35, azim=-60)

    xs = [n * 100 for n in NOISE_LEVELS]
    ax = fig.add_subplot(2, 3, 4)
    for m, st, col in zip(METHODS, ("o--", "s-", "^-", "x:"), ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        ax.plot(xs, [summary[m][str(n)]["success_rate"] * 100 for n in NOISE_LEVELS], st, color=col, label=m)
    ax.set_xlabel("noise σ (% of bbox diagonal)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(d) success rate vs noise  (24 trials / point)\nsuccess: rot<5°, GT RMSE<1% diag")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    for m, st, col in zip(METHODS[1:3], ("s-", "^-"), ("tab:orange", "tab:green")):
        pts = [(x, summary[m][str(n)]["median_gt_rmse_frac_success"] * 100, summary[m][str(n)]["n_success"])
               for x, n in zip(xs, NOISE_LEVELS) if summary[m][str(n)]["median_gt_rmse_frac_success"] is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], st, color=col, label=m)
        for x, y, ns in pts:
            ax.annotate(f"n={ns}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=col)
    ax.set_xlabel("noise σ (% of bbox diagonal)")
    ax.set_ylabel("median GT point RMSE (% diag)")
    ax.set_title("(e) residual error of SUCCESSFUL trials only\n(n = number of successes; small n at 0.5%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
    ks = np.arange(1, N_STITCH)
    for mode, col in (("pairwise", "tab:purple"), ("to_map", "tab:green")):
        s = stitching[f"{NOISE_LEVELS[0]}|{mode}"]
        ax.plot(ks, np.array(s["median_curve"]) * 100, "o-", color=col, label=f"{mode} (median)")
        ax.fill_between(ks, np.array(s["min_curve"]) * 100, np.array(s["max_curve"]) * 100, color=col, alpha=0.15)
    ax.set_xlabel("scan index (scan 0 = anchor)")
    ax.set_ylabel("cumulative GT point RMSE (% diag)")
    ax.set_title(f"(f) stitching {N_STITCH} scans, σ {NOISE_LEVELS[0]*100:.1f}% diag\n"
                 f"{len(STITCH_SEEDS)} seeds, band = min-max")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Simulated partial dental scans -> global + fine registration  "
                 "(public CC BY 4.0 model, synthetic - not a clinical scan)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print("saved figure")


if __name__ == "__main__":
    main()
