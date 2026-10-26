# Small config sweep for the global stage (noise 0.002, seed 0, 8 views).
import time
import numpy as np
import open3d as o3d
import run_registration as rr

reg = o3d.pipelines.registration
mesh, centroids = rr.load_model()
center, ax_w, ax_d, ax_v = rr.arch_frame(centroids)
diag = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
dense = mesh.sample_points_uniformly(200_000)
model_pc = mesh.sample_points_uniformly(60_000)
targets = rr.make_view_targets(dense, center, ax_w, ax_d, ax_v, rr.N_VIEWS)
print("model_pc has normals:", model_pc.has_normals())


def gt_rmse(T_est, T_true, pts):
    P = np.c_[pts, np.ones(len(pts))]
    return float(np.sqrt(np.mean(np.sum(((P @ T_est.T) - (P @ T_true.T))[:, :3] ** 2, 1))))


def feat(pc, voxel, fr, keep_normals):
    d = pc.voxel_down_sample(voxel)
    if not keep_normals or not d.has_normals():
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = reg.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * fr, max_nn=100))
    return d, f


def run(cfg):
    rng = np.random.default_rng(7)
    voxel = diag * cfg["voxel"]
    ok, rots = 0, []
    t0 = time.time()
    for k, tgt in enumerate(targets):
        scan = rr.simulate_scan(dense, tgt, center, ax_v, diag, k, rng, 0.002)
        if cfg["orient"]:
            # scanner knows its own viewpoint: orient normals toward it (before the unknown pose is applied)
            radial = tgt - center; radial -= ax_v * (radial @ ax_v); radial /= np.linalg.norm(radial)
            vd = radial * 0.6 + ax_v * 0.8 * (1 if k % 2 == 0 else -1); vd /= np.linalg.norm(vd)
            scan.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
            scan.orient_normals_towards_camera_location(tgt + vd * diag * 0.8)
        T_rand = np.eye(4)
        T_rand[:3, :3] = rr.random_rotation(rng, 180)
        T_rand[:3, 3] = rng.uniform(-1, 1, 3) * 0.5 * diag
        moved = o3d.geometry.PointCloud(scan).transform(T_rand)
        T_true = np.linalg.inv(T_rand)
        s, sf = feat(moved, voxel, cfg["fr"], cfg["orient"])
        t, tf = feat(model_pc, voxel, cfg["fr"], cfg["orient"])
        dist = voxel * 1.5
        res = reg.registration_ransac_based_on_feature_matching(
            s, t, sf, tf, cfg["mutual"], dist, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist)],
            reg.RANSACConvergenceCriteria(cfg["iters"], 0.9999))
        icp = rr.icp_refine(moved, model_pc, diag * 0.01, res.transformation)
        rot = rr.rt_errors(icp.transformation, T_true, diag)[0]
        e = gt_rmse(icp.transformation, T_true, np.asarray(scan.points)) / diag
        rots.append(rot)
        ok += (rot < 5 and e < 0.01)
    print(f"{cfg}  success {ok}/8  median rot {np.median(rots):.1f}  ({time.time()-t0:.0f}s)")


for cfg in [
    dict(voxel=0.01, fr=5, mutual=True, iters=100_000, orient=False),   # original
    dict(voxel=0.01, fr=5, mutual=False, iters=1_000_000, orient=False),
    dict(voxel=0.01, fr=5, mutual=False, iters=1_000_000, orient=True),
    dict(voxel=0.02, fr=5, mutual=False, iters=1_000_000, orient=True),
    dict(voxel=0.015, fr=8, mutual=False, iters=1_000_000, orient=True),
]:
    run(cfg)
