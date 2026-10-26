# Sweep 2: scan field-of-view x multi-hypothesis RANSAC (noise 0.002, 3 seeds x 8 views = 24 trials).
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
voxel = diag * 0.01


def gt_rmse(T_est, T_true, pts):
    P = np.c_[pts, np.ones(len(pts))]
    return float(np.sqrt(np.mean(np.sum(((P @ T_est.T) - (P @ T_true.T))[:, :3] ** 2, 1))))


def feat(pc):
    d = pc.voxel_down_sample(voxel)
    if not d.has_normals():
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return d, reg.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


t_down, t_f = feat(model_pc)


def run(fov, restarts):
    ok, rots = 0, []
    t0 = time.time()
    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed)
        for k, tgt in enumerate(targets):
            scan = rr.simulate_scan(dense, tgt, center, ax_v, diag, k, rng, 0.002, fov=fov)
            radial = tgt - center; radial -= ax_v * (radial @ ax_v); radial /= np.linalg.norm(radial)
            vd = radial * 0.6 + ax_v * 0.8 * (1 if k % 2 == 0 else -1); vd /= np.linalg.norm(vd)
            scan.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
            scan.orient_normals_towards_camera_location(tgt + vd * diag * 0.8)
            T_rand = np.eye(4)
            T_rand[:3, :3] = rr.random_rotation(rng, 180)
            T_rand[:3, 3] = rng.uniform(-1, 1, 3) * 0.5 * diag
            moved = o3d.geometry.PointCloud(scan).transform(T_rand)
            T_true = np.linalg.inv(T_rand)
            s, sf = feat(moved)
            best = None
            for r in range(restarts):
                o3d.utility.random.seed(1000 * seed + 10 * k + r)
                dist = voxel * 1.5
                res = reg.registration_ransac_based_on_feature_matching(
                    s, t_down, sf, t_f, False, dist, reg.TransformationEstimationPointToPoint(False), 3,
                    [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist)],
                    reg.RANSACConvergenceCriteria(1_000_000, 0.9999))
                icp = rr.icp_refine(moved, model_pc, voxel, res.transformation)
                if best is None or icp.fitness > best.fitness:
                    best = icp
            rot = rr.rt_errors(best.transformation, T_true, diag)[0]
            e = gt_rmse(best.transformation, T_true, np.asarray(scan.points)) / diag
            rots.append(rot)
            ok += (rot < 5 and e < 0.01)
    print(f"fov={fov} restarts={restarts}  success {ok}/24  median rot {np.median(rots):.2f}  ({time.time()-t0:.0f}s)")


for fov, restarts in [(0.28, 1), (0.28, 5), (0.38, 1), (0.38, 5)]:
    run(fov, restarts)
