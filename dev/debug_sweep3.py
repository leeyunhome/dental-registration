# Tuning sweep on SEPARATE seeds (50, 51) -- the reported experiment uses seeds 0,1,2.
# Tests: statistical outlier removal, noise-adaptive voxel, tight-threshold hypothesis selection.
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


def run(noise, sor, adaptive, select):
    voxel = diag * (max(0.01, 2.5 * noise) if adaptive else 0.01)
    tgt_feat = rr.features(model_pc, voxel)
    ok, t0 = 0, time.time()
    for seed in (50, 51):
        rng = np.random.default_rng(seed)
        for k, tgt in enumerate(targets):
            scan, _ = rr.simulate_scan(dense, tgt, center, ax_v, diag, k, rng, noise, diag * 0.01)
            T_rand = np.eye(4); T_rand[:3, :3] = rr.random_rotation(rng, 180)
            T_rand[:3, 3] = rng.uniform(-1, 1, 3) * 0.5 * diag
            moved = o3d.geometry.PointCloud(scan).transform(T_rand)
            src = moved
            if sor:
                src, _ = moved.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            s_feat = rr.features(src, voxel)
            best, best_score = None, -1
            for r in range(rr.RESTARTS):
                o3d.utility.random.seed(seed * 1000 + k * 10 + r)
                dist = voxel * 1.5
                ran = reg.registration_ransac_based_on_feature_matching(
                    s_feat[0], tgt_feat[0], s_feat[1], tgt_feat[1], False, dist,
                    reg.TransformationEstimationPointToPoint(False), 3,
                    [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist)],
                    reg.RANSACConvergenceCriteria(rr.RANSAC_ITERS, rr.RANSAC_CONF))
                icp = rr.icp_refine(src, model_pc, voxel, ran.transformation)
                if select == "fitness":
                    score = icp.fitness
                else:  # tight-threshold fitness: fraction of points within 0.3 voxel
                    score = reg.evaluate_registration(src, model_pc, voxel * 0.3, icp.transformation).fitness
                if score > best_score:
                    best, best_score = icp, score
            rot, rmse = rr.pose_errors(best.transformation, np.linalg.inv(T_rand), np.asarray(scan.points))
            ok += (rot < 5 and rmse / diag < 0.01)
    print(f"noise={noise} sor={sor} adaptive={adaptive} select={select}: {ok}/16 ({time.time()-t0:.0f}s)", flush=True)


for noise in (0.002, 0.005):
    run(noise, False, False, "fitness")      # current method
    run(noise, True, False, "fitness")
    run(noise, False, True, "fitness")
    run(noise, False, False, "tight")
    run(noise, True, True, "tight")
