# Sanity checks before trusting the full experiment.
import numpy as np
import open3d as o3d
import run_registration as rr

mesh, centroids = rr.load_model()
center, ax_w, ax_d, ax_v = rr.arch_frame(centroids)
diag = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
dense = mesh.sample_points_uniformly(200_000)
model_pc = mesh.sample_points_uniformly(60_000)
voxel = diag * 0.01
targets = rr.make_view_targets(dense, center, ax_w, ax_d, ax_v, rr.N_VIEWS)
print("center", center.round(3), "vertical axis", ax_v.round(3))
print("dist of model center from origin:", round(float(np.linalg.norm(center)), 3), "diag", round(diag, 3))

rng = np.random.default_rng(0)
scan = rr.simulate_scan(dense, targets[3], center, ax_v, diag, 3, rng, 0.002)
print("scan points:", len(scan.points))

# 1) identity pose: ICP from identity must succeed -> validates the metric
icp = rr.icp_refine(scan, model_pc, voxel, np.eye(4))
print("[identity] fitness", round(icp.fitness, 3), "rmse", round(icp.inlier_rmse, 4),
      "rot/trans", [round(x, 4) for x in rr.rt_errors(icp.transformation, np.eye(4), diag)])

# 2) metric sensitivity: a 1 degree rotation about the ORIGIN
R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, np.deg2rad(1.0)]))
T = np.eye(4); T[:3, :3] = R
P = np.asarray(scan.points)
gt_rmse = float(np.sqrt(np.mean(np.sum((P @ R.T - P) ** 2, 1))))
print("[1deg about origin] trans-component err frac:", round(rr.rt_errors(T, np.eye(4), diag)[2], 4),
      " point-RMSE frac:", round(gt_rmse / diag, 4))
