| method | noise σ (%diag) | success | median rot err (°) | median GT RMSE (%diag) | mean time (s) |
|---|---|---|---|---|---|
| icp_only | 0.2 | 0/24 (0%) | - | - | 0.04 |
| icp_only | 0.5 | 0/24 (0%) | - | - | 0.04 |
| icp_only | 1.0 | 0/24 (0%) | - | - | 0.04 |
| ransac1+icp | 0.2 | 13/24 (54%) | 0.036 | 0.039 | 0.55 |
| ransac1+icp | 0.5 | 1/24 (4%) | 0.054 | 0.120 | 0.65 |
| ransac1+icp | 1.0 | 0/24 (0%) | - | - | 0.80 |
| ransac5+icp | 0.2 | 18/24 (75%) | 0.039 | 0.046 | 2.70 |
| ransac5+icp | 0.5 | 4/24 (17%) | 0.140 | 0.242 | 3.23 |
| ransac5+icp | 1.0 | 0/24 (0%) | - | - | 3.99 |
| ransac5+icp_unoriented | 0.2 | 13/24 (54%) | 0.054 | 0.067 | - |
| ransac5+icp_unoriented | 0.5 | 4/24 (17%) | 0.259 | 0.347 | - |
| ransac5+icp_unoriented | 1.0 | 0/24 (0%) | - | - | - |

Stitching (16 scans swept along one side, coarse-to-fine ICP; per-step motion <= 5° / 2% diag; 5 seeds):

| noise σ (%diag) | strategy | final cum. GT RMSE (%diag): median [min, max] | RMSE after 7 scans (%diag), median | stitched-vs-GT Chamfer (%diag), median |
|---|---|---|---|---|
| 0.2 | pairwise | 2.249 [1.686, 12.622] | 0.755 | 0.390 |
| 0.2 | to_map | 11.966 [7.070, 22.612] | 0.824 | 0.563 |
| 0.5 | pairwise | 10.386 [6.710, 15.008] | 1.648 | 0.881 |
| 0.5 | to_map | 7.922 [6.903, 79.296] | 7.816 | 0.994 |
