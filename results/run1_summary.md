# Run 1 (A) summary — kept for run-to-run variance reporting

Same code/config as the final run for experiment (A). Open3D's RANSAC is multi-threaded, so
success counts vary slightly between runs. Copied verbatim from the run-1 console output.

| method | noise σ (%diag) | success |
|---|---|---|
| icp_only | 0.2 / 0.5 / 1.0 | 0/24 · 0/24 · 0/24 |
| ransac1+icp | 0.2 / 0.5 / 1.0 | 11/24 · 1/24 · 0/24 |
| ransac5+icp | 0.2 / 0.5 / 1.0 | 18/24 · 7/24 · 0/24 |
| ransac5+icp_unoriented | 0.2 / 0.5 / 1.0 | 10/24 · 0/24 · 1/24 |
