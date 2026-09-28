# Stage 1 summary (auto-generated numbers)

- split `train`: n=5000, class counts=[1667, 1667, 1666], clipped-pixel fraction=0.00000
- split `test`: n=1000, class counts=[334, 333, 333], clipped-pixel fraction=0.00000
- split `test_aligned`: n=600, class counts=[200, 200, 200], clipped-pixel fraction=0.00000
- split `proto`: n=900, class counts=[300, 300, 300], clipped-pixel fraction=0.00000

Transformation sensitivity (medium level; 300 base images; mean [95% CI]):

| Transformation | ΔPC (rel) | ΔGM (rel) | ΔPC (noise-floor units) | ΔGM (noise-floor units) | selectivity log2(PC/GM) |
|---|---:|---:|---:|---:|---:|
| shape | 0.928 [0.917, 0.940] | 0.840 [0.825, 0.855] | 7.89 | 3.71 | 0.17 [0.15, 0.19] |
| contrast | 0.041 [0.038, 0.045] | 0.226 [0.223, 0.229] | 0.32 | 0.99 | -3.02 [-3.22, -2.84] |
| brightness | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.00 | 0.00 | 0.36 [-0.03, 0.71] |
| brightness_gain(pixel) | 0.002 [0.002, 0.003] | 0.249 [0.248, 0.249] | 0.02 | 1.08 | -8.34 [-8.50, -8.16] |
| position | 0.876 [0.868, 0.882] | 0.873 [0.858, 0.888] | 7.38 | 3.84 | 0.02 [-0.01, 0.06] |
| rotation | 0.676 [0.664, 0.689] | 0.643 [0.621, 0.664] | 5.91 | 2.92 | 0.11 [0.07, 0.15] |
| texture | 0.232 [0.221, 0.243] | 0.268 [0.257, 0.278] | 1.92 | 1.08 | -0.25 [-0.30, -0.20] |
| size | 0.709 [0.696, 0.723] | 0.714 [0.697, 0.731] | 6.01 | 3.16 | 0.01 [-0.03, 0.04] |
| noise_resample | 0.129 [0.118, 0.140] | 0.238 [0.221, 0.254] | 0.99 | 1.00 | -0.98 [-1.00, -0.95] |

Observations (rule-based from the numbers above):
- shape: relatively larger response from **PC** (selectivity +0.17); hypothesised: PC -> consistent.
- contrast: relatively larger response from **GM** (selectivity -3.02); hypothesised: GM -> consistent.
- brightness offset: rel ΔPC=0.0000, rel ΔGM=0.0000. NOTE both operators are offset-invariant BY CONSTRUCTION (zero-DC log-Gabor, zero-sum Sobel); nonzero values come only from clipping.
- brightness gain (pixel x k): rel ΔPC=0.0024, rel ΔGM=0.2487.
- noise re-sample floor: rel ΔPC=0.1289, rel ΔGM=0.2380.
- nearest-prototype accuracy (chance 0.333): {"test_aligned": {"E_edge": 0.903333306312561, "E_contrast": 0.8066666722297668, "E_total": 0.9016666412353516, "pixel": 0.7066666483879089}, "test": {"E_edge": 0.3659999966621399, "E_contrast": 0.3479999899864197, "E_total": 0.3630000054836273, "pixel": 0.3449999988079071}}
