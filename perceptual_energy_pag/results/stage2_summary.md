# Stage 2 summary (auto-generated numbers)

Main comparison (primary split = test_aligned; means with 95% bootstrap CI):

| Model | Clean Acc | PGD Acc (8/255) | Cos(v,v̂) LS | R² force LS | R² force NNLS | ρ_edge LS | ρ_contrast LS | R² wrong-proto | R² random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.996 | 0.000 | 0.098 | 0.011 [0.011, 0.012] | 0.001 [0.000, 0.001] | 0.376 [0.358, 0.394] | 0.624 | 0.0134 | 0.00051 |
| AT | 0.828 | 0.395 | 0.337 | 0.122 [0.117, 0.127] | 0.000 [0.000, 0.000] | 0.304 [0.285, 0.324] | 0.696 | 0.1370 | 0.00049 |

Paired comparisons vs CE (mean difference [95% CI], Wilcoxon one-sided p for >):

- [test_aligned] AT - CE ls_r2: 0.1107 [0.1056, 0.1159]  frac>0=0.99  p=3.94e-100
- [test_aligned] AT - CE nnls_r2: -0.0004 [-0.0006, -0.0002]  frac>0=0.68  p=8.45e-01
- [test_aligned] AT - CE ls_rho_e: -0.0718 [-0.0939, -0.0490]  frac>0=0.37  p=1.00e+00
- [test_aligned] AT - CE nnls_rho_e: -0.1667 [-0.5000, 0.0000]  frac>0=0.00  p=1.00e+00
- [test_aligned] AT - CE cos_gsum: -0.2018 [-0.2105, -0.1930]  frac>0=0.01  p=1.00e+00
- [test_aligned] AT - CE r2_ls_wrong: 0.1237 [0.1196, 0.1278]  frac>0=1.00  p=3.10e-100
- [test_aligned] AT - CE coloc_absv_pc: 0.1533 [0.1454, 0.1613]  frac>0=0.93  p=3.24e-94
- [test] AT - CE ls_r2: 0.1395 [0.1354, 0.1433]  frac>0=1.00  p=1.67e-165
- [test] AT - CE nnls_r2: -0.0004 [-0.0005, -0.0003]  frac>0=0.68  p=9.87e-01
- [test] AT - CE ls_rho_e: -0.0923 [-0.1097, -0.0750]  frac>0=0.36  p=1.00e+00
- [test] AT - CE nnls_rho_e: 0.0000 [0.0000, 0.0000]  frac>0=0.00  p=nan
- [test] AT - CE cos_gsum: -0.2384 [-0.2440, -0.2327]  frac>0=0.00  p=1.00e+00
- [test] AT - CE r2_ls_wrong: 0.1389 [0.1351, 0.1426]  frac>0=1.00  p=1.66e-165
- [test] AT - CE coloc_absv_pc: 0.1683 [0.1623, 0.1743]  frac>0=0.96  p=1.31e-162

Zero-parameter cosines (test_aligned):

- CE: cos(v,g_e)=-0.0411, cos(v,g_c)=-0.0645, cos(v,g_e+g_c)=-0.0458, cos(v,g_pixel)=-0.0173, wrong-proto=-0.0504, sign-scrambled=0.0010, random=-0.0016
- AT: cos(v,g_e)=-0.2271, cos(v,g_c)=-0.3032, cos(v,g_e+g_c)=-0.2476, cos(v,g_pixel)=-0.1359, wrong-proto=-0.2792, sign-scrambled=0.0014, random=0.0006

Post-hoc sign-agnostic co-localisation cos(|v|, map) (test_aligned; shuffled = map of another image):

- CE: |v|~PC 0.414 (shuffled 0.336), |v|~GM 0.498 (shuffled 0.439), |v|~|g_e+g_c| 0.510, |v|~const 0.631
- AT: |v|~PC 0.567 (shuffled 0.340), |v|~GM 0.508 (shuffled 0.339), |v|~|g_e+g_c| 0.486, |v|~const 0.259
