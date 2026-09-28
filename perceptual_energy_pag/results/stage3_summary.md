# Stage 3 summary (auto-generated numbers)

Pooled step-wise correlation corr(Δf_y, -ΔE_y) [95% grouped bootstrap CI over base images]:

| Factor | CE | AT |
|---|---:|---:|
| ALL | -0.068 [-0.094, -0.039] | -0.082 [-0.111, -0.055] |
| shape | -0.223 [-0.292, -0.145] | -0.289 [-0.384, -0.197] |
| contrast | -0.153 [-0.314, 0.018] | -0.249 [-0.417, -0.070] |
| brightness | -0.046 [-0.096, 0.046] | -0.118 [-0.231, 0.152] |
| texture | 0.064 [-0.064, 0.209] | 0.106 [-0.013, 0.228] |
| position | 0.002 [-0.037, 0.038] | -0.030 [-0.065, 0.003] |
| rotation | 0.004 [-0.081, 0.086] | -0.145 [-0.216, -0.066] |

Edge-only / contrast-only energy (factor=ALL): corr(Δf,-ΔE_edge), corr(Δf,-ΔE_contrast):
- CE: edge -0.036 [-0.059, -0.007], contrast -0.188 [-0.218, -0.154]; median per-trajectory corr(f,-E)=-0.022
- AT: edge -0.013 [-0.039, 0.011], contrast -0.383 [-0.418, -0.346]; median per-trajectory corr(f,-E)=-0.094

Paired differences in pooled corr vs CE:
- AT - CE: -0.014 [-0.042, 0.011]

Local scaling β (test_aligned, all classes):

| Model | mean β | median β | CV(β) | frac β>0 | regression f=-βE+C: β_reg | R² |
|---|---:|---:|---:|---:|---:|---:|
| CE | -0.06961 | -0.06444 | 1.17 | 0.14 | -0.1688 | 0.121 |
| AT | -0.02581 | -0.02278 | 0.65 | 0.00 | -0.04669 | 0.074 |

Per-class regression R²: CE/0: β_reg=-0.101, R²=0.094, CV(β)=1.22; CE/1: β_reg=-0.12, R²=0.227, CV(β)=0.50; CE/2: β_reg=-0.22, R²=0.211, CV(β)=1.58; AT/0: β_reg=-0.0577, R²=0.105, CV(β)=0.39; AT/1: β_reg=-0.02, R²=0.158, CV(β)=0.46; AT/2: β_reg=-0.0599, R²=0.216, CV(β)=0.43

Gradient-flow (mean over images; final step and average over steps):
- CE: mean cos on classifier path=-0.012, on energy path=-0.027; final normalised distance=0.743 (√2≈1.414 = orthogonal); f_y gain along energy path=-1.250
- AT: mean cos on classifier path=-0.186, on energy path=-0.073; final normalised distance=1.059 (√2≈1.414 = orthogonal); f_y gain along energy path=-0.207
