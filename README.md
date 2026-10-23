# hampel-ahi-swe

MATLAB code for the paper

> P. T. T. Ha, P. D. Phong, T. D. Tan, L. Q. Hai, **"Two-Stage Filtering Strategy for Doppler Velocity Field Processing in Shear Wave Elastography"**, *Revista Internacional de Métodos Numéricos para Cálculo y Diseño en Ingeniería (RIMNI)*, under revision (manuscript ID 91941), 2026.

The pipeline is: 2-D FDTD viscoelastic (Kelvin–Voigt) shear-wave simulation → composite noise (white Gaussian + sparse bipolar impulses) → **Stage 1: temporal Hampel filter** → **Stage 2: adaptive LMS noise canceller with spatial weight transfer** → Algebraic Helmholtz Inversion (AHI) → shear modulus μ and shear viscosity η maps.

## Requirements
* MATLAB R2019b or later (tested on MATLAB Online, R2024)
* Signal Processing Toolbox (`hampel`, `medfilt1`, `dftmtx`)
* Image Processing Toolbox (`wiener2`, `ssim`)

No other dependencies. Each script is self-contained apart from the LMS filter functions in the same folder.

## Repository layout

| Folder / file | Purpose | Where it appears in the paper |
|---|---|---|
| `original_paper_code/Hample_LMS_RIMNI_Final.m` | Full original pipeline (FDTD → noise → Hampel → LMS → AHI) and the figures of the first submission | Sections 2–3, Figs. 4–8 |
| `original_paper_code/LMS_filter.m` | LMS adaptive noise canceller, zero initial weights (first column of each row) | Eqs. (9)–(13) |
| `original_paper_code/LMS_filter_2024.m` | Same filter accepting the converged weights of the previous location (spatial weight transfer) | Eq. (14), Fig. 3 |
| `revision_experiments/LMS_filter_wt.m` | Refactored LMS/ANC function used by all revision experiments: identical update rule, explicit `w_init`/`w_final` for weight transfer, filters the whole sequence | Section 2.3.2 |
| `revision_experiments/ablation_montecarlo.m` | Matched ablation over 7 configurations, 10 Monte-Carlo runs, mean ± SD | **Table 4** |
| `revision_experiments/sigma_sweep.m` | Sweep of the Gaussian noise level σ_G | **Table 5** |
| `revision_experiments/impulse_sweep.m` | Sweep of impulse density p and amplitude A_I | **Table 6** |
| `revision_experiments/clean_distortion.m` | Distortion of the noise-free signal by each filter (amplitude bias, phase error, waveform RMSE, false-replacement rate) | **Table 3** |
| `revision_experiments/regen_figures.m` | Re-exports Fig. 4 (noise histograms) and Fig. 8 (μ maps + MAE) with seed 2025 | Figs. 4, 8 |
| `revision_experiments/khoi6_thay_the.m` | Drop-in replacement for the LMS block of the original script with a `USE_TRANSFER` switch | — |

## Reproducing the results

```matlab
cd revision_experiments
ablation_montecarlo   % Table 4  (~10 min on MATLAB Online)
sigma_sweep           % Table 5
impulse_sweep         % Table 6
clean_distortion      % Table 3
regen_figures         % Figs. 4 and 8 -> ./figures/
```
All scripts fix the random seed (`rng(2025)`), so the numbers in the paper are reproduced exactly.

## Simulation and filter parameters (Tables 1–2 of the paper)

| Parameter | Value |
|---|---|
| Grid | 80 × 80 nodes, Δx = Δy = 1 mm |
| Time step / samples | Δt = 0.08 ms, N_t = 4000 |
| Excitation | f₀ = 100 Hz harmonic source at the corner node, amplitude A = 0.005 |
| Background / inclusion | μ = 6 / 9 kPa, η = 1.2 / 1.8 Pa·s, inclusion radius 15 mm |
| Gaussian noise | σ_G = 1.2 × 10⁻⁶ m/s |
| Impulses | bipolar, A_I = 1 × 10⁻³ m/s, density p = 0.2 % |
| Hampel | half-window k = 5 (K = 11), threshold λ = 5 |
| LMS | order M = 2, step μ = 0.2, normalised update w ← w + μ·e·x/‖x‖ |
| Laplacian | 4·del2(V)/Δx² |
| Seed / runs | 2025 / 10 |

## License
MIT — see `LICENSE`.

## Citation
If you use this code, please cite the paper above. A Zenodo DOI for the archived release is given in the paper's *Availability of Data and Materials* section.
