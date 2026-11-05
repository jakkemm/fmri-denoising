# fMRI Denoising with Statistical Models

My master's thesis project comparing three statistical approaches to denoising task-based functional MRI (fMRI): **standard General Linear Model (GLM), PCA-assisted GLM, and Independent Component Analysis (ICA)**.

fMRI brain scans are noisy (head motion, scanner drift, heartbeat and more), and many methods exist to clean them up. This project asks whether more elaborate denoising actually helps: each method is judged by how well it predicts task-related brain activity in scans it has not seen. Along the way it combines statistical modeling, time-series analysis, dimensionality reduction, cross-validation, and computational benchmarking.

**Dataset:** [Visual Object Recognition (OpenNeuro ds000105)](https://openneuro.org/datasets/ds000105)

**Thesis:** *Selected Statistical Methods in fMRI Image Denoising* ([full text](thesis/))

## Key findings

- **PCA-assisted GLM had the highest CV prediction accuracy for all six participants**, but only by a small margin (1.62% vs. 1.48% for standard GLM), at roughly 200 times the fitting time.
- **Standard GLM - the best trade-off between accuracy and computational cost.**
- **ICA-based denoising performed worse than standard GLM for four of the six participants.**

The main conclusion of the analysis is that **more elaborate denoising does not automatically produce meaningfully better results**: the benefit depends on the experimental design, the preprocessing, the baseline model and how components are selected.

## Results at a glance

| Method | Mean per-subject median outer-CV $R^2$ | Mean full-data fitting time | Mean outer-CV runtime |
| --- | ---: | ---: | ---: |
| Standard GLM | 1.48% | **1.7 s** | 16.8 s |
| PCA-assisted GLM | **1.62%** | 356.1 s | 3,137.2 s |
| ICA-based denoising | 0.76% | 64.3 s | 695.9 s |

*$R^2$ is computed per voxel on held-out runs, summarized as the median over voxels for each participant, then averaged over the six participants. Absolute values are small because most of the variance in a voxel's time series is noise rather than task response, so the comparison between methods is what matters. Runtimes were measured on a MacBook Air (M1, 2020) with 8 GB of RAM.*

### Selected results

**Cross-validated prediction accuracy**

![Median outer-CV $R^2$ for the three methods across participants](results/plots/group/median_r2.png)

*PCA-assisted GLM achieved the highest median prediction accuracy for every participant, although the improvement over standard GLM was small.*

**Computational cost.**

![Runtime comparison across denoising methods](results/plots/group/runtime.png)

*The additional model selection and component-analysis steps substantially increased runtime without delivering a proportional improvement in predictive performance.*

**Face-versus-house contrast $t$-maps**

![Comparison of face-versus-house contrast $t$-maps across methods](results/plots/individual/sub-3_t_face_vs_house.png)

*Contrast t-maps for participant 3, showing the face > house comparison. Standard GLM and PCA-assisted GLM produce similar spatial patterns; ICA-based denoising introduces more noticeable differences.*

## Detailed findings

The outer leave-one-run-out CV results showed that PCA-assisted GLM produces consistent (and small) improvement in task-response prediction. The mean of the per-subject median $R^2$ values was 1.62% for this method, compared with 1.48% for standard GLM. The improvement did not translate into consistently more stable beta coefficients, measured by jackknife signal-to-noise ratio.

Standard GLM achieved very similar predictive performance, while being much less computationally heavy, making it the most favorable overall trade-off in this analysis, as well as a strong comparative baseline.

ICA-based denoising had the smallest mean median $R^2$ of 0.76%, with poor results for subjects 1-4 and small improvements over standard GLM for participants 5 and 6. Its results additionally showed greater subject dependence and differences in the estimated spatial patterns. The possible explanations include uncertain component classification and the absence of explicit pre-processing motion correction.

## Methods

### 1. Standard GLM

The baseline model estimates task-related BOLD responses using a General Linear Model. Task regressors are constructed by convolving stimulus timing with a canonical haemodynamic response function (HRF). An intercept and discrete cosine transform (DCT) regressors account for low-frequency drift.

Regression coefficients are estimated using **Feasible Generalized Least Squares (FGLS)** with a run-specific AR(1) temporal covariance model and prewhitening.

### 2. PCA-assisted GLM

Inspired by GLMdenoise [[1]](#references), this method extends the baseline model with data-derived nuisance regressors.

1. Fit the baseline GLM and identify a noise pool using inner CV task-prediction accuracy and a mean-signal criterion.
2. Apply PCA to the detrended noise-pool time series for each run.
3. Select the number of components through inner LORO CV, using a 95%-of-maximum-improvement rule and a maximum of 20 components.
4. Refit the GLM with the selected principal components as additional nuisance regressors.

This approach tests whether structured variability captured directly from the data can improve estimation of task-related responses.

### 3. ICA-based denoising

The ICA pipeline uses probabilistic PCA and Bayesian model-order selection to determine a reduced-dimensional representation, followed by whitening and FastICA to estimate independent components.

A permutation-based procedure classifies components as task-related or nuisance. The nuisance contribution is reconstructed and subtracted from the data, after which task-related responses are estimated using the same FGLS framework as the other methods.

This provides a comparison between regression-based nuisance modeling and denoising based on estimated independent components.

## Evaluation

All three approaches are evaluated using a common **leave-one-run-out cross-validation framework**. For each fold, one functional run is held out for testing while the remaining runs are used to estimate the model. This evaluates prediction on data not used during fitting.

The comparison includes:

- **Predictive performance:** voxelwise CV $R^2$, including changes relative to the standard GLM.
- **Coefficient stability:** jackknife SNR of task-related estimates across training sets.
- **Spatial interpretation:** task-related beta maps and face-versus-house contrast $t$-maps.
- **Computational efficiency:** full-data fitting time and total outer-CV runtime.

The dataset contains six participants performing a one-back repetition-detection task involving eight visual categories. Each participant has 11 or 12 functional runs, with 121 time points per run.

Because the dataset does not provide a noise-free reference signal, denoising quality is assessed indirectly through predictive performance, estimate stability, and spatial patterns rather than direct comparison against ground-truth clean images.

## Tech focus

**Statistical modeling · Linear algebra · Time-series analysis · PCA and ICA · Cross-validation · Model selection · Numerical computation · Data visualization**

## Repository structure

```text
fmri-denoising/
├── general_linear_model/             # Standard and PCA-assisted GLM
├── independent_component_analysis/   # ICA-based denoising
├── evaluation/                       # Metrics, maps, and comparisons
├── utils/                            # Data loading and utilities
├── thesis/                           # Master's thesis (LaTeX source and PDF)
├── figures/                          # Selected result figures used in this README
├── main.py                           # Pipelines and evaluation
└── plot_results.py                   # Result visualization
```

## Usage

Place the OpenNeuro dataset under `data/ds000105`, then run the desired pipeline.

```bash
# GLM + PCA-assisted denoising for one participant
python main.py --subject sub-1 --run_glm

# ICA-based denoising
python main.py --subject sub-1 --run_ica --ica_threshold 0

# Run the comparison and evaluation for one participant
python main.py --subject sub-1 --evaluate

# Generate figures
python plot_results.py --individual
python plot_results.py --group
python plot_results.py --components
```

Evaluating one participant takes roughly an hour in total on average (outer-CV runtime of about 52 minutes for PCA-assisted GLM and about 12 minutes for ICA, see the table above).

Results are serialized under `results/` (not included in the repo), with generated plots stored in `results/plots/`.

## Limitations and scope

This is an academic research implementation built for a master's thesis, definitely not a production-ready neuroimaging package. The analysis is limited to six participants and one visual block-design experiment. Explicit motion correction and regressors derived from external physiological recordings were not included in the dataset, thus also not included in the models.
These limitations, together with the component-selection procedures, matter when interpreting the results or generalizing them to other fMRI experiments.

## References

1. Kay, K. N., Rokem, A., Winawer, J., Dougherty, R. F., & Wandell, B. A. (2013). GLMdenoise: a fast, automated technique for denoising task-based fMRI data. *Frontiers in Neuroscience*, 7, 247.
2. Dataset: [OpenNeuro ds000105](https://openneuro.org/datasets/ds000105). See the dataset page for the recommended citation.
