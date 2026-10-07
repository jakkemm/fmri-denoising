# fmri-denoising

Comparing three data-driven ways of denoising task fMRI: **standard GLM** **GLM + PCA noise regressors** and **ICA-based component classification**. Code behind my thesis (see `thesis/`).

fMRI signal is noisy. Motion, physiology and scanner drift all end up in the voxel time series, and they hide the task response. The implemented methods utilize the underlying experimental structure to create nuisance regressors, as well as learn the noise from the data itself. Finally they are judged on one question: does it predict the task response better in a run it has never seen?

Data: [OpenNeuro ds000105](https://openneuro.org/datasets/ds000105) (face / house / etc. viewing task), subjects 1-6.

## The most important methods

**GLM + PCA**

1. Fit a baseline GLM (task + drift) with FGLS (AR(1) whitening), leave-one-run-out
2. Pick a noise pool: voxels with cross-validated R² < 0 and enough mean signal
3. Run PCA on the detrended noise pool, per run. The top components become nuisance regressors
4. Choose the number of components `k` by cross-validated R² (95%-of-max-improvement rule, `k` up to 20)
5. Refit the GLM with the selected regressors

Full pseudocode in [`glm_pca_pseudocode.txt`](glm_pca_pseudocode.txt).

**ICA**
Decompose each run with ICA, use permutation testing (1000 permutations) to separate task components from nuisance ones, remove the nuisance part, refit.

## How it's judged

Everything is evaluated out-of-sample with leave-one-run-out cross-validation:

- voxelwise cross-validated R² (and the change vs. the standard GLM)
- jackknife SNR
- beta maps and face > house contrast t-maps
- runtime

## Layout

```
general_linear_model/            GLM + PCA pipeline
independent_component_analysis/  ICA pipeline
evaluation/                      CV metrics, maps, plots
utils/                           data loading, pickling
thesis/                          write-up
main.py                          run pipelines + evaluation
plot_results.py                  make figures
```

## Usage

Put the dataset in `data/ds000105`, then:

```bash
# GLM + PCA for one subject
python main.py --subject sub-1 --run_glm

# ICA (threshold optional)
python main.py --subject sub-1 --run_ica --ica_threshold 0

# full comparison for one subject
python main.py --subject sub-1 --evaluate

# figures
python plot_results.py --individual          # per-subject maps
python plot_results.py --group               # R², SNR, runtime across subjects
python plot_results.py --components          # selected PCA / ICA components
```

Results are pickled to `results/`, plots to `results/plots/`.

## Notes

- The code was written for my thesis rather than as a polished package.
- Fixed random seed (42) for the ICA runs.
