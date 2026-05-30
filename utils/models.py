import numpy as np
from scipy.linalg import block_diag, toeplitz
from sklearn.utils.extmath import randomized_svd
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

from load_data import img_to_2d, load_data

class GLMdenoiser:
    """Implements the method of GLMdenoise"""
    
    def __init__(self, images, events, t1_img, tr):
        self.images = images
        self.events = events
        self.t1_img = t1_img
        self.tr = tr
        self.condition_names = []

    def _build_stim_trains(self, events_df, n_scans):
        """Build binary stimulus trains at TR resolution (NOT convolved)."""
        
        trains = []
        
        for trial_type in self.condition_names:
            sub = events_df[events_df["trial_type"].astype(str) == trial_type]
            train = np.zeros(n_scans)
            
            onsets = np.round(sub["onset"].to_numpy() / self.tr).astype(int)
            durations = np.maximum(1, np.round(sub["duration"].to_numpy() / self.tr).astype(int))
            indices = np.concatenate([np.arange(o, o + d) for o, d in zip(onsets, durations)])
            
            indices = indices[(indices >= 0) & (indices < n_scans)]

            train[indices] = 1.0
            trains.append(train)
        
        return trains
    
    @staticmethod
    def _convolve_hrf(stim_train, hrf):
        """Full convolution truncated back to the original singal length."""
        return np.convolve(stim_train, hrf, mode="full")[:len(stim_train)]
    
    def _task_matrix_from_hrf(self, stim_trains, hrf):
        """Build (n_scans, n_cond) task design matrix with `hrf` convolution."""
        all_conv = [self._convolve_hrf(st, hrf) for st in stim_trains]
        return np.column_stack(all_conv)
    
    def _build_polynomial_regressors(self, n_scans):
        """
        Build Legendre-like polynomial basis, whose degree grows with run length:
        one polynomial every 2 minutes.
        """
        L_minutes = (n_scans * self.tr) / 60.0
        max_degree = int(round(L_minutes / 2.0))
        t = np.linspace(-1.0, 1.0, n_scans)
        return np.column_stack([t ** d for d in range(max_degree + 1)])
    
    @staticmethod
    def _residualize(Q, A):
        """Project out the Q from the A"""
        return A - Q @ (Q.T @ A)
    
    def _get_q_aug(self, Q_poly, noise_pcs, n_noise):
        if n_noise <= 0 or noise_pcs is None:
            return Q_poly
        return np.column_stack([Q_poly, noise_pcs[:, :n_noise]])
    
    def build_all_runs(self):
        """Load images and assemble per-run_dicts.
        
        Each dict contains:
          Y           (n_scans, n_voxels)    BOLD data
          mean_signal (n_voxels)            temporal mean
          n_scans     int
          stim_trains list[n_cond]           raw binary trains at TR
          X_poly      (n_scans, n_poly)      polynomial nuisance basis
          Q_poly      (n_scans, n_poly)      orthonormal basis for X_poly (QR decomposition)
        """
        
        all_conditions = set()
        for events_df in self.events:
            all_conditions.update(events_df["trial_type"].astype(str).unique())
        self.condition_names = sorted(all_conditions)
        
        runs = []

        for img, events_df in zip(self.images, self.events):
            Y, mean_signal, n_scans = img_to_2d(img)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            
            stim_trains = self._build_stim_trains(events_df, n_scans)
            X_poly = self._build_polynomial_regressors(n_scans)
            
            # Using the Frisch–Waugh–Lovell trick 
            # with residualizing once and fitting only the task part later
            Q_poly, _ = np.linalg.qr(X_poly, mode="reduced")

            runs.append({
                "Y": Y,
                "mean_signal": mean_signal,
                "n_scans": n_scans,
                "stim_trains": stim_trains,
                "X_poly": X_poly,
                "Q_poly": Q_poly,
            })

        return runs
    
    def _estimate_betas(self, runs, hrf):
        """
        Fix HRF, estimate betas with FWL (Frisch-Waugh-Lovell) trick and OLS
        
        Returns betas (n_cond, V) and per-voxel R-squared (V)
        """
        
        X_res_blocks = []
        Y_res_blocks = []
        
        for run in runs:
            X_task = self._task_matrix_from_hrf(run["stim_trains"], hrf)
            Q = run["Q_poly"]
            
            # Continuation of the Frisch–Waugh–Lovell trick
            X_res_blocks.append(self._residualize(Q, X_task))
            Y_res_blocks.append(self._residualize(Q, run["Y"]))

        X_res = np.vstack(X_res_blocks)
        Y_res = np.vstack(Y_res_blocks)
        
        betas, _, _, _ = np.linalg.lstsq(X_res, Y_res, rcond=None)
        Y_hat = X_res @ betas
        
        ss_res = np.sum((Y_res - Y_hat) ** 2, axis=0)
        ss_tot = np.sum((Y_res - Y_res.mean(axis=0)) ** 2, axis=0)
        r2 = np.zeros_like(ss_tot)
        mask = ss_tot > 0
        r2[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
        
        return betas, r2

    def _estimate_hrf_fir(self, runs, betas, best_voxels, hrf_len):
        """Fix betas, estimate HRF via FIR basis (polynomial trends projected out)."""
        
        A_list = []
        y_list = []
        
        for run in runs:
            Q = run["Q_poly"]
            Y_res = self._residualize(Q, run["Y"])
            
            for v in best_voxels:
                beta_v = betas[:, v]
                
                # Weighted neural signal
                N_v = beta_v @ np.asarray(run["stim_trains"])
                
                # FIR matrix
                A = toeplitz(N_v, np.zeros(hrf_len))
                
                # Remove polynomials from FIR matrix
                A_res = self._residualize(Q, A)
                
                A_list.append(A_res)
                y_list.append(Y_res[:, v])
        
        A_all = np.vstack(A_list)
        y_all = np.concatenate(y_list)
        h, _, _, _ = np.linalg.lstsq(A_all, y_all, rcond=None)
        return h
    
    @staticmethod
    def _r2_between(h1, h2):
        """R2 of h2 as predictor of h1."""
        ss_tot = np.sum((h1 - h1.mean())**2)
        if ss_tot == 0:
            return 1.0
        # Else return r2 clipped to 0, 1 as in paper
        return np.clip(1.0 - np.sum((h1 - h2) ** 2) / ss_tot, 0.0, 1.0)
                
    def estimate_hrf(self, runs, initial_hrf, hrf_len=None, max_iter=50):
        """
        Estimates optimal HRF by Step 2. from the paper (iterative linear fitting):
        
        1. Fix HRF - OLS betas
        2. Select 50 best voxels with highest R2
        3. Fix betas - OLS HRF with FIR 
        4. End when R2 > 0.99
        Fall back to initial_hrf if R2(initial, fitted) < 0.5
        """
        
        if hrf_len is None:
            hrf_len = int(round(20.0 / self.tr))
        
        if len(initial_hrf) < hrf_len:
            initial_hrf = np.pad(initial_hrf, (0, hrf_len - len(initial_hrf)))
        else:
            initial_hrf = initial_hrf[:hrf_len]
        
        current_hrf = initial_hrf.copy()
        
        for i in range(max_iter):
            # 1. Fix HRF - estimate betas and polynomials
            betas, r2 = self._estimate_betas(runs, current_hrf)
            
            # 2. Select 50 best voxels
            num_voxels = min(50, r2.shape[0])
            best_voxels = np.argsort(r2)[-num_voxels:]
            prev_hrf = current_hrf.copy()
            
            # 3. Fix betas - estimate HRF
            current_hrf = self._estimate_hrf_fir(runs, betas, best_voxels, hrf_len)
            
            # 4. Convergence check
            r2_between = self._r2_between(prev_hrf, current_hrf)
            if r2_between > 0.99:
                break
            print(f"  HRF iter {i+1}: R2 value: {r2_between:.4f}")
        
        # Normalization of HRF (peak at 1.0)
        peak = np.max(current_hrf)
        if peak > 0:
            current_hrf /= peak

        # Fallback if poorly estimated (from the paper)
        if self._r2_between(initial_hrf, current_hrf) < 0.50:
            print("  HRF poorly estimated - reverting to initial HRF")
            current_hrf = initial_hrf.copy()
            peak = np.max(current_hrf)
            if peak > 0:
                current_hrf /= peak

        return current_hrf
    
    def _cross_val_r2(self, runs, hrf, noise_pcs_per_run, n_noise=0):
        """Cross Validation leaving one run out and calculating R2."""
        
        XtX_list = []
        XtY_list = []
        X_res_list = []
        Y_res_list = []
        
        for i, run in enumerate(runs):
            X_task = self._task_matrix_from_hrf(run["stim_trains"], hrf)
            pcs_i = noise_pcs_per_run[i] if noise_pcs_per_run is not None else None
            Q = self._get_q_aug(run["Q_poly"], pcs_i, n_noise)
            
            X_res = self._residualize(Q, X_task)
            Y_res = self._residualize(Q, run["Y"])
            
            X_res_list.append(X_res)
            Y_res_list.append(Y_res)
            XtX_list.append(X_res.T @ X_res)
            XtY_list.append(X_res.T @ Y_res)
        
        XtX_total = sum(XtX_list)
        XtY_total = sum(XtY_list)
        
        Y_pred_blocks = []
        Y_true_blocks = []
        
        for i in range(len(runs)):
            XtX_train = XtX_total - XtX_list[i]
            XtY_train = XtY_total - XtY_list[i]
            betas = np.linalg.solve(XtX_train, XtY_train)
            
            Y_pred_blocks.append(X_res_list[i] @ betas)
            Y_true_blocks.append(Y_res_list[i])
        
        Y_pred = np.vstack(Y_pred_blocks)
        Y_true = np.vstack(Y_true_blocks)

        ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
        ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2, axis=0)
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    
    def _select_noise_pool(self, runs, cv_r2):
        """
        Noise pool: voxels where
          1. cross-validated R² < 0  (not task-related)
          2. mean signal > 0.5 x 99th percentile  (inside brain)
        """
        
        mean_signal = np.mean(np.vstack([run["mean_signal"] for run in runs]), axis=0)
        threshold = 0.5 * np.percentile(mean_signal, 99)
        mask = (cv_r2 < 0.0) & (mean_signal > threshold)
        return np.where(mask)[0]

    def _compute_noise_pcs(self, runs, noise_voxels, max_components):
        """
        Per-run: extract noise-pool voxels, project out poly trends,
        unit-normalise each time-series, compute PCs via randomised SVD.
        """
        
        pcs_per_run = []
        
        for run in runs:
            Y_noise = run["Y"][:, noise_voxels]
            Q = run["Q_poly"]

            Y_res = self._residualize(Q, Y_noise)

            # Unit-normalise each noise voxel time-series
            norms = np.linalg.norm(Y_res, axis=0, keepdims=True)
            Y_norm = Y_res / np.where(norms > 0, norms, 1.0)

            k = min(max_components, Y_norm.shape[1], Y_norm.shape[0] - 1)
            U, _, _ = randomized_svd(Y_norm, n_components=k, random_state=0)

            # Zero-pad columns if noise pool has fewer voxels than max_components
            if k < max_components:
                U = np.pad(U, ((0, 0), (0, max_components - k)))

            pcs_per_run.append(U)  # (T, max_components), orthonormal, perp to Q_poly

        return pcs_per_run

    def _select_n_noise_regressors(self, runs, hrf, noise_pcs_per_run, max_noise):
        """
        Steps 6-7: sweep n = 0..max_noise noise regressors, cross-validate
        each, then pick the minimum n that captures >= 95% of the maximum
        performance gain over task-responsive voxels.
        """
        
        print(f"  Sweeping 0..{max_noise} noise regressors:")
        r2_by_n = []
        
        for n in range(max_noise + 1):
            r2 = self._cross_val_r2(runs, hrf, noise_pcs_per_run, n_noise=n)
            r2_by_n.append(r2)
            print(f"    n={n:2d}  median R²={np.median(r2):.4f}")

        r2_array = np.array(r2_by_n)

        task_mask = np.any(r2_array > 0.0, axis=0)
        if not np.any(task_mask):
            print("  Warning: no task voxels found; using n=0.")
            return 0, np.zeros(max_noise + 1), r2_array

        median_r2 = np.array([
            np.median(r2_array[n, task_mask]) 
            for n in range(max_noise + 1)
        ])

        # Minimum n within 5% of max improvement (step 7 criterion)
        r2_baseline = median_r2[0]
        improvement = np.max(median_r2) - r2_baseline
        threshold = r2_baseline + 0.95 * improvement
        crosses = np.where(median_r2 >= threshold)[0]
        optimal_n = int(crosses[0]) if len(crosses) > 0 else 0

        return optimal_n, median_r2, r2_array

    def _final_fit(self, runs, hrf, noise_pcs_per_run, n_noise, n_boot=100):
        """
        Step 8: fit final model and estimate error bars via bootstrapping.

        Speed trick: precompute XtX and XtY per run once; each bootstrap
        sample is just a weighted sum of those.
          -> O(n_cond^2 x V) per iteration  vs  O(T x n_cond x V) naive.

        Returns
        -------
        beta_median : (n_cond, V)  median across bootstraps, in % BOLD
        beta_se     : (n_cond, V)  0.5 x 68% range, in % BOLD
        """
        XtX_list, XtY_list, mean_signals = [], [], []

        for i, run in enumerate(runs):
            X_task = self._task_matrix_from_hrf(run["stim_trains"], hrf)
            pcs_i = noise_pcs_per_run[i] if noise_pcs_per_run is not None else None
            Q = self._get_q_aug(run, pcs_i, n_noise)

            X_res = self._residualize(Q, X_task)
            Y_res = self._residualize(Q, run["Y"])

            XtX_list.append(X_res.T @ X_res)
            XtY_list.append(X_res.T @ Y_res)
            mean_signals.append(run["mean_signal"])

        XtX_arr = np.array(XtX_list)   # (R, n_cond, n_cond)
        XtY_arr = np.array(XtY_list)   # (R, n_cond, V)
        mean_signal = np.mean(mean_signals, axis=0)  # (V,)

        R = len(runs)
        n_cond = XtY_arr.shape[1]
        n_voxels = XtY_arr.shape[2]
        betas_boots = np.empty((n_boot, n_cond, n_voxels))

        rng = np.random.default_rng(seed=0)
        for b in range(n_boot):
            sample = rng.choice(R, R, replace=True)
            counts = np.bincount(sample, minlength=R).astype(float)

            # Weighted normal equations - no matrix stacking needed
            XtX = np.einsum("r,rij->ij", counts, XtX_arr)
            XtY = np.einsum("r,rij->ij", counts, XtY_arr)

            try:
                betas_boots[b] = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                betas_boots[b], _, _, _ = np.linalg.lstsq(XtX, XtY, rcond=None)

        beta_median = np.median(betas_boots, axis=0)
        lo = np.percentile(betas_boots, 16, axis=0)
        hi = np.percentile(betas_boots, 84, axis=0)
        beta_se = 0.5 * (hi - lo)

        # Convert to % BOLD signal change
        safe_mean = np.where(mean_signal > 0, mean_signal, 1.0)
        beta_median_pct = beta_median / safe_mean * 100.0
        beta_se_pct = beta_se / safe_mean * 100.0

        return beta_median_pct, beta_se_pct

    def full_workflow(self, runs_num=None, max_noise: int = 10, n_boot: int = 100):
        """
        Run the complete GLMdenoise pipeline (steps 1-8).
        
        Returns dict with keys:
          condition_names       list[str]
          beta                  (n_cond, V)  % BOLD, median over bootstrap
          beta_se               (n_cond, V)  % BOLD, 0.5x68% range
          hrf                   (hrf_len,)   estimated HRF
          cv_r2_baseline        (V,)         CV-R2 without noise regressors
          cv_r2_final           (V,)         CV-R2 with optimal noise count
          noise_voxels          (N,)         indices of noise-pool voxels
          optimal_n_noise       int
          median_r2_by_noise    (max_noise+1,)
          runs                  list of run dicts
        """
        
        # --- load data ------------------------------------------------
        print("Building runs...")
        runs = self.build_all_runs()
        if runs_num is not None:
            runs = runs[:runs_num]

        # --- Step 1: initial HRF (Glover canonical) ------------------
        initial_hrf = glover_hrf(self.tr, oversampling=1)

        # --- Step 2: iterative HRF estimation ------------------------
        print("\nStep 2: Estimating HRF...")
        optimal_hrf = self.estimate_hrf(runs, initial_hrf)

        # --- Step 3: baseline cross-validated R² ---------------------
        print("\nStep 3: Computing cross-validated R²...")
        cv_r2_baseline = self._cross_val_r2(runs, optimal_hrf)
        print(f"  Median CV-R² (baseline): {np.median(cv_r2_baseline):.4f}")

        # --- Step 4: noise pool --------------------------------------
        print("\nStep 4: Selecting noise pool...")
        noise_voxels = self._select_noise_pool(runs, cv_r2_baseline)
        print(f"  Noise pool: {len(noise_voxels)} voxels")
        if len(noise_voxels) == 0:
            raise RuntimeError(
                "Noise pool is empty — check data quality or brain masking."
            )

        # --- Step 5: PCA on noise pool -------------------------------
        print("\nStep 5: Computing noise PCs...")
        noise_pcs_per_run = self._compute_noise_pcs(
            runs, noise_voxels, max_components=max_noise
        )

        # --- Steps 6-7: select number of noise regressors ------------
        print("\nSteps 6-7: Cross-validating noise regressor counts...")
        optimal_n, median_r2_by_noise, _ = self._select_n_noise_regressors(
            runs, optimal_hrf, noise_pcs_per_run, max_noise=max_noise
        )
        print(f"  Optimal number of noise regressors: {optimal_n}")

        # --- Step 3 (final): CV-R² with selected noise ---------------
        cv_r2_final = self._cross_val_r2(
            runs, optimal_hrf, noise_pcs_per_run, n_noise=optimal_n
        )
        print(f"  Median CV-R² (final):    {np.median(cv_r2_final):.4f}")

        # --- Step 8: bootstrap final fit -----------------------------
        print(f"\nStep 8: Bootstrap final fit ({n_boot} iterations)...")
        beta_median, beta_se = self._final_fit(
            runs, optimal_hrf, noise_pcs_per_run, optimal_n, n_boot=n_boot
        )

        return {
            "condition_names": self.condition_names,
            "beta": beta_median,
            "beta_se": beta_se,
            "hrf": optimal_hrf,
            "cv_r2_baseline": cv_r2_baseline,
            "cv_r2_final": cv_r2_final,
            "noise_voxels": noise_voxels,
            "optimal_n_noise": optimal_n,
            "median_r2_by_noise": median_r2_by_noise,
            "runs": runs,
        }
    

if __name__ == "__main__":
    main_dir = "/Users/jakubkempa/Documents/magisterka/data/ds000105_R2.0.2"
    subject = "sub-1"
    all_data = load_data(main_dir, subject=subject)
    
    glm = GLMdenoiser(*all_data)
    fit = glm.full_workflow()
    
    breakpoint()
