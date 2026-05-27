import numpy as np
from scipy.linalg import block_diag, toeplitz
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

from tqdm import tqdm

from load_data import img_to_2d, load_data

class GLMdenoiser:
    """Implements the method of GLMdenoise"""
    
    def __init__(self, images, events, t1_img, tr):
        self.images = images
        self.events = events
        self.t1_img = t1_img
        self.tr = tr

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
        max_idx = len(stim_train)
        return np.convolve(stim_train, hrf, mode="full")[:max_idx]
    
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
    
    def build_all_runs(self):
        """Load images and assemble per-run_dicts.
        
        Returns
        -------
        runs : list of dicts, each containing
               - Y           (n_scans, n_voxels)  BOLD data
               - mean_signal (n_voxels,)           temporal mean
               - n_scans     int
               - stim_trains list[n_cond]           raw binary trains at TR
               - X_poly      (n_scans, n_poly)      polynomial nuisance basis
        condition_names : sorted list[str]
        """
        
        all_conditions = set()
        for events_df in self.events:
            all_conditions.update(events_df["trial_type"].astype(str).unique())
        self.condition_names = sorted(all_conditions)
        
        runs = []

        for img, events_df in zip(self.images, self.events):
            Y, mean_signal, n_scans = img_to_2d(img)
            
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
    
    def _build_block_design(self, runs, hrf):
        """
        Assemble the joint design matrix across runs:
          columns 0 ... n_cond-1  : task regressors (shared betas, stacked)
          columns n_cond ... end  : polynomial regressors (block-diagonal, separate per run)
        """
        
        X_task_blocks, X_poly_blocks, Y_blocks = [], [], []
        
        for run in runs:
            task_block = self._task_matrix_from_hrf(run["stim_trains"], hrf)
            
            X_task_blocks.append(task_block)
            X_poly_blocks.append(run["X_poly"])
            Y_blocks.append(run["Y"])
        
        Y_all = np.vstack(Y_blocks)
        X_task_all = np.vstack(X_task_blocks)   # Shared betas
        X_poly_all = block_diag(*(r["X_poly"] for r in runs))
        
        X_all = np.column_stack([X_task_all, X_poly_all])
        return X_all, Y_all
    
    def _estimate_betas(self, runs, hrf):
        """Fix HRF, estimate beta and poly weights with Ordinary Least Squares"""
        
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
    
    @staticmethod
    def _residualize(Q, A):
        return A - Q @ (Q.T @ A)

    def _estimate_hrf_fir(self, runs, betas, best_voxels, hrf_len):
        """Estimates HRF as in paper."""
        
        A_list, y_list = [], []
        
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
        Estimates optimal HRF by Step 2. from the paper:
        
        1. Fix HRF - OLS betas and polynomial weights
        2. Select best voxels with highest R2
        3. Fix betas - OLS HRF with FIR 
        4. End when R2 > 0.99
        """
        
        if hrf_len is None:
            hrf_len = int(round(20.0 / self.tr))
        
        if len(initial_hrf) < hrf_len:
            initial_hrf = np.pad(initial_hrf, (0, hrf_len - len(initial_hrf)))
        else:
            initial_hrf = initial_hrf[:hrf_len]
        
        current_hrf = initial_hrf.copy()
        
        for i in range(max_iter):
            print(f"Starting iteration {i}:")
            
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
            print(f"  continuing... r2 value: {r2_between:.6f}")
        
        # Normalization of HRF (peak at 1.0)
        peak = np.max(current_hrf)
        if peak > 0:
            current_hrf /= peak

        # Fallback if poorly estimated (from the paper)
        if self._r2_between(initial_hrf, current_hrf) < 0.50:
            current_hrf = initial_hrf.copy()
            peak = np.max(current_hrf)
            if peak > 0:
                current_hrf /= peak

        return current_hrf

    def fit_standard_glm(self, hrf=None):
        """Main GLM fit with provided HRF."""
        
        runs = self.build_all_runs()
        if hrf is None:
            hrf = glover_hrf(self.tr, oversampling=1)
        
        n_cond = len(self.condition_names)
        X_all, Y_all = self._build_block_design(runs, hrf)
        B, _, _, _ = np.linalg.lstsq(X_all, Y_all, rcond=None)

        return {
            "condition_names": self.condition_names,
            "beta": B[:n_cond, :],
            "runs": runs
        }

    def full_workflow(self, runs_num=None):
        """Main function for full GLM fit."""
        runs = self.build_all_runs()
        
        k = runs_num if runs_num is not None else len(runs)
        runs = runs[:k]
        
        initial_hrf = glover_hrf(self.tr, oversampling=1)
        optimal_hrf = self.estimate_hrf(runs, initial_hrf)
        
        return self.fit_standard_glm(optimal_hrf)
    

if __name__ == "__main__":
    main_dir = "/Users/jakubkempa/Documents/magisterka/data/ds000105_R2.0.2"
    subject = "sub-1"
    all_data = load_data(main_dir, subject=subject)
    
    glm = GLMdenoiser(*all_data)
    fit = glm.full_workflow()
    
    breakpoint()
