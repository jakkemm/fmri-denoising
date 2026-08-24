import numpy as np


def iter_chunks(Y, chunk_size):
    for start in range(0, Y.shape[1], chunk_size):
        stop = min(start + chunk_size, Y.shape[1])

        chunk_slice = slice(start, stop)
        yield chunk_slice, Y[:, chunk_slice]

def log(module, message):
    print(f"[{module}] {message}", flush=True)

def calculate_r2(y_true, y_pred):
    mean_true = np.mean(y_true, axis=0, keepdims=True)
    
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - mean_true)**2, axis=0)
    
    r2 = np.full(ss_tot.shape, np.nan, dtype=np.float32)
    
    valid = ss_tot > 0

    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return r2