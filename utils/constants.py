from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]

class CONST:
    run_duration = 300
    tr = 2.5
    n_scans = 121
    
    categories = (
        "face",
        "house",
        "cat",
        "bottle",
        "scissors",
        "shoe",
        "chair",
        "scrambledpix"
    )
    n_categories = len(categories)

@dataclass
class RawData:
    Y: FloatArray               # fMRI data (t x v)
    events_df: pd.DataFrame

@dataclass
class RunData:
    Y: FloatArray
    X: FloatArray
    task_slice: slice
    drift_slice: slice
    pca_slice: slice | None
