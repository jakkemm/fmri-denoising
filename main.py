import argparse
from pathlib import Path

from general_linear_model.pipeline import PCADenoisingPipeline, PCADenoisingResult
from utils.load_data import dataclass_from_pickle, dataclass_to_pickle, load_data

BASE_PATH = Path.cwd() / "data" / "ds000105"


def run_glm_pca_pipeline(subject):
    runs, _ = load_data(
        base_path=BASE_PATH,
        subject=subject
    )
    
    pipeline = PCADenoisingPipeline(
        k_max=20,
        high_pass_cutoff=128.0,
        chunk_size=5_000,
        verbose=True
    )
    final_fit = pipeline.fit(runs)
    
    pickle_path = Path.cwd() / f"result-{subject}.pkl"
    dataclass_to_pickle(final_fit, pickle_path)
    
def load_glm_pca_results(subject):
    pickle_path = Path.cwd() / f"result-{subject}.pkl"
    final_fit = dataclass_from_pickle(PCADenoisingResult, pickle_path)
    
    return final_fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--run_glm", action="store_true")
    parser.add_argument("--load_result", action="store_true")
    args = parser.parse_args()

    if args.run_glm:
        run_glm_pca_pipeline(subject=args.subject)
    elif args.load_result:
        result = load_glm_pca_results(subject=args.subject)
    