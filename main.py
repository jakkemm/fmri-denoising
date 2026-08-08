from general_linear_model.pipeline import PCADenoisingPipeline
from utils.load_data import load_data

if __name__ == "__main__":
    runs, t1_img = load_data(
        base_path="/Users/jakubkempa/Documents/magisterka/data/ds000105/",
        subject="sub-1"
    )
    
    pipeline = PCADenoisingPipeline(
        k_max=20,
        high_pass_cutoff=128.0,
        chunk_size=5_000,
        verbose=True
    )
    pipeline.fit(runs)
