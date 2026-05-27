from utils.load_data import load_data
from utils.models import GLMdenoiser

if __name__ == "__main__":
    base_path = "/Users/jakubkempa/Documents/magisterka/ds000105_R2.0.2"
    all_data = load_data(base_path, subject="sub-1")

    glm = GLMdenoiser(*all_data)
    glm.fit_standard_glm()

