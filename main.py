from GeneralLinearModel.FGLS import FGLSRegressor
from GeneralLinearModel.GLMMatrix import GLMMatrixBuilder
from utils.load_data import load_data

if __name__ == "__main__":
    runs, t1_img = load_data(
        base_path="/Users/jakubkempa/Documents/magisterka/data/ds000105/",
        subject="sub-1"
    )
    
    builder = GLMMatrixBuilder()
    glm_data = builder.build(runs)
    
    model = FGLSRegressor()
    model.fit(glm_data.X, glm_data.Y)