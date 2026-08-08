from general_linear_model.cv import LeaveOneRunOutEvaluator
from utils.load_data import load_data

if __name__ == "__main__":
    runs, t1_img = load_data(
        base_path="/Users/jakubkempa/Documents/magisterka/data/ds000105/",
        subject="sub-1"
    )
    
    evaluator = LeaveOneRunOutEvaluator(verbose=True)
    evaluator.evaluate(runs)