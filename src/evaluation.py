from typing import Dict, List, Tuple

from pyspark import SparkContext



def load_truth_map(sc: SparkContext, test_file: str) -> Dict[Tuple[str, str], float]:
    test_truth = (
        sc.textFile(test_file)
        .filter(lambda x: x.strip() != "")
        .map(lambda line: line.split(","))
        .filter(lambda x: x[0] != "user_id")
    )
    return {(x[0], x[1]): float(x[2]) for x in test_truth.collect()}



def compute_rmse(predictions: List[Tuple[str, str, float]], truth_map: Dict[Tuple[str, str], float]):
    errors = [abs(pred - truth_map.get((u, b), 3.5)) for u, b, pred in predictions]
    mse = sum(e ** 2 for e in errors) / len(errors)
    rmse = mse ** 0.5
    return rmse, errors



def summarize_error_distribution(errors: List[float]):
    bins = [0, 1, 2, 3, 4, 5]
    dist = {f"({bins[i]},{bins[i + 1]}]": 0 for i in range(len(bins) - 1)}
    for e in errors:
        for i in range(len(bins) - 1):
            if bins[i] < e <= bins[i + 1]:
                dist[f"({bins[i]},{bins[i + 1]}]"] += 1
                break
    return dist
