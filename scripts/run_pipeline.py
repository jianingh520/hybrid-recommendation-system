import sys
from datetime import datetime

from pyspark import SparkContext

from src.collaborative_filtering import run_item_based_cf
from src.evaluation import compute_rmse, load_truth_map, summarize_error_distribution
from src.hybrid import combine_predictions
from src.model_training import run_xgboost_model



def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python scripts/run_pipeline.py <data_folder> <test_file> <output_file>")

    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    sc = SparkContext(appName="hybrid-recommender-system")
    sc.setLogLevel("ERROR")

    start = datetime.now()

    cf_predictions_rdd = run_item_based_cf(
        sc=sc,
        train_file=f"{folder_path}/yelp_train.csv",
        test_file=test_file,
    )
    cf_predictions = cf_predictions_rdd.collect()

    model_predictions, ids = run_xgboost_model(
        sc=sc,
        folder_path=folder_path,
        test_file=test_file,
    )

    hybrid_predictions = combine_predictions(
        cf_predictions=cf_predictions,
        model_predictions=model_predictions,
        ids=ids,
        alpha=0.1,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("user_id,business_id,prediction\n")
        for user_id, business_id, pred in hybrid_predictions:
            f.write(f"{user_id},{business_id},{pred}\n")

    truth_map = load_truth_map(sc, test_file)
    rmse, errors = compute_rmse(hybrid_predictions, truth_map)
    error_distribution = summarize_error_distribution(errors)

    print(f"Hybrid RMSE: {rmse}")
    print("Error Distribution:")
    for bucket, count in sorted(error_distribution.items()):
        print(f"{bucket}: {count}")
    print(f"Duration: {(datetime.now() - start).total_seconds()}")

    sc.stop()


if __name__ == "__main__":
    main()
