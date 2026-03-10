import json
from typing import List, Tuple

import numpy as np
import xgboost as xgb
from pyspark import SparkContext

from .data_loader import (
    extract_top_categories,
    extract_top_cities,
    load_business_features,
    load_checkin_features,
    load_photo_features,
    load_user_features,
)
from .feature_engineering import extract_features



def run_xgboost_model(sc: SparkContext, folder_path: str, test_file: str) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    user_feat = load_user_features(sc, f"{folder_path}/user.json")
    top_categories = extract_top_categories(sc, f"{folder_path}/business.json", top_n=15)
    top_cities = extract_top_cities(sc, f"{folder_path}/business.json", top_n=10)
    biz_feat = load_business_features(sc, f"{folder_path}/business.json", top_categories, top_cities)
    checkin_feat = load_checkin_features(sc, f"{folder_path}/checkin.json")
    photo_feat = load_photo_features(sc, f"{folder_path}/photo.json")

    review_rdd = (
        sc.textFile(f"{folder_path}/review_train.json")
        .filter(lambda x: x.strip() != "")
        .map(json.loads)
    )
    train_data = extract_features(
        review_rdd, user_feat, biz_feat, checkin_feat, photo_feat, top_categories, top_cities
    )
    x_train = train_data.map(lambda x: x[0]).collect()
    y_train = train_data.map(lambda x: x[1]).collect()

    test_rdd = (
        sc.textFile(test_file)
        .filter(lambda x: x.strip() != "")
        .map(lambda line: dict(zip(["user_id", "business_id", "stars"], line.strip().split(","))))
        .filter(lambda x: x["user_id"] != "user_id")
    )
    test_data = extract_features(
        test_rdd, user_feat, biz_feat, checkin_feat, photo_feat, top_categories, top_cities
    )
    x_test = test_data.map(lambda x: x[0]).collect()
    ids = test_data.map(lambda x: (x[2], x[3])).collect()

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(np.array(x_train), np.array(y_train))
    predictions = model.predict(np.array(x_test))
    return predictions, ids
