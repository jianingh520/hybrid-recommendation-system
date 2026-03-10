import math
from typing import Dict, Tuple

from pyspark import SparkContext

from .utils import pearson_similarity



def run_item_based_cf(sc: SparkContext, train_file: str, test_file: str):
    train_rdd = sc.textFile(train_file)
    header = train_rdd.first()
    train_data = train_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))

    ratings_rdd = train_data.map(lambda x: float(x[2]))
    rating_sum = ratings_rdd.reduce(lambda a, b: a + b)
    rating_count = ratings_rdd.count()
    global_avg_all = rating_sum / rating_count if rating_count > 0 else 3.77

    user_business_ratings = train_data.map(lambda x: ((x[0], x[1]), float(x[2])))
    business_users = train_data.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict)
    business_user_dict: Dict[str, Dict[str, float]] = business_users.collectAsMap()

    user_business_map = (
        user_business_ratings.map(lambda x: (x[0][0], (x[0][1], x[1])))
        .groupByKey()
        .mapValues(dict)
        .collectAsMap()
    )
    business_avg = (
        train_data.map(lambda x: (x[1], (float(x[2]), 1)))
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        .mapValues(lambda x: x[0] / x[1])
        .collectAsMap()
    )

    test_rdd = sc.textFile(test_file)
    header2 = test_rdd.first()
    test_data = test_rdd.filter(lambda x: x != header2).map(lambda x: x.split(","))

    sim_cache: Dict[Tuple[str, str], Tuple[float, int]] = {}

    def get_similarity(b1: str, b2: str) -> Tuple[float, int]:
        key = tuple(sorted((b1, b2)))
        if key not in sim_cache:
            sim_cache[key] = pearson_similarity(business_user_dict[b1], business_user_dict[b2])
        return sim_cache[key]

    def predict(user: str, business: str) -> float:
        user_exists = user in user_business_map
        business_exists = business in business_user_dict

        if user_exists and business_exists:
            rated_items = user_business_map[user]
            sims = []
            for other_item in rated_items:
                if other_item not in business_user_dict:
                    continue
                sim, common_users = get_similarity(other_item, business)
                sim = max(min(sim, 1.0), 0)
                baseline = business_avg.get(other_item, global_avg_all)
                sim_strength = sim * (math.log1p(common_users) / math.log1p(5))
                sims.append((sim_strength, rated_items[other_item] - baseline))

            if sims:
                numerator = sum(sim * rating for sim, rating in sims)
                denominator = sum(abs(sim) for sim, _ in sims)
                business_baseline = business_avg.get(business, global_avg_all)
                business_support = len(business_user_dict[business])
                confidence = min(1.0, math.log1p(business_support) / math.log1p(50))
                adjusted_deviation = (numerator / denominator) if denominator != 0 else 0
                pred = business_baseline + confidence * adjusted_deviation
                return max(1.0, min(5.0, pred))
                
        # cold-start strategies
        if user_exists and not business_exists:
            user_ratings = list(user_business_map[user].values())
            user_avg = sum(user_ratings) / len(user_ratings)
            return max(1.0, min(5.0, 0.7 * user_avg + 0.3 * global_avg_all))

        if not user_exists and business_exists:
            business_baseline = business_avg.get(business, global_avg_all)
            return max(1.0, min(5.0, 0.7 * business_baseline + 0.3 * global_avg_all))

        return max(1.0, min(5.0, global_avg_all))

    return test_data.map(lambda x: (x[0], x[1], predict(x[0], x[1])))
