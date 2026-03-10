from typing import Dict, List

from .utils import log_transform_features


def add_cross_features(user_features: Dict[str, float], biz_features: Dict[str, float]) -> List[float]:
    return [
        user_features.get("user_avg_stars", 3.5) * biz_features.get("biz_stars", 3.5),
        user_features.get("user_review_count", 0) * biz_features.get("biz_review_count", 0),
        user_features.get("user_useful", 0) * biz_features.get("has_tv", 0),
        biz_features.get("price_range", 2) * user_features.get("user_fans", 0),
        biz_features.get("delivery_available", 0) * biz_features.get("has_parking_street", 0),
    ]



def extract_features(data_rdd, user_feat, biz_feat, checkin_feat, photo_feat, top_categories, top_cities):
    def build(row):
        user_id = row["user_id"]
        biz_id = row["business_id"]
        label = float(row["stars"]) if ("stars" in row and row["stars"] != "") else None

        uf = user_feat.get(user_id, {})
        bf = biz_feat.get(biz_id, {})

        features = [
            uf.get("user_review_count", 0),
            uf.get("user_avg_stars", 3.5),
            uf.get("user_fans", 0),
            uf.get("user_useful", 0),
            uf.get("user_funny", 0),
            uf.get("user_cool", 0),
            uf.get("user_is_elite", 0),
            bf.get("biz_stars", 3.5),
            bf.get("biz_review_count", 0),
            bf.get("is_open", 1),
            bf.get("price_range", 2),
            checkin_feat.get(biz_id, 0),
            photo_feat.get(biz_id, 0),
            bf.get("latitude", 0.0),
            bf.get("longitude", 0.0),
            uf.get("user_tenure", 0),
            uf.get("user_compliments", 0),
            bf.get("outdoor_seating", 0),
            bf.get("bike_parking", 0),
            bf.get("has_food_category", 0),
            bf.get("has_restaurant_category", 0),
            bf.get("good_for_kids", 0),
            bf.get("has_tv", 0),
            bf.get("noise_level", 1),
            bf.get("attire", 0),
            bf.get("delivery_available", 0),
            bf.get("good_for_groups", 0),
            bf.get("accepts_reservations", 0),
            bf.get("has_takeout", 0),
            bf.get("accepts_credit_cards", 0),
            bf.get("has_parking_garage", 0),
            bf.get("has_parking_street", 0),
            bf.get("has_parking_validated", 0),
            bf.get("has_parking_lot", 0),
            bf.get("has_parking_valet", 0),
        ]
        features += [bf.get(f"cat_{c}", 0) for c in top_categories]
        features += [bf.get(f"city_{c}", 0) for c in top_cities]
        features.extend(add_cross_features(uf, bf))

        log_indices = [0, 2, 3, 4, 5, 11, 12, 16]
        log_indices += [len(features) - 5 + 1, len(features) - 5 + 3]
        features = log_transform_features(features, log_indices)

        return (features, label, user_id, biz_id)

    return data_rdd.map(build)
