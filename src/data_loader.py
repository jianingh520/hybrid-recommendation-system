from datetime import datetime
from typing import Dict, List

from pyspark import SparkContext

from .utils import (
    parse_attire,
    parse_bool,
    parse_noise_level,
    safe_json_parse,
    safe_parse_business_parking,
)


def extract_top_categories(sc: SparkContext, biz_path: str, top_n: int = 15) -> List[str]:
    biz_rdd = sc.textFile(biz_path).map(safe_json_parse).filter(lambda x: x is not None)
    categories = biz_rdd.flatMap(
        lambda b: b.get("categories", "").split(", ") if b.get("categories") else []
    )
    top_categories = categories.map(lambda c: (c, 1)).reduceByKey(lambda a, b: a + b).takeOrdered(
        top_n, key=lambda x: -x[1]
    )
    return [cat for cat, _ in top_categories]



def extract_top_cities(sc: SparkContext, biz_path: str, top_n: int = 10) -> List[str]:
    biz_rdd = sc.textFile(biz_path).map(safe_json_parse).filter(lambda x: x is not None)
    cities = biz_rdd.map(lambda b: (b.get("city", ""), 1)).reduceByKey(lambda a, b: a + b)
    top_city_list = cities.takeOrdered(top_n, key=lambda x: -x[1])
    return [city for city, _ in top_city_list]



def load_user_features(sc: SparkContext, user_path: str) -> Dict[str, Dict[str, float]]:
    user_rdd = sc.textFile(user_path).map(safe_json_parse).filter(lambda x: x is not None)
    return user_rdd.map(
        lambda u: (
            u["user_id"],
            {
                "user_review_count": u.get("review_count", 0),
                "user_avg_stars": u.get("average_stars", 3.5),
                "user_fans": u.get("fans", 0),
                "user_useful": u.get("useful", 0),
                "user_tenure": datetime.now().year - int(u.get("yelping_since", "2000-01")[:4]),
                "user_compliments": sum(
                    [
                        u.get("compliment_hot", 0),
                        u.get("compliment_more", 0),
                        u.get("compliment_profile", 0),
                        u.get("compliment_cute", 0),
                        u.get("compliment_list", 0),
                        u.get("compliment_note", 0),
                        u.get("compliment_plain", 0),
                        u.get("compliment_cool", 0),
                        u.get("compliment_funny", 0),
                        u.get("compliment_writer", 0),
                        u.get("compliment_photos", 0),
                    ]
                ),
                "user_funny": u.get("funny", 0),
                "user_cool": u.get("cool", 0),
                "user_is_elite": 0 if u.get("elite", "None") == "None" else 1,
            },
        )
    ).collectAsMap()



def load_business_features(
    sc: SparkContext,
    biz_path: str,
    top_categories: List[str],
    top_cities: List[str],
) -> Dict[str, Dict[str, float]]:
    def category_flags(cat_string: str) -> Dict[str, int]:
        flags = {f"cat_{c}": 0 for c in top_categories}
        if cat_string:
            for c in cat_string.split(", "):
                if c in top_categories:
                    flags[f"cat_{c}"] = 1
        return flags

    def city_flags(city: str) -> Dict[str, int]:
        return {f"city_{c}": int(city == c) for c in top_cities}

    biz_rdd = sc.textFile(biz_path).map(safe_json_parse).filter(lambda x: x is not None)
    return biz_rdd.map(
        lambda b: (
            b["business_id"],
            {
                "biz_stars": b.get("stars", 3.5),
                "biz_review_count": b.get("review_count", 0),
                "is_open": b.get("is_open", 1),
                "price_range": int(b["attributes"].get("RestaurantsPriceRange2", 2))
                if b.get("attributes") and b["attributes"].get("RestaurantsPriceRange2")
                else 2,
                "outdoor_seating": 1
                if b.get("attributes") and b["attributes"].get("OutdoorSeating") == "True"
                else 0,
                "bike_parking": 1
                if b.get("attributes") and b["attributes"].get("BikeParking") == "True"
                else 0,
                "has_food_category": 1 if b.get("categories") and "Food" in b["categories"] else 0,
                "has_restaurant_category": 1
                if b.get("categories") and "Restaurants" in b["categories"]
                else 0,
                "good_for_kids": parse_bool(b.get("attributes", {}).get("GoodForKids"))
                if b.get("attributes")
                else 0,
                "has_tv": parse_bool(b.get("attributes", {}).get("HasTV")) if b.get("attributes") else 0,
                "noise_level": parse_noise_level(b.get("attributes", {}).get("NoiseLevel"))
                if b.get("attributes")
                else 1,
                "attire": parse_attire(b.get("attributes", {}).get("RestaurantsAttire"))
                if b.get("attributes")
                else 0,
                "delivery_available": parse_bool(
                    b.get("attributes", {}).get("RestaurantsDelivery")
                )
                if b.get("attributes")
                else 0,
                "good_for_groups": parse_bool(
                    b.get("attributes", {}).get("RestaurantsGoodForGroups")
                )
                if b.get("attributes")
                else 0,
                "accepts_reservations": parse_bool(
                    b.get("attributes", {}).get("RestaurantsReservations")
                )
                if b.get("attributes")
                else 0,
                "has_takeout": parse_bool(b.get("attributes", {}).get("RestaurantsTakeOut"))
                if b.get("attributes")
                else 0,
                "latitude": b.get("latitude", 0.0),
                "longitude": b.get("longitude", 0.0),
                **category_flags(b.get("categories")),
                **city_flags(b.get("city")),
                "accepts_credit_cards": parse_bool(
                    b.get("attributes", {}).get("BusinessAcceptsCreditCards")
                )
                if b.get("attributes")
                else 0,
                **safe_parse_business_parking(b),
            },
        )
    ).collectAsMap()



def load_checkin_features(sc: SparkContext, checkin_path: str) -> Dict[str, int]:
    checkin_rdd = sc.textFile(checkin_path).map(safe_json_parse).filter(lambda x: x is not None)
    return checkin_rdd.map(lambda c: (c["business_id"], sum(c["time"].values()))).collectAsMap()



def load_photo_features(sc: SparkContext, photo_path: str) -> Dict[str, int]:
    photo_rdd = sc.textFile(photo_path).map(safe_json_parse).filter(lambda x: x is not None)
    return photo_rdd.map(lambda p: (p["business_id"], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
