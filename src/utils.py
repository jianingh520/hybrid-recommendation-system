import ast
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_json_parse(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def pearson_similarity(
    b1_ratings: Dict[str, float],
    b2_ratings: Dict[str, float],
) -> Tuple[float, int]:
    common_users = set(b1_ratings.keys()) & set(b2_ratings.keys())
    if not common_users:
        return 0.0, 0

    b1_vals = [b1_ratings[u] for u in common_users]
    b2_vals = [b2_ratings[u] for u in common_users]

    avg1 = sum(b1_vals) / len(b1_vals)
    avg2 = sum(b2_vals) / len(b2_vals)

    numerator = sum((b1_ratings[u] - avg1) * (b2_ratings[u] - avg2) for u in common_users)
    denom1 = math.sqrt(sum((b1_ratings[u] - avg1) ** 2 for u in common_users))
    denom2 = math.sqrt(sum((b2_ratings[u] - avg2) ** 2 for u in common_users))

    if denom1 == 0 or denom2 == 0:
        return 0.0, len(common_users)
    return numerator / (denom1 * denom2), len(common_users)


def log_transform_features(feature_vector: List[float], log_indices: Iterable[int]) -> List[float]:
    log_index_set = set(log_indices)
    transformed: List[float] = []
    for i, val in enumerate(feature_vector):
        if i in log_index_set:
            try:
                transformed.append(math.log1p(float(val)))
            except (TypeError, ValueError):
                transformed.append(0.0)
        else:
            transformed.append(val)
    return transformed


def safe_parse_business_parking(business_record: Dict[str, Any]) -> Dict[str, int]:
    attributes = business_record.get("attributes") or {}
    parking_raw = attributes.get("BusinessParking")
    if parking_raw:
        try:
            parsed = ast.literal_eval(parking_raw)
            return {
                "has_parking_garage": int(parsed.get("garage", False)),
                "has_parking_street": int(parsed.get("street", False)),
                "has_parking_validated": int(parsed.get("validated", False)),
                "has_parking_lot": int(parsed.get("lot", False)),
                "has_parking_valet": int(parsed.get("valet", False)),
            }
        except (ValueError, SyntaxError):
            pass

    return {
        "has_parking_garage": 0,
        "has_parking_street": 0,
        "has_parking_validated": 0,
        "has_parking_lot": 0,
        "has_parking_valet": 0,
    }


def parse_bool(value: Optional[str]) -> int:
    return 1 if value == "True" else 0


def parse_noise_level(value: Optional[str]) -> int:
    levels = {"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}
    return levels.get(value.lower(), 1) if value else 1


def parse_attire(value: Optional[str]) -> int:
    attire_map = {"casual": 0, "dressy": 1, "formal": 2}
    return attire_map.get(value.lower(), 0) if value else 0
