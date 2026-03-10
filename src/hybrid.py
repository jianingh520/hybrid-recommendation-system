from typing import Dict, Iterable, List, Sequence, Tuple



def combine_predictions(
    cf_predictions: Iterable[Tuple[str, str, float]],
    model_predictions: Sequence[float],
    ids: Sequence[Tuple[str, str]],
    alpha: float = 0.1,
) -> List[Tuple[str, str, float]]:
    cf_dict: Dict[Tuple[str, str], float] = {(u, b): pred for u, b, pred in cf_predictions}
    hybrid_preds: List[Tuple[str, str, float]] = []

    for i, (user_id, business_id) in enumerate(ids):
        cf_score = cf_dict.get((user_id, business_id), 3.5)
        model_score = model_predictions[i]
        hybrid_score = alpha * cf_score + (1 - alpha) * model_score
        hybrid_preds.append((user_id, business_id, hybrid_score))

    return hybrid_preds
