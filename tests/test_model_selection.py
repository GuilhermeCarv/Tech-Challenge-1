import pandas as pd

from src.model_selection import ModelComparator


def test_selecionar_campeao_prioriza_f1_e_respeita_recall_minimo():
    comparison = pd.DataFrame(
        [
            {"model": "logistic_regression", "cv_f1_mean": 0.71, "cv_roc_auc_mean": 0.78, "cv_recall_mean": 0.72},
            {"model": "random_forest", "cv_f1_mean": 0.74, "cv_roc_auc_mean": 0.81, "cv_recall_mean": 0.60},
        ]
    )

    comparator = ModelComparator(minimum_recall=0.7)

    assert comparator.selecionar_campeao(comparison) == "logistic_regression"
