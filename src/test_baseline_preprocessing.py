import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.churn.baseline_training import BaselineTrainer
from src.churn.mlp_training import MLPTrainer


def test_preparar_features_keeps_only_numeric_model_features():
    dataset = pd.DataFrame(
        {
            "Churn Value": [0, 1],
            "Churn Score": [10, 90],
            "Monthly Charges": [55.0, 80.0],
            "Contract": ["Month-to-month", "Two year"],
            "tenure_bucket": ["0-12", "13-24"],
        }
    )

    features, target = BaselineTrainer().preparar_features(dataset)

    assert target.tolist() == [0, 1]
    assert "Monthly Charges" in features.columns
    assert "Churn Value" not in features.columns
    assert "Contract" not in features.columns


def test_mlp_pipeline_uses_sklearn_classifier():
    pipeline = MLPTrainer(hidden_layer_sizes=(16, 8)).criar_pipeline()

    assert isinstance(pipeline.named_steps["classifier"], MLPClassifier)
