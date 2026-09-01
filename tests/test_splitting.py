import numpy as np

from src.splitting import get_or_create_split_indices


def test_split_is_persisted_and_reused(tmp_path):
    target = np.array([0] * 80 + [1] * 20)
    artifact = tmp_path / "split_indices.npz"

    first = get_or_create_split_indices(target, artifact_path=artifact)
    second = get_or_create_split_indices(target, artifact_path=artifact)

    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    assert len(np.unique(np.concatenate((first.train, first.validation, first.test)))) == len(target)
