
import pytest

from src.phase_0.split_policy import (
    select_hindi_splits,
    build_hindi_split_provenance,
)


@pytest.fixture
def hindi_splits():
    """Create simulated dataset splits without downloading any data."""
    return {
        "train": object(),
        "test": object(),
    }


def test_swaps_hindi_train_and_test_splits(hindi_splits):
    """Training uses the original test split when swapping is enabled."""
    config = {
        "train_split": "train",
        "test_split": "test",
        "swap_train_test": True,
    }

    train, evaluation = select_hindi_splits(hindi_splits, config)

    assert train is hindi_splits["test"]
    assert evaluation is hindi_splits["train"]


def test_preserves_original_splits_when_swap_is_false(hindi_splits):
    """Original train and test splits are preserved when swapping is disabled."""
    config = {
        "train_split": "train",
        "test_split": "test",
        "swap_train_test": False,
    }

    train, evaluation = select_hindi_splits(hindi_splits, config)

    assert train is hindi_splits["train"]
    assert evaluation is hindi_splits["test"]


def test_missing_hindi_split_raises_value_error():
    """A clear error is raised if a required split is unavailable."""
    dataset_splits = {"train": object()}

    config = {
        "train_split": "train",
        "test_split": "test",
        "swap_train_test": True,
    }

    with pytest.raises(ValueError, match="Expected Hindi splits"):
        select_hindi_splits(dataset_splits, config)


def test_hindi_split_provenance_when_swapped():
    """Provenance records the actual split assignment when swapping is enabled."""
    config = {
        "train_split": "train",
        "test_split": "test",
        "swap_train_test": True,
    }

    provenance = build_hindi_split_provenance(config)

    assert provenance == {
        "original_train_split": "train",
        "original_test_split": "test",
        "actual_training_split": "test",
        "actual_evaluation_split": "train",
        "swap_train_test": True,
    }


def test_hindi_split_provenance_when_not_swapped():
    """Provenance records the original split assignment when swapping is disabled."""
    config = {
        "train_split": "train",
        "test_split": "test",
        "swap_train_test": False,
    }

    provenance = build_hindi_split_provenance(config)

    assert provenance["actual_training_split"] == "train"
    assert provenance["actual_evaluation_split"] == "test"
    assert provenance["swap_train_test"] is False
