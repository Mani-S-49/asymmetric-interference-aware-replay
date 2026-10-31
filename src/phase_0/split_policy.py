def select_hindi_splits(dataset_splits, config):
    """Select Hindi training and evaluation splits according to the experiment protocol.

    When swap_train_test is True:
        - Training uses the dataset's original test split.
        - Evaluation uses the dataset's original train split.

    Args:
        dataset_splits: Mapping of split names to dataset objects.
        config: Hindi dataset configuration dictionary.

    Returns:
        A tuple containing (train_split, evaluation_split).

    Raises:
        ValueError: If either configured split is unavailable.
    """
    original_train_split = config["train_split"]
    original_test_split = config["test_split"]

    if (
        original_train_split not in dataset_splits
        or original_test_split not in dataset_splits
    ):
        raise ValueError(
            f"Expected Hindi splits {original_train_split!r} and "
            f"{original_test_split!r}; "
            f"available splits: {list(dataset_splits.keys())}"
        )

    if config.get("swap_train_test", False):
        train = dataset_splits[original_test_split]
        evaluation = dataset_splits[original_train_split]
    else:
        train = dataset_splits[original_train_split]
        evaluation = dataset_splits[original_test_split]

    return train, evaluation
def build_hindi_split_provenance(config):
    """Describe the original and actual Hindi split assignments."""
    original_train = config["train_split"]
    original_test = config["test_split"]
    swapped = bool(config.get("swap_train_test", False))

    return {
        "original_train_split": original_train,
        "original_test_split": original_test,
        "actual_training_split": original_test if swapped else original_train,
        "actual_evaluation_split": original_train if swapped else original_test,
        "swap_train_test": swapped,
    }