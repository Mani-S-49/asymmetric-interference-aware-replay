from datasets import Dataset, load_dataset
from src.phase_0.split_policy import select_hindi_splits

def _load_split(dataset_id, config_name, split):
    if config_name:
        return load_dataset(dataset_id, config_name, split=split)
    return load_dataset(dataset_id, split=split)

def load_english(config):
    c = config["datasets"]["english"]
    train = _load_split(c["name"], c.get("config"), c["train_split"])
    test = _load_split(c["name"], c.get("config"), c["test_split"])
    train = train.shuffle(seed=int(c["shuffle_seed"])).select(range(min(int(c["train_size"]), len(train))))
    test = test.shuffle(seed=int(c["shuffle_seed"])).select(range(min(int(c["test_size"]), len(test))))
    return _normalize_english(train, c), _normalize_english(test, c)

def _normalize_english(ds, c):
    if c["text_column"] not in ds.column_names or c["label_column"] not in ds.column_names:
        raise ValueError(f"Unexpected IMDb columns: {ds.column_names}")
    def convert(row):
        label = row[c["label_column"]]
        if label not in (0, 1):
            raise ValueError(f"Expected IMDb label 0/1, got {label!r}")
        return {"text": str(row[c["text_column"]]), "labels": int(label)}
    return ds.map(convert, remove_columns=ds.column_names)

def load_hindi(config):
    c = config["datasets"]["hindi"]
    ds = load_dataset(c["name"], c["config"])

    train, test = select_hindi_splits(ds, c)

    seed = int(config["phase"]["seed"])

    if c.get("train_size") is not None:
        train = train.shuffle(seed=seed).select(
            range(min(int(c["train_size"]), len(train)))
        )

    if c.get("test_size") is not None:
        test = test.shuffle(seed=seed).select(
            range(min(int(c["test_size"]), len(test)))
        )

    return _normalize_hindi(train, c), _normalize_hindi(test, c)

def _normalize_hindi(ds, c):
    text_col = c["text_column"]
    label_text_col = c.get("label_text_column")
    source = label_text_col if label_text_col in ds.column_names else c["label_column"]
    if text_col not in ds.column_names or source not in ds.column_names:
        raise ValueError(f"Unexpected Hindi columns: {ds.column_names}")
    def convert(row):
        raw = row[source]
        if isinstance(raw, str):
            value = raw.strip().casefold()
            if value == c["positive_label"].casefold():
                label = 1
            elif value == c["negative_label"].casefold():
                label = 0
            else:
                raise ValueError(f"Unexpected Hindi label {raw!r}")
        elif raw in (0, 1):
            label = int(raw)
        else:
            raise ValueError(f"Unexpected Hindi label {raw!r}")
        return {"text": str(row[text_col]), "labels": label}
    return ds.map(convert, remove_columns=ds.column_names)
