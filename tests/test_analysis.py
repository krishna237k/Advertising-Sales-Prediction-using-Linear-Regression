import numpy as np
from src.generate_data import generate
from src.analysis import clean, train_eval, FEATURES


def test_cleaning_removes_dupes_nans_outliers():
    raw = generate()
    df, log = clean(raw)
    assert log["duplicates_removed"] >= 1
    assert df[FEATURES].isna().sum().sum() == 0
    assert df["Newspaper"].max() < 200          # injected 320 outlier removed
    assert 190 <= len(df) <= 201


def test_model_quality_and_tv_radio_beat_newspaper():
    df, _ = clean(generate())
    model, m, feat = train_eval(df)
    assert m["r2_test"] > 0.8 and m["rmse_test"] < 3
    assert abs(model.coef_[2]) < 0.02           # newspaper ~ 0
    assert feat.index[0] == "TV"
