"""Creates data/advertising.csv: 200 records modelled on the classic 'Advertising' dataset
(TV, Radio, Newspaper spend in $1000s -> Sales in 1000s of units).

Want to use the real dataset? Just replace data/advertising.csv with it (same column names:
TV, Radio, Newspaper, Sales) and re-run `python -m src.analysis`.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "advertising.csv"


def generate(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tv = rng.uniform(0.7, 296.4, n)
    radio = rng.uniform(0, 49.6, n)
    # newspaper mildly correlated with radio, as in the original data
    news = np.clip(0.3 * radio + rng.gamma(3.0, 8.5, n) - 1, 0.3, 114)
    sales = 2.94 + 0.0458 * tv + 0.1885 * radio - 0.001 * news + rng.normal(0, 1.7, n)
    df = pd.DataFrame(dict(TV=tv, Radio=radio, Newspaper=news, Sales=np.clip(sales, 1.6, None))).round(1)

    # inject realistic data-quality problems for the cleaning step to catch
    df.loc[rng.choice(n, 4, replace=False), "Newspaper"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "Radio"] = np.nan
    df.loc[17, "Newspaper"] = 320.0       # outlier (typo-like)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # duplicate row
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    generate().to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
