from __future__ import annotations

from pathlib import Path
import pandas as pd


MANUAL_INPUT_PATH = Path('data/processed/manual_macro_input.csv')


def load_latest_macro_signal() -> dict:
    if not MANUAL_INPUT_PATH.exists():
        raise FileNotFoundError('manual_macro_input.csv not found under data/processed/')

    df = pd.read_csv(MANUAL_INPUT_PATH)
    if df.empty:
        raise ValueError('manual_macro_input.csv is empty')

    df['date'] = pd.to_datetime(df['date'])
    row = df.sort_values('date').iloc[-1].to_dict()

    total = (
        row.get('regime_score', 0)
        + row.get('liquidity_score', 0)
        + row.get('news_sentiment_score', 0)
        + 0.5 * row.get('china_score', 0)
        + 0.5 * row.get('us_score', 0)
    )

    normalized = max(-1.0, min(1.0, total / 6.0))
    row['macro_composite'] = normalized
    return row
