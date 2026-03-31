from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


class MarketDataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config['data']['raw_data_dir'])
        self.required_columns = config['data']['required_columns']
        self.tickers = config['universe']['tickers']

    def load_all(self) -> Dict[str, pd.DataFrame]:
        dataset: Dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            path = self.raw_dir / f'{ticker}.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path)
            dataset[ticker] = self._standardize(df, ticker)
        if not dataset:
            raise ValueError('No ETF csv files found in data/raw. Please add at least one file.')
        return dataset

    def _standardize(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f'{ticker} missing required columns: {missing}')

        result = df[self.required_columns].copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            result[col] = pd.to_numeric(result[col], errors='coerce')

        result = result.dropna(subset=['open', 'high', 'low', 'close'])
        result['ticker'] = ticker
        return result
