from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.features import add_features


class SignalEngine:
    def __init__(self, config: dict):
        self.config = config
        self.signal_cfg = config['signals']

    def generate_latest_signals(self, dataset: Dict[str, pd.DataFrame], macro_input: dict) -> pd.DataFrame:
        rows: List[dict] = []
        macro_score = macro_input.get('macro_composite', 0.0)

        for ticker, df in dataset.items():
            feat = add_features(df, self.config)
            row = feat.iloc[-1].copy()
            score = self._compute_total_score(row, macro_score)
            action = self._decide_action(score, row)
            rows.append({
                'ticker': ticker,
                'date': row['date'],
                'close': row['close'],
                'trend_score': round(float(row['trend_score']), 4),
                'momentum_score': round(float(row['momentum_score']), 4),
                'mean_reversion_score': round(float(row['mean_reversion_score']), 4),
                'breakout_score': round(float(row['breakout_score'] + row['breakdown_score']), 4),
                'relative_strength': round(float(row['relative_strength_proxy']), 4),
                'rsi': round(float(row['rsi']), 2),
                'adx': round(float(row['adx']), 2),
                'atr': round(float(row['atr']) if pd.notna(row['atr']) else 0.0, 4),
                'macro_score': round(float(macro_score), 4),
                'signal_score': round(float(score), 4),
                'action': action,
            })

        signals = pd.DataFrame(rows).sort_values('signal_score', ascending=False).reset_index(drop=True)
        return signals

    def _compute_total_score(self, row: pd.Series, macro_score: float) -> float:
        cfg = self.signal_cfg

        trend_component = row['trend_score']
        momentum_component = np.tanh(row['momentum_score'] * 8)
        mean_reversion_component = np.tanh(row['mean_reversion_score'] * 8)
        breakout_component = row['breakout_score'] + row['breakdown_score']
        relative_strength_component = np.tanh(row['relative_strength_proxy'] / 3)

        volatility_penalty = 0.0
        if pd.notna(row['std_20']) and row['std_20'] > 0.03:
            volatility_penalty = -0.10

        overbought_penalty = -0.10 if row['rsi'] > 75 else 0.0
        oversold_bonus = 0.05 if row['rsi'] < 35 else 0.0
        trend_strength_bonus = 0.05 if row['adx'] > 25 else 0.0

        total = (
            cfg['trend_weight'] * trend_component +
            cfg['momentum_weight'] * momentum_component +
            cfg['mean_reversion_weight'] * mean_reversion_component +
            cfg['breakout_weight'] * breakout_component +
            cfg['relative_strength_weight'] * relative_strength_component +
            cfg['news_score_weight'] * macro_score +
            volatility_penalty + overbought_penalty + oversold_bonus + trend_strength_bonus
        )
        return float(max(0.0, min(1.0, (total + 0.5))))

    def _decide_action(self, score: float, row: pd.Series) -> str:
        buy_th = self.signal_cfg['timing_threshold_buy']
        sell_th = self.signal_cfg['timing_threshold_sell']
        if score >= buy_th and row['close'] > row['ma_20']:
            return 'BUY'
        if score <= sell_th or row['close'] < row['ma_20'] * 0.985:
            return 'SELL'
        return 'HOLD'
