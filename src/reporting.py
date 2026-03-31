from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd


REPORT_DIR = Path('reports')
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_daily_outputs(signals: pd.DataFrame, portfolio: pd.DataFrame, orders: pd.DataFrame, macro_input: dict) -> None:
    signals.to_csv(REPORT_DIR / 'latest_signals.csv', index=False)
    portfolio.to_csv(REPORT_DIR / 'latest_portfolio_summary.csv', index=False)
    orders.to_csv(REPORT_DIR / 'latest_orders.csv', index=False)

    top_text = signals.head(5)[['ticker', 'signal_score', 'action']].to_markdown(index=False)
    port_text = portfolio[['ticker', 'target_weight', 'target_value', 'shares']].to_markdown(index=False) if not portfolio.empty else 'No target positions.'
    orders_text = orders.to_markdown(index=False) if not orders.empty else 'No orders.'

    content = f"""# Daily ETF Quant Report

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Macro Input

- Date: {macro_input.get('date')}
- Regime Score: {macro_input.get('regime_score')}
- Liquidity Score: {macro_input.get('liquidity_score')}
- News Sentiment Score: {macro_input.get('news_sentiment_score')}
- China Score: {macro_input.get('china_score')}
- US Score: {macro_input.get('us_score')}
- Macro Composite: {round(float(macro_input.get('macro_composite', 0.0)), 4)}

## Top Signals

{top_text}

## Target Portfolio

{port_text}

## Order Sheet

{orders_text}
"""
    (REPORT_DIR / 'daily_report.md').write_text(content, encoding='utf-8')


def save_backtest_outputs(equity: pd.DataFrame, summary: pd.DataFrame) -> None:
    equity.to_csv(REPORT_DIR / 'backtest_equity_curve.csv', index=False)
    summary.to_csv(REPORT_DIR / 'backtest_summary.csv', index=False)
