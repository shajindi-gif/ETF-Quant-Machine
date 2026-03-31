from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from src.config_loader import load_config
from src.data_loader import MarketDataLoader
from src.manual_macro import load_latest_macro_signal
from src.signal_engine import SignalEngine
from src.portfolio import PortfolioConstructor
from src.backtester import Backtester
from src.reporting import save_daily_outputs, save_backtest_outputs


POSITIONS_PATH = Path('data/processed/current_positions.json')


def read_current_positions() -> dict:
    if not POSITIONS_PATH.exists():
        return {}
    with POSITIONS_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def run_daily() -> None:
    config = load_config()
    loader = MarketDataLoader(config)
    dataset = loader.load_all()
    macro_input = load_latest_macro_signal()

    engine = SignalEngine(config)
    signals = engine.generate_latest_signals(dataset, macro_input)

    portfolio_builder = PortfolioConstructor(config)
    target_portfolio = portfolio_builder.build_target_portfolio(signals)
    current_positions = read_current_positions()
    order_sheet = portfolio_builder.build_order_sheet(target_portfolio, current_positions)

    save_daily_outputs(signals, target_portfolio, order_sheet, macro_input)
    print('Daily run complete. Files saved under reports/.')


def run_backtest() -> None:
    config = load_config()
    loader = MarketDataLoader(config)
    dataset = loader.load_all()

    macro_input = load_latest_macro_signal()
    macro_bias = float(macro_input.get('macro_composite', 0.0))

    bt = Backtester(config)
    equity, summary = bt.run(dataset, macro_bias=macro_bias)
    save_backtest_outputs(equity, summary)
    print('Backtest complete. Files saved under reports/.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETF Quant Machine')
    parser.add_argument('command', choices=['run-daily', 'backtest'])
    args = parser.parse_args()

    if args.command == 'run-daily':
        run_daily()
    elif args.command == 'backtest':
        run_backtest()
