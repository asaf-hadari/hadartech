import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from trading_bot import data as mdata
from trading_bot.backtest import run_backtest
from trading_bot.config import BotConfig, BrokerCosts, StrategyParams
from trading_bot.optimizer import SelfImprover, sample_params
from trading_bot.paper_trader import PaperTrader
from trading_bot.portfolio import Portfolio
from trading_bot.strategy import compute_signals


@pytest.fixture(scope="module")
def md():
    return mdata.synthetic(n_symbols=30, n_days=900, seed=11)


def test_no_lookahead(md):
    """Signals on day t must not change when future bars are added."""
    p = StrategyParams()
    full = compute_signals(md, p)
    cut = md.slice(None, md.dates[600])
    part = compute_signals(cut, p)
    assert part.entry.equals(full.entry.loc[part.entry.index])


def test_portfolio_accounting():
    costs = BrokerCosts(commission_pct=0.001, slippage_pct=0.0, min_commission=0)
    pf = Portfolio(cash=10_000)
    assert pf.buy("X", 100, 50, "d1", 90, costs)
    assert pf.cash == pytest.approx(10_000 - 5000 - 5)
    pnl = pf.sell("X", 110, "d2", costs)
    assert pnl == pytest.approx(5500 - 5.5 - 5000)
    assert not pf.positions


def test_backtest_deterministic_and_sane(md):
    r1 = run_backtest(md, StrategyParams())
    r2 = run_backtest(md, StrategyParams())
    assert r1.equity.equals(r2.equity)
    assert (r1.equity > 0).all()
    assert r1.metrics["trades"] > 0


def test_param_sampling_in_bounds():
    import random
    from trading_bot.config import PARAM_SPACE
    rng = random.Random(0)
    for _ in range(200):
        p = sample_params(rng, StrategyParams())
        for k, (lo, hi, _) in PARAM_SPACE.items():
            assert lo <= getattr(p, k) <= hi


def test_optimizer_never_worse_on_dev(md, tmp_path):
    si = SelfImprover(md, BotConfig(), state_dir=tmp_path, holdout_months=6, seed=0, log=lambda *_: None)
    start, *_ = si._eval_dev(StrategyParams())
    p, rep = si.run(max_iters=15, patience=15)
    end, *_ = si._eval_dev(p)
    assert end >= start
    saved = json.loads((tmp_path / "champion.json").read_text())
    assert "holdout_metrics" in saved and "target_met_out_of_sample" in saved


def test_paper_trader_steps(md, tmp_path):
    cfg = BotConfig()
    days = md.dates[400:430]
    for d in days:
        cur = md.slice(None, d)
        prices = {s: float(cur.close[s].iloc[-1]) for s in cur.symbols}
        pt = PaperTrader(cfg, lambda: cur, lambda syms: {s: prices[s] for s in syms},
                         state_dir=tmp_path, log=lambda *_: None)
        now = datetime(d.year, d.month, d.day, 23, 0, tzinfo=ZoneInfo("UTC"))
        eq = pt.step(now)
        assert np.isfinite(eq) and eq > 0
    state = json.loads((tmp_path / "paper_state.json").read_text())
    assert len(state["equity_log"]) == len(days)
    assert state["portfolio"]["trades"], "expected some virtual trades"
