"""Command line interface.

  python -m trading_bot scan      --market nasdaq100
  python -m trading_bot backtest  --market ta35
  python -m trading_bot optimize  --market nasdaq100 [--forever]
  python -m trading_bot paper     --market nasdaq100 --interval 15
  python -m trading_bot status

Add --synthetic to any command to run offline on generated data (testing only).
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import data as mdata
from .backtest import run_backtest, format_metrics
from .config import BotConfig, StrategyParams
from .optimizer import SelfImprover
from .strategy import compute_signals
from .universe import get_universe


def load_cfg(args):
    cfg = BotConfig.load(args.config) if args.config and Path(args.config).exists() else BotConfig()
    cfg.market = args.market or cfg.market
    if args.capital:
        cfg.initial_capital = args.capital
    if args.target is not None:
        cfg.target_monthly_return = args.target / 100
    champ = Path(args.state_dir) / "champion.json"
    if champ.exists() and not args.default_params:
        cfg.params = StrategyParams.from_dict(json.loads(champ.read_text())["params"])
    return cfg


def get_data(args, cfg, period=None):
    if args.synthetic:
        return mdata.synthetic(n_symbols=60, n_days=1500)
    syms = get_universe(cfg.market, args.symbols_file)
    print(f"Downloading {len(syms)} symbols ({cfg.market})...")
    md = mdata.download_yahoo(syms, period=period or args.period, cache_dir=f"{args.state_dir}/cache")
    if cfg.market == "nasdaq_all":
        md = md.liquid()
    print(f"Loaded {len(md.symbols)} symbols, {md.dates[0].date()} -> {md.dates[-1].date()}")
    return md


def live_price_fn(args):
    if args.synthetic:
        return lambda syms: {}
    return mdata.live_prices


def cmd_scan(args):
    cfg = load_cfg(args)
    md = get_data(args, cfg, period="2y")
    sig = compute_signals(md, cfg.params)
    last = sig.entry.index[-1]
    row = sig.entry.loc[last]
    picks = sig.score.loc[last][row[row].index].sort_values(ascending=False)
    print(f"\nTrend candidates on {last.date()} ({len(picks)} of {len(md.symbols)} symbols):")
    for s, sc in picks.head(args.top).items():
        print(f"  {s:<10} momentum score {sc:6.2f}  close {md.close[s].iloc[-1]:10.2f}")
    exits = sig.trend_exit.loc[last]
    print(f"\nSymbols in downtrend (fast EMA < slow EMA): {int(exits.sum())}")


def cmd_backtest(args):
    cfg = load_cfg(args)
    md = get_data(args, cfg)
    res = run_backtest(md, cfg.params, cfg.initial_capital, cfg.costs)
    print("\nParameters:", cfg.params.to_dict())
    print(format_metrics(res.metrics))
    out = Path(args.state_dir)
    out.mkdir(exist_ok=True)
    res.equity.to_csv(out / "backtest_equity.csv")
    res.trades.to_csv(out / "backtest_trades.csv", index=False)
    print(f"\nSaved equity curve and trades to {out}/")


def cmd_optimize(args):
    cfg = load_cfg(args)
    md = get_data(args, cfg)
    si = SelfImprover(md, cfg, state_dir=args.state_dir, holdout_months=args.holdout_months,
                      seed=args.seed)
    _, rep = si.run(max_iters=args.iters, patience=args.patience, forever=args.forever,
                    max_hours=args.max_hours)
    dm, hm = rep["development_metrics"], rep.get("holdout_metrics", {})
    print("\n=== RESULT ===")
    print(f"Development {rep['development_period']}: avg monthly {dm['avg_monthly_return']:.2%}, "
          f"max DD {dm['max_drawdown']:.1%}")
    if hm:
        print(f"Holdout (unseen) {rep['holdout_period']}: avg monthly {hm['avg_monthly_return']:.2%}, "
              f"max DD {hm['max_drawdown']:.1%}, trades {hm['trades']:.0f}")
    print(f"Target {cfg.target_monthly_return:.0%} met out-of-sample: {rep['target_met_out_of_sample']}")
    print(f"Champion saved to {args.state_dir}/champion.json")


def cmd_paper(args):
    from .paper_trader import PaperTrader
    cfg = load_cfg(args)
    holder = {}

    def fetch():
        holder["data"] = get_data(args, cfg, period="2y")
        return holder["data"]

    pt = PaperTrader(cfg, fetch, live_price_fn(args), state_dir=args.state_dir,
                     reoptimize_every_days=args.reopt_days, reopt_iters=args.iters)
    print(f"Paper trading {cfg.market} with virtual capital {cfg.initial_capital:,.0f}. Ctrl+C to stop.")
    while True:
        try:
            eq = pt.step()
            print(f"{datetime.now():%Y-%m-%d %H:%M}  equity {eq:,.2f}  positions {len(pt.pf.positions)}")
            if not args.no_reopt:
                pt.maybe_reoptimize(get_data(args, cfg))
        except KeyboardInterrupt:
            break
        except Exception as e:  # keep the bot alive on network hiccups
            print(f"step failed: {e!r}")
        if args.once:
            break
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            break


def cmd_status(args):
    from .paper_trader import PaperTrader
    cfg = load_cfg(args)
    pt = PaperTrader(cfg, lambda: None, live_price_fn(args), state_dir=args.state_dir)
    print(json.dumps(pt.status(), indent=2, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trading_bot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scan", "backtest", "optimize", "paper", "status"])
    ap.add_argument("--market", default=None, help="nasdaq100 | nasdaq_all | ta35")
    ap.add_argument("--symbols-file", help="text file with your own symbols")
    ap.add_argument("--config", default="bot_config.json")
    ap.add_argument("--state-dir", default="bot_state")
    ap.add_argument("--period", default="5y", help="history to download (yfinance period)")
    ap.add_argument("--capital", type=float)
    ap.add_argument("--target", type=float, help="target avg monthly return in %% (default 10)")
    ap.add_argument("--default-params", action="store_true", help="ignore saved champion")
    ap.add_argument("--synthetic", action="store_true", help="offline demo data (not real!)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--forever", action="store_true", help="optimise until target is reached")
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--holdout-months", type=int, default=12)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--interval", type=float, default=15, help="paper: minutes between checks")
    ap.add_argument("--reopt-days", type=int, default=7)
    ap.add_argument("--no-reopt", action="store_true")
    ap.add_argument("--once", action="store_true", help="paper: run a single step and exit")
    args = ap.parse_args(argv)
    {"scan": cmd_scan, "backtest": cmd_backtest, "optimize": cmd_optimize,
     "paper": cmd_paper, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
