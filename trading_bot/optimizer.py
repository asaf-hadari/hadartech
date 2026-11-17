"""Self-improvement loop.

The bot keeps searching for better parameters while the average monthly return
is below the target. To avoid "improving" by memorising the past (overfitting):

1. Development / holdout split - the last `holdout_months` are NEVER used to
   choose parameters. They are only used to report an honest out-of-sample score.
2. Robust score - a candidate is scored on several separate time windows inside
   the development period; inconsistency across windows and deep drawdowns are
   penalised, and too few trades are rejected.
3. A new champion is adopted only if it beats the current one on that score.
"""
from dataclasses import replace
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest, monthly_returns
from .config import PARAM_SPACE, StrategyParams

MIN_TRADES = 30


def sample_params(rng, base=None, scale=None):
    """Random point in the search space; if `base` is given, a mutation around it."""
    vals = {}
    for k, (lo, hi, typ) in PARAM_SPACE.items():
        if base is None:
            v = rng.uniform(lo, hi)
        else:
            cur = getattr(base, k)
            if rng.random() > 0.5:  # mutate ~half of the parameters
                vals[k] = cur
                continue
            v = cur + rng.gauss(0, (hi - lo) * (scale or 0.15))
        v = min(max(v, lo), hi)
        vals[k] = int(round(v)) if typ is int else round(v, 4)
    p = replace(base or StrategyParams(), **vals)
    if p.slow_ema < p.fast_ema * 2:
        p = replace(p, slow_ema=min(PARAM_SPACE["slow_ema"][1], p.fast_ema * 2))
    return p


def split(data, holdout_months):
    cut = data.dates[-1] - pd.DateOffset(months=holdout_months)
    return data.slice(None, cut), cut


def robust_score(result, n_windows=4):
    m = result.metrics
    if not m or m.get("trades", 0) < MIN_TRADES:
        return -math.inf, {}
    mr = monthly_returns(result.equity)
    if len(mr) < n_windows * 3:
        return -math.inf, {}
    windows = [w.mean() for w in np.array_split(mr.to_numpy(), n_windows)]
    mean_w, std_w = float(np.mean(windows)), float(np.std(windows))
    dd_pen = max(0.0, -m["max_drawdown"] - 0.20) * 0.05  # penalise DD beyond 20%
    score = mean_w - 0.5 * std_w - dd_pen
    return score, {"window_means": [round(w, 4) for w in windows], "score": score}


def evaluate(data, params, capital, costs, start=None):
    res = run_backtest(data, params, capital, costs, start=start)
    score, info = robust_score(res)
    return score, res, info


class SelfImprover:
    def __init__(self, data, cfg, state_dir="bot_state", holdout_months=12, seed=None, log=print):
        self.data, self.cfg, self.log = data, cfg, log
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.champion_path = self.state_dir / "champion.json"
        self.rng = random.Random(seed)
        self.dev, self.cut = split(data, holdout_months)
        self.holdout_months = holdout_months

    def _eval_dev(self, p):
        return evaluate(self.dev, p, self.cfg.initial_capital, self.cfg.costs)

    def _eval_holdout(self, p):
        # run on full data but start trading at the cut so indicators are warmed up
        res = run_backtest(self.data, p, self.cfg.initial_capital, self.cfg.costs, start=self.cut)
        return res

    def load_champion(self):
        if self.champion_path.exists():
            d = json.loads(self.champion_path.read_text())
            return StrategyParams.from_dict(d["params"])
        return self.cfg.params

    def run(self, max_iters=300, patience=150, forever=False, max_hours=None):
        target = self.cfg.target_monthly_return
        best_p = self.load_champion()
        best_s, best_res, best_info = self._eval_dev(best_p)
        self.log(f"Start: dev score={best_s:.4f}  dev avg monthly="
                 f"{best_res.metrics.get('avg_monthly_return', float('nan')):.2%}  target={target:.0%}")
        history, since_improve, t_start, it = [], 0, time.time(), 0

        while True:
            it += 1
            explore = self.rng.random() < 0.3 or not math.isfinite(best_s)
            scale = 0.25 if since_improve > patience // 2 else 0.12
            cand = sample_params(self.rng, None if explore else best_p, scale)
            s, res, info = self._eval_dev(cand)
            if s > best_s:
                best_p, best_s, best_res, best_info = cand, s, res, info
                since_improve = 0
                m = res.metrics
                self.log(f"[{it}] new champion: score={s:.4f} avg monthly={m['avg_monthly_return']:.2%} "
                         f"maxDD={m['max_drawdown']:.1%} trades={m['trades']}")
                history.append({"iter": it, "score": s, "avg_monthly": m["avg_monthly_return"]})
                self._save(best_p, best_res, best_info, history, final=False)
            else:
                since_improve += 1

            dev_avg = best_res.metrics.get("avg_monthly_return", -1) if best_res.metrics else -1
            if dev_avg >= target:
                self.log(f"Target reached on development data at iteration {it}.")
                break
            if max_hours and time.time() - t_start > max_hours * 3600:
                self.log("Time budget exhausted.")
                break
            if not forever and (it >= max_iters or since_improve >= patience):
                self.log(f"Stopped after {it} iterations ({since_improve} without improvement).")
                break

        hold = self._eval_holdout(best_p)
        report = self._save(best_p, best_res, best_info, history, final=True, holdout=hold)
        return best_p, report

    def _save(self, p, dev_res, info, history, final, holdout=None):
        clean = lambda m: {k: (None if isinstance(v, float) and math.isnan(v) else float(v))
                           for k, v in m.items()}
        report = {
            "updated": pd.Timestamp.now().isoformat(timespec="seconds"),
            "params": p.to_dict(),
            "target_monthly_return": self.cfg.target_monthly_return,
            "development_period": [str(self.dev.dates[0].date()), str(self.cut.date())],
            "development_metrics": clean(dev_res.metrics),
            "robustness": info,
            "history": history[-200:],
        }
        if holdout is not None:
            report["holdout_period"] = [str(self.cut.date()), str(self.data.dates[-1].date())]
            report["holdout_metrics"] = clean(holdout.metrics) if holdout.metrics else {}
            hm = holdout.metrics.get("avg_monthly_return", float("nan")) if holdout.metrics else float("nan")
            report["target_met_out_of_sample"] = bool(hm >= self.cfg.target_monthly_return)
        self.champion_path.write_text(json.dumps(report, indent=2))
        return report
