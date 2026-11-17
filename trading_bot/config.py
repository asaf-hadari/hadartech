"""Default configuration. Every value can be overridden from the CLI or a JSON file."""
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path

STATE_DIR = Path("bot_state")


@dataclass
class StrategyParams:
    fast_ema: int = 20            # short trend EMA
    slow_ema: int = 100           # long trend EMA
    adx_period: int = 14
    adx_min: float = 20.0         # minimal trend strength to enter
    atr_period: int = 14
    atr_stop_mult: float = 3.0    # trailing stop = highest close - mult * ATR
    momentum_lookback: int = 63   # ~3 months, used for ranking candidates
    max_positions: int = 10
    risk_per_trade: float = 0.01  # max equity lost if the stop is hit (1%)
    breadth_min: float = 0.40     # market filter: % of universe above slow EMA

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


# Search space used by the optimizer: (min, max, type)
PARAM_SPACE = {
    "fast_ema": (5, 50, int),
    "slow_ema": (50, 250, int),
    "adx_min": (10.0, 35.0, float),
    "atr_stop_mult": (1.5, 6.0, float),
    "momentum_lookback": (20, 180, int),
    "max_positions": (3, 20, int),
    "risk_per_trade": (0.005, 0.03, float),
    "breadth_min": (0.0, 0.7, float),
}


@dataclass
class BrokerCosts:
    commission_pct: float = 0.001  # 0.1% per side
    slippage_pct: float = 0.0005   # 0.05% per side
    min_commission: float = 1.0


@dataclass
class BotConfig:
    market: str = "nasdaq100"           # nasdaq100 | nasdaq_all | ta35 | custom
    initial_capital: float = 100_000.0
    target_monthly_return: float = 0.10  # the 10% goal from the spec
    params: StrategyParams = field(default_factory=StrategyParams)
    costs: BrokerCosts = field(default_factory=BrokerCosts)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(
            market=d.get("market", "nasdaq100"),
            initial_capital=d.get("initial_capital", 100_000.0),
            target_monthly_return=d.get("target_monthly_return", 0.10),
            params=StrategyParams.from_dict(d.get("params", {})),
            costs=BrokerCosts(**d.get("costs", {})),
        )
