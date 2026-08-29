"""Configuration validation and the live-trading safety gate."""

from __future__ import annotations

import pytest
import yaml

from tradebot.core.config import (
    Settings,
    enforce_live_gate,
    load_tunables,
)
from tradebot.core.errors import ConfigError, SafetyError
from tradebot.core.types import TradingMode


class TestShippedConfig:
    def test_default_config_is_valid(self, tunables):
        assert tunables.account.initial_capital == 200.0
        assert tunables.risk.max_margin_per_trade == 5.0
        assert tunables.risk.max_total_allocated_margin == 20.0
        assert tunables.risk.max_leverage == 5
        assert tunables.risk.max_concurrent_positions == 4
        assert tunables.scanner.top_markets == 25
        assert tunables.trade.max_duration_sec <= 3600

    def test_all_eight_strategies_are_configured(self, tunables):
        expected = {
            "momentum",
            "trend_following",
            "breakout",
            "mean_reversion",
            "volume_spike",
            "volatility_expansion",
            "vwap",
            "support_resistance",
        }
        assert set(tunables.strategies) == expected

    def test_scanner_weights_normalise_to_one(self, tunables):
        assert sum(tunables.scanner.weights.normalised().values()) == pytest.approx(1.0)

    def test_opportunity_weights_normalise_to_one(self, tunables):
        assert sum(tunables.opportunity.weights.normalised().values()) == pytest.approx(1.0)

    def test_panic_regime_permits_no_strategy(self, tunables):
        assert tunables.regime.weights_for("PANIC") == {}

    def test_every_regime_weight_names_a_configured_strategy(self, tunables):
        known = set(tunables.strategies)
        for regime, weights in tunables.regime.strategy_weights.items():
            unknown = set(weights) - known
            assert not unknown, f"regime {regime} references unknown strategies {unknown}"

    def test_max_trade_duration_never_exceeds_sixty_minutes(self, tunables):
        assert tunables.trade.max_duration_sec <= 3600


class TestValidation:
    def test_risk_per_trade_outside_its_own_bounds_is_rejected(self):
        with pytest.raises(Exception):
            load_tunables_from_dict({"risk": {"risk_per_trade": 0.5, "max_risk_per_trade": 0.02}})

    def test_hourly_loss_above_daily_loss_is_rejected(self):
        with pytest.raises(Exception):
            load_tunables_from_dict({"risk": {"max_hourly_loss": 0.05, "max_daily_loss": 0.02}})

    def test_stop_tighter_than_round_trip_fee_is_rejected(self):
        """A stop inside the fee cannot produce a profitable trade, ever."""
        with pytest.raises(Exception):
            load_tunables_from_dict(
                {
                    "stops": {"min_stop_pct": 0.0001, "max_stop_pct": 0.01},
                    "edge": {"taker_fee": 0.0004},
                }
            )

    def test_unknown_key_is_rejected(self):
        with pytest.raises(Exception):
            load_tunables_from_dict({"risk": {"totally_made_up": 1}})

    def test_missing_file_raises_config_error(self):
        with pytest.raises(ConfigError):
            load_tunables("config/does-not-exist.yaml")

    def test_malformed_yaml_raises_config_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("risk: [unclosed\n")
        with pytest.raises(ConfigError):
            load_tunables(bad)


class TestLiveGate:
    """Reaching LIVE must require three independent confirmations."""

    def _settings(self, **overrides):
        base = {
            "trading_mode": TradingMode.LIVE,
            "i_understand_live_trading_risk": "YES",
            "binance_api_key": "key",
            "binance_api_secret": "secret",
            "binance_testnet": False,
        }
        base.update(overrides)
        return Settings(**base)

    def test_all_three_confirmations_present_passes(self):
        enforce_live_gate(self._settings(), live_flag=True)

    def test_missing_cli_flag_blocks(self):
        with pytest.raises(SafetyError, match="--live"):
            enforce_live_gate(self._settings(), live_flag=False)

    def test_missing_acknowledgement_blocks(self):
        with pytest.raises(SafetyError, match="I_UNDERSTAND"):
            enforce_live_gate(self._settings(i_understand_live_trading_risk="NO"), live_flag=True)

    def test_missing_credentials_block(self):
        with pytest.raises(SafetyError, match="BINANCE_API_KEY"):
            enforce_live_gate(
                self._settings(binance_api_key="", binance_api_secret=""), live_flag=True
            )

    def test_testnet_with_live_mode_is_contradictory(self):
        with pytest.raises(SafetyError, match="TESTNET"):
            enforce_live_gate(self._settings(binance_testnet=True), live_flag=True)

    def test_live_flag_without_live_mode_is_refused(self):
        """Never guess the operator's intent when the switches disagree."""
        settings = Settings(trading_mode=TradingMode.PAPER)
        with pytest.raises(SafetyError):
            enforce_live_gate(settings, live_flag=True)

    def test_paper_mode_needs_no_confirmation(self):
        enforce_live_gate(Settings(trading_mode=TradingMode.PAPER), live_flag=False)

    def test_backtest_mode_needs_no_confirmation(self):
        enforce_live_gate(Settings(trading_mode=TradingMode.BACKTEST), live_flag=False)


class TestSecretsHandling:
    def test_redacted_never_contains_the_secret(self):
        settings = Settings(binance_api_secret="hunter2-hunter2", binance_api_key="abc123")
        blob = str(settings.redacted())
        assert "hunter2-hunter2" not in blob
        assert "abc123" not in blob
        assert "<set:" in blob

    def test_unset_secret_is_marked_unset(self):
        assert Settings(binance_api_key="").redacted()["binance_api_key"] == "<unset>"


def load_tunables_from_dict(data: dict, tmp: str = "/tmp/_cfg_test.yaml") -> object:
    from pathlib import Path

    path = Path(tmp)
    path.write_text(yaml.safe_dump(data))
    try:
        return load_tunables(path)
    finally:
        path.unlink(missing_ok=True)
