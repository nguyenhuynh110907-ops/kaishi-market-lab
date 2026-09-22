from decimal import Decimal

import pytest

from kaishi_bot.dashboard_models import RuntimeMode
from kaishi_bot.safety import SafetyError, SafetyGate


def test_all_non_armed_states_reject_writes() -> None:
    gate = SafetyGate(credentials_available=True)
    for mode in (RuntimeMode.PAPER, RuntimeMode.READ_ONLY, RuntimeMode.LIVE):
        gate.set_mode(mode)
        with pytest.raises(SafetyError):
            gate.authorize_write(Decimal("1"), Decimal("0"))


def test_live_challenge_arms_and_mode_change_disarms() -> None:
    now = [100.0]
    gate = SafetyGate(credentials_available=True, clock=lambda: now[0])
    gate.set_mode(RuntimeMode.LIVE)
    phrase = gate.create_challenge()
    gate.arm(phrase)
    gate.authorize_write(Decimal("200"), Decimal("9800"))
    gate.set_mode(RuntimeMode.READ_ONLY)
    assert gate.armed is False


def test_challenge_expiry_stale_data_and_risk_caps_fail_closed() -> None:
    now = [100.0]
    gate = SafetyGate(credentials_available=True, clock=lambda: now[0])
    gate.set_mode(RuntimeMode.LIVE)
    phrase = gate.create_challenge()
    now[0] += 61
    with pytest.raises(SafetyError, match="expired"):
        gate.arm(phrase)
    phrase = gate.create_challenge()
    gate.arm(phrase)
    with pytest.raises(SafetyError, match="per-entry"):
        gate.authorize_write(Decimal("200.01"), Decimal("0"))
    with pytest.raises(SafetyError, match="daily"):
        gate.authorize_write(Decimal("1"), Decimal("10000"))
    gate.mark_data_stale()
    assert gate.armed is False


def test_missing_credentials_cannot_create_live_challenge() -> None:
    gate = SafetyGate(credentials_available=False)
    gate.set_mode(RuntimeMode.LIVE)
    with pytest.raises(SafetyError, match="credentials"):
        gate.create_challenge()


def test_disarmed_live_allows_only_reduce_only_protection() -> None:
    gate = SafetyGate(credentials_available=True)
    gate.set_mode(RuntimeMode.LIVE)
    gate.authorize_write(Decimal("0"), Decimal("0"), reduce_only=True)
    with pytest.raises(SafetyError, match="not armed"):
        gate.authorize_write(Decimal("0.01"), Decimal("0"), reduce_only=False)
