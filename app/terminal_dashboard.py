# ============================================================
# ชื่อโค้ด: Production Runtime Terminal Dashboard
# ที่อยู่ไฟล์: app/terminal_dashboard.py
# คำสั่งรัน: python app/terminal_dashboard.py
# เวอร์ชัน: v1.3.0
# ============================================================

"""
terminal_dashboard.py
Version: v1.3.0

Purpose:
- Production terminal dashboard for runtime/dashboard_state.json
- Reads only the current production schema
- No demo provider
- No fake fallback values
- Safe under missing or stale state conditions

CHANGELOG (v1.3.0)
------------------------------------------------------------
- REWRITE: dashboard reader to match production state schema
           meta/runtime/market/indicators/signal/guard/report/position/logs
- REMOVE: legacy schema merge logic (header/data_ingest/live_analytics/...)
- KEEP: terminal-only dashboard via Rich
- ADD: stale/missing file detection and explicit degraded status
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

APP_TITLE = "PRODUCTION TRADE DASHBOARD"
APP_VERSION = "v1.3.0"
REFRESH_PER_SECOND = 4
STATE_STALE_SECONDS = 8.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "runtime" / "dashboard_state.json"

console = Console()


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, precision: int = 2, default: str = "--") -> str:
    number = _to_float(value)
    if number is None:
        return default
    return f"{number:.{precision}f}"


def _fmt_text(value: Any, default: str = "--") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "--"


def _style_status(value: str) -> str:
    upper = value.upper()
    if any(key in upper for key in ["APPROVED", "RUNNING", "ACTIVE", "OK", "HOLD", "CANDIDATE", "POSITION", "MONITOR"]):
        return "bold bright_green"
    if any(key in upper for key in ["BLOCK", "REJECT", "ERROR", "EXIT", "STALE", "MISSING"]):
        return "bold red"
    if any(key in upper for key in ["STARTUP", "UPDATED", "DUPLICATE", "SKIPPED", "DEGRADED"]):
        return "bold yellow"
    return "white"


def _load_state() -> tuple[dict[str, Any], str, float | None]:
    if not STATE_PATH.exists():
        return {}, "MISSING", None

    try:
        raw_text = STATE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception as exc:
        return {"error": str(exc)}, "ERROR", None

    mtime = STATE_PATH.stat().st_mtime
    age = max(0.0, time.time() - mtime)
    if age > STATE_STALE_SECONDS:
        return data, "STALE", age
    return data, "OK", age


def _table(rows: list[tuple[str, str, str]], title: str) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=12, style="bold white")
    table.add_column(ratio=1, style="white")
    table.add_column(ratio=20, style="white")
    for key, value, style in rows:
        table.add_row(key, ":", Text(value, style=style))
    return Panel(table, title=Text(title, style="bold bright_cyan"), border_style="bright_cyan", box=box.ROUNDED)


def _pretty_mapping(mapping: Mapping[str, Any] | None) -> str:
    if not mapping:
        return "--"
    parts: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def _build_header(data: dict[str, Any], file_status: str, age_seconds: float | None) -> Panel:
    runtime = data.get("runtime", {})
    meta = data.get("meta", {})
    status_value = _fmt_text(runtime.get("status"), file_status)
    rows = [
        ("Time", _now_text(), "white"),
        ("File", file_status if file_status != "OK" else "LIVE", _style_status(file_status if file_status != "OK" else status_value)),
        ("Runtime", status_value, _style_status(status_value)),
        ("App", _fmt_text(runtime.get("app_name")), "white"),
        ("Env", _fmt_text(runtime.get("environment")), "white"),
        ("Symbol", _fmt_text(runtime.get("active_symbol")), "bold bright_green"),
        ("TF", _fmt_text(runtime.get("timeframe")), "white"),
        ("Updated", _fmt_text(meta.get("updated_at")), "white"),
        ("AgeSec", _fmt(age_seconds, 1), "white"),
    ]
    return _table(rows, f"{APP_TITLE} {APP_VERSION}")


def _build_market(data: dict[str, Any]) -> Panel:
    market = data.get("market", {})
    rows = [
        ("Symbol", _fmt_text(market.get("symbol")), "white"),
        ("Bar Time", _fmt_text(market.get("bar_time")), "white"),
        ("Open", _fmt(market.get("open")), "white"),
        ("High", _fmt(market.get("high")), "white"),
        ("Low", _fmt(market.get("low")), "white"),
        ("Close", _fmt(market.get("close")), "bold bright_green"),
        ("Bid", _fmt(market.get("bid")), "white"),
        ("Ask", _fmt(market.get("ask")), "white"),
        ("Spread", _fmt(market.get("spread"), 3), "white"),
    ]
    return _table(rows, "Market")


def _build_indicators(data: dict[str, Any]) -> Panel:
    indicators = data.get("indicators", {})
    rows = [
        ("EMA20", _fmt(indicators.get("ema20")), "white"),
        ("EMA50", _fmt(indicators.get("ema50")), "white"),
        ("EMA200", _fmt(indicators.get("ema200")), "white"),
        ("EMA20 Slope", _fmt(indicators.get("ema20_slope"), 4), "white"),
        ("RSI14", _fmt(indicators.get("rsi14"), 2), "white"),
        ("MACD Hist", _fmt(indicators.get("macd_histogram"), 4), "white"),
        ("ADX14", _fmt(indicators.get("adx14"), 2), "white"),
        ("DI+", _fmt(indicators.get("di_plus"), 2), "white"),
        ("DI-", _fmt(indicators.get("di_minus"), 2), "white"),
        ("ATR", _fmt(indicators.get("atr"), 2), "white"),
        ("BB Upper", _fmt(indicators.get("bb_upper")), "white"),
        ("BB Mid", _fmt(indicators.get("bb_mid")), "white"),
        ("BB Lower", _fmt(indicators.get("bb_lower")), "white"),
    ]
    return _table(rows, "Indicators")


def _build_signal_guard(data: dict[str, Any]) -> Panel:
    signal = data.get("signal", {})
    guard = data.get("guard", {})
    rows = [
        ("Signal Status", _fmt_text(signal.get("status")), _style_status(_fmt_text(signal.get("status"), "--"))),
        ("Side", _fmt_text(signal.get("side")), "bold bright_green" if _fmt_text(signal.get("side")) == "BUY" else "bold red" if _fmt_text(signal.get("side")) == "SELL" else "white"),
        ("Score", _fmt(signal.get("score"), 4), "white"),
        ("Entry", _fmt(signal.get("entry")), "white"),
        ("SL", _fmt(signal.get("sl")), "white"),
        ("TP", _fmt(signal.get("tp")), "white"),
        ("Reason", _fmt_text(signal.get("reason")), "white"),
        ("Guard", _fmt_text(guard.get("status")), _style_status(_fmt_text(guard.get("status"), "--"))),
        ("Guard Reason", _fmt_text(guard.get("reason")), "white"),
        ("Spread OK", _fmt_bool(guard.get("spread_ok")), "white"),
        ("RR OK", _fmt_bool(guard.get("rr_ok")), "white"),
        ("Cooldown OK", _fmt_bool(guard.get("cooldown_ok")), "white"),
    ]
    return _table(rows, "Signal + Guard")


def _build_position(data: dict[str, Any]) -> Panel:
    position = data.get("position", {})
    rows = [
        ("Ticket", _fmt_text(position.get("ticket")), "white"),
        ("Symbol", _fmt_text(position.get("symbol")), "white"),
        ("Side", _fmt_text(position.get("side")), "white"),
        ("Entry", _fmt(position.get("entry_price")), "white"),
        ("Current", _fmt(position.get("current_price")), "white"),
        ("SL", _fmt(position.get("sl")), "white"),
        ("TP", _fmt(position.get("tp")), "white"),
        ("PnL", _fmt(position.get("pnl"), 2), "bold bright_green"),
        ("Exit", _fmt_text(position.get("exit_decision")), _style_status(_fmt_text(position.get("exit_decision"), "--"))),
        ("Exit Reason", _fmt_text(position.get("exit_reason")), "white"),
        ("Close Tried", _fmt_bool(position.get("close_attempted")), "white"),
        ("Close Result", _fmt_text(position.get("close_result")), "white"),
    ]
    return _table(rows, "Position")


def _build_report_logs(data: dict[str, Any]) -> Panel:
    report = data.get("report", {})
    logs = data.get("logs", {})
    body = Table.grid(expand=True)
    body.add_column(ratio=1)
    body.add_row(Text("Summary", style="bold white"))
    body.add_row(Text(_pretty_mapping(report.get("summary"))[:220], style="white"))
    body.add_row(Text("Last Event", style="bold white"))
    body.add_row(Text(_pretty_mapping(report.get("last_event"))[:220], style="white"))
    body.add_row(Text("Last Decision At", style="bold white"))
    body.add_row(Text(_fmt_text(report.get("last_decision_at")), style="white"))
    body.add_row(Text("Recent Logs", style="bold white"))
    recent = logs.get("recent") or []
    if recent:
        for item in recent[-5:]:
            body.add_row(Text(_pretty_mapping(item)[:220], style="white"))
    else:
        body.add_row(Text("--", style="grey70"))
    return Panel(body, title=Text("Report + Logs", style="bold bright_cyan"), border_style="bright_cyan", box=box.ROUNDED)


def build_dashboard() -> Layout:
    data, file_status, age_seconds = _load_state()

    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=11),
        Layout(name="middle", ratio=1),
        Layout(name="bottom", ratio=1),
    )
    layout["middle"].split_row(
        Layout(name="market"),
        Layout(name="indicators"),
        Layout(name="signal_guard"),
    )
    layout["bottom"].split_row(
        Layout(name="position"),
        Layout(name="report_logs"),
    )

    layout["header"].update(_build_header(data, file_status, age_seconds))
    layout["market"].update(_build_market(data))
    layout["indicators"].update(_build_indicators(data))
    layout["signal_guard"].update(_build_signal_guard(data))
    layout["position"].update(_build_position(data))
    layout["report_logs"].update(_build_report_logs(data))
    return layout


def main() -> int:
    console.clear()
    try:
        with Live(build_dashboard(), console=console, refresh_per_second=REFRESH_PER_SECOND, screen=True) as live:
            while True:
                live.update(build_dashboard())
                time.sleep(1 / REFRESH_PER_SECOND)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
