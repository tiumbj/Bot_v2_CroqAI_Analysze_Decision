from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_state_writer import DashboardStateWriter

from models.schemas import (  # noqa: E402
    AISettings,
    AppSettings,
    ModelSettings,
    RiskSettings,
    SymbolRegistry,
)
from storage.db import DatabaseManager  # noqa: E402


@dataclass(frozen=True)
class RuntimeContext:
    app_settings: AppSettings
    risk_settings: RiskSettings
    ai_settings: AISettings
    model_settings: ModelSettings
    symbol_registry: SymbolRegistry


def load_yaml(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing config file: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a YAML object: {file_path}")
    return content


def build_runtime_context() -> RuntimeContext:
    config_dir = PROJECT_ROOT / "config"

    app_settings = AppSettings.model_validate(load_yaml(config_dir / "settings.yaml"))
    risk_settings = RiskSettings.model_validate(load_yaml(config_dir / "risk.yaml"))
    ai_settings = AISettings.model_validate(load_yaml(config_dir / "ai.yaml"))
    model_settings = ModelSettings.model_validate(load_yaml(config_dir / "model.yaml"))
    symbol_registry = SymbolRegistry.model_validate(load_yaml(config_dir / "symbol.yaml"))

    configured_symbols = set(app_settings.symbols)
    registered_symbols = set(symbol_registry.symbols.keys())

    missing_symbols = sorted(configured_symbols - registered_symbols)
    if missing_symbols:
        raise ValueError(
            "Missing symbol contract(s) in symbol.yaml: " + ", ".join(missing_symbols)
        )

    return RuntimeContext(
        app_settings=app_settings,
        risk_settings=risk_settings,
        ai_settings=ai_settings,
        model_settings=model_settings,
        symbol_registry=symbol_registry,
    )


def initialize_storage(runtime: RuntimeContext) -> None:
    db_path = PROJECT_ROOT / runtime.app_settings.sqlite_path
    db_manager = DatabaseManager(str(db_path))
    db_manager.initialize()

    log_dir = PROJECT_ROOT / runtime.app_settings.log_directory
    log_dir.mkdir(parents=True, exist_ok=True)


def print_runtime_summary(runtime: RuntimeContext) -> None:
    summary = {
        "app_name": runtime.app_settings.app_name,
        "environment": runtime.app_settings.environment.value,
        "symbols": runtime.app_settings.symbols,
        "timeframe": runtime.app_settings.timeframe,
        "dry_run": runtime.app_settings.dry_run,
        "database": runtime.app_settings.sqlite_path,
        "meta_model_enabled": runtime.model_settings.meta_model_enabled,
        "ai_model": runtime.ai_settings.model_name,
        "risk_per_trade_pct": runtime.risk_settings.risk_per_trade_pct,
        "minimum_rr": runtime.risk_settings.minimum_rr,
        "registered_symbol_contracts": sorted(runtime.symbol_registry.symbols.keys()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_processed_setup_state(file_path: Path) -> tuple[list[str], set[str]]:
    if not file_path.exists():
        return ([], set())

    try:
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return ([], set())

    raw_items: list[Any]
    if isinstance(payload, dict):
        raw_items = payload.get("items", [])
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    ordered_items: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered_items.append(value)

    return (ordered_items, seen)


def _save_processed_setup_state(file_path: Path, ordered_setup_ids: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
    payload = {"items": ordered_setup_ids}

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    temp_path.replace(file_path)


def _remember_processed_setup(
    setup_id: str,
    ordered_setup_ids: list[str],
    processed_setup_ids: set[str],
    file_path: Path,
    max_items: int,
) -> None:
    if setup_id in processed_setup_ids:
        return

    ordered_setup_ids.append(setup_id)
    processed_setup_ids.add(setup_id)

    if max_items > 0 and len(ordered_setup_ids) > max_items:
        trimmed_items = ordered_setup_ids[-max_items:]
        ordered_setup_ids[:] = trimmed_items
        processed_setup_ids.clear()
        processed_setup_ids.update(trimmed_items)

    _save_processed_setup_state(file_path, ordered_setup_ids)


def _extract_runtime_setup_id(
    candidate: dict[str, Any],
    default_symbol: str,
    default_timeframe: str,
) -> str:
    metadata = candidate.get("metadata", {}) or {}
    bar_time = candidate.get("bar_time", "")
    direction = candidate.get("decision") or candidate.get("direction") or ""
    return (
        candidate.get("setup_key")
        or metadata.get("candidate_id")
        or f"{default_symbol}|{default_timeframe}|{bar_time}|{direction}"
    )


def _format_runtime_accept_line(candidate: dict[str, Any]) -> str:
    symbol = (
        candidate.get("display_symbol")
        or candidate.get("runtime_symbol")
        or candidate.get("canonical_symbol")
        or "unknown_symbol"
    )
    direction = candidate.get("decision") or candidate.get("direction") or ""
    guard = str(candidate.get("guard", "") or "").strip()

    try:
        score_text = f"{float(candidate.get('score', 0.0) or 0.0):.2f}"
    except (TypeError, ValueError):
        score_text = "0.00"

    def _price_text(value: Any) -> str:
        try:
            return f"{float(value or 0.0):.2f}"
        except (TypeError, ValueError):
            return "0.00"

    line = (
        f"[{symbol}] "
        f"APPROVED={direction} "
        f"SCORE={score_text} "
        f"ENTRY={_price_text(candidate.get('entry'))} "
        f"SL={_price_text(candidate.get('sl'))} "
        f"TP={_price_text(candidate.get('tp'))} "
        f"CANONICAL={candidate.get('canonical_symbol', 'UNKNOWN')} "
        f"SETUP={candidate.get('setup_key', '')}"
    )
    if guard:
        line += f" GUARD={guard}"
    return line


def _format_runtime_reject_line(candidate: dict[str, Any]) -> str:
    symbol = (
        candidate.get("display_symbol")
        or candidate.get("runtime_symbol")
        or candidate.get("canonical_symbol")
        or "unknown_symbol"
    )
    line = (
        f"[{symbol}] "
        f"REJECTED={candidate.get('status', 'rejected')} "
        f"REASON={candidate.get('reason', '')} "
        f"CANONICAL={candidate.get('canonical_symbol', 'UNKNOWN')} "
        f"SETUP={candidate.get('setup_key', '')}"
    )
    duplicate_of = str(candidate.get("duplicate_of", "") or "").strip()
    if duplicate_of:
        line += f" DUPLICATE_OF={duplicate_of}"
    return line


def _format_runtime_summary_line(
    *,
    processed_symbols: int,
    input_candidates: int,
    approved_candidates: int,
    rejected_candidates: int,
    duplicates_blocked: int,
    unique_underlying_setups: int,
    timeframe: str,
) -> str:
    tf = str(timeframe or "NA").upper().strip() or "NA"
    return (
        f"SUMMARY processed={processed_symbols} "
        f"input_candidates={input_candidates} "
        f"approved={approved_candidates} "
        f"rejected={rejected_candidates} "
        f"duplicates_blocked={duplicates_blocked} "
        f"unique_underlying_setups={unique_underlying_setups} "
        f"timeframe={tf}"
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _extract_last_bar_payload(frame_data: Any) -> dict[str, Any]:
    try:
        if frame_data is None or len(frame_data) == 0:
            return {}
    except Exception:
        return {}

    try:
        last_row = frame_data.iloc[-1]
    except Exception:
        return {}

    payload: dict[str, Any] = {
        "open": _safe_float(last_row.get("open")),
        "high": _safe_float(last_row.get("high")),
        "low": _safe_float(last_row.get("low")),
        "close": _safe_float(last_row.get("close")),
    }

    bar_time = None
    try:
        row_time = last_row.get("time")
        if row_time is not None:
            bar_time = _safe_iso_timestamp(row_time)
    except Exception:
        bar_time = None

    if not bar_time:
        try:
            index_value = frame_data.index[-1]
            bar_time = _safe_iso_timestamp(index_value)
        except Exception:
            bar_time = None

    payload["bar_time"] = bar_time
    return payload


def _build_market_payload(
    *,
    symbol: str,
    timeframe: str,
    market_frame: Any,
    tick: Any | None = None,
    spread: float | None = None,
) -> dict[str, Any]:
    payload = _extract_last_bar_payload(getattr(market_frame, "data", None))
    payload["symbol"] = symbol
    payload["timeframe"] = timeframe

    if tick is not None:
        payload["bid"] = _safe_float(getattr(tick, "bid", None))
        payload["ask"] = _safe_float(getattr(tick, "ask", None))
        if spread is None:
            spread = _safe_float(getattr(tick, "spread_points", None))

    payload["spread"] = _safe_float(spread)
    return payload


def _snapshot_attr(snapshot: Any, *names: str) -> Any:
    for name in names:
        value = getattr(snapshot, name, None)
        if value is not None:
            return value
    return None


def _build_indicator_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "ema20": _safe_float(_snapshot_attr(snapshot, "ema_20", "ema20")),
        "ema50": _safe_float(_snapshot_attr(snapshot, "ema_50", "ema50")),
        "ema200": _safe_float(_snapshot_attr(snapshot, "ema_200", "ema200")),
        "ema20_slope": _safe_float(_snapshot_attr(snapshot, "ema20_slope", "ema_20_slope")),
        "rsi14": _safe_float(_snapshot_attr(snapshot, "rsi_14", "rsi14")),
        "macd_histogram": _safe_float(_snapshot_attr(snapshot, "macd_histogram", "macd_hist")),
        "adx14": _safe_float(_snapshot_attr(snapshot, "adx_14", "adx14")),
        "di_plus": _safe_float(
            _snapshot_attr(
                snapshot,
                "di_plus_14",
                "di_plus",
                "plus_di_14",
                "plus_di",
                "pdi",
            )
        ),
        "di_minus": _safe_float(
            _snapshot_attr(
                snapshot,
                "di_minus_14",
                "di_minus",
                "minus_di_14",
                "minus_di",
                "mdi",
            )
        ),
        "atr": _safe_float(_snapshot_attr(snapshot, "atr_14", "atr14", "atr")),
        "bb_upper": _safe_float(_snapshot_attr(snapshot, "bb_upper", "bollinger_upper")),
        "bb_mid": _safe_float(_snapshot_attr(snapshot, "bb_mid", "bb_middle", "bollinger_mid")),
        "bb_lower": _safe_float(_snapshot_attr(snapshot, "bb_lower", "bollinger_lower")),
    }


def _build_signal_payload(candidate: dict[str, Any], *, status: str, reason: str | None) -> dict[str, Any]:
    side = candidate.get("decision") or candidate.get("direction")
    return {
        "status": status,
        "side": side,
        "score": _safe_float(candidate.get("score")),
        "entry": _safe_float(candidate.get("entry")),
        "sl": _safe_float(candidate.get("sl")),
        "tp": _safe_float(candidate.get("tp")),
        "reason": reason,
        "request_id": candidate.get("setup_key"),
    }


def _build_guard_payload(
    *,
    status: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "status": status,
        "reason": reason,
        "spread_ok": metadata.get("spread_ok"),
        "rr_ok": metadata.get("rr_ok"),
        "cooldown_ok": metadata.get("cooldown_ok"),
    }


def main() -> int:
    try:
        runtime = build_runtime_context()
        initialize_storage(runtime)

        dashboard_state_path = PROJECT_ROOT / "runtime" / "dashboard_state.json"
        dashboard_writer = DashboardStateWriter(state_path=dashboard_state_path)
        dashboard_writer.bootstrap(
            app_name=runtime.app_settings.app_name,
            environment=runtime.app_settings.environment.value,
            timeframe=runtime.app_settings.timeframe,
            symbols=runtime.app_settings.symbols,
            dry_run=runtime.app_settings.dry_run,
            status="running",
        )

        print_runtime_summary(runtime)
        import os

        if os.getenv("RUN_POSITION_MONITOR") == "1":
            print("RUN_POSITION_MONITOR=1 - production position monitor mode enabled")
            import time
            from datetime import datetime
            from core.feature_engine import FeatureEngine
            from core.market_data import MarketDataService
            from core.mt5_gateway import MT5Gateway

            try:
                import MetaTrader5 as mt5
            except Exception:
                mt5 = None

            interval_raw = os.getenv("POSITION_MONITOR_INTERVAL_SECONDS", "").strip()
            interval_seconds = 10
            if interval_raw:
                try:
                    interval_seconds = max(int(interval_raw), 1)
                except ValueError:
                    interval_seconds = 10

            max_cycles_raw = os.getenv("POSITION_MONITOR_MAX_CYCLES", "").strip()
            max_cycles: int | None = None
            if max_cycles_raw:
                try:
                    max_cycles = max(int(max_cycles_raw), 1)
                except ValueError:
                    max_cycles = None

            min_profit_points_raw = os.getenv("POSITION_MONITOR_MIN_PROFIT_POINTS", "").strip()
            min_profit_points = 50.0
            if min_profit_points_raw:
                try:
                    min_profit_points = float(min_profit_points_raw)
                except ValueError:
                    min_profit_points = 50.0

            execution_enabled = os.getenv("ENABLE_POSITION_CLOSE_EXECUTION") == "1"

            monitor_log_path = PROJECT_ROOT / "storage" / "position_monitor.jsonl"
            monitor_log_path.parent.mkdir(parents=True, exist_ok=True)

            reported_position_failures: set[str] = set()
            reported_market_failures: set[str] = set()
            closed_tickets: set[Any] = set()

            gateway = MT5Gateway()
            gateway.initialize()
            gateway.ensure_connection()
            market_data = MarketDataService(
                gateway=gateway,
                timeframe=runtime.app_settings.timeframe,
                max_bars_fetch=runtime.app_settings.max_bars_fetch,
            )
            feature_engine = FeatureEngine()

            def append_monitor_log(payload: dict[str, Any]) -> None:
                with monitor_log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(payload, ensure_ascii=False) + "\n")

                try:
                    dashboard_writer.append_log(payload, status="position_monitor")
                    dashboard_writer.update_position(payload, status="position_monitor")
                except Exception as exc:
                    print(f"[DASHBOARD-WRITER] append_monitor_log failed: {exc}")

            def eval_exit_action(snapshot: Any, side: str, profit_points: float) -> tuple[str, str]:
                close = float(getattr(snapshot, "close", 0.0) or 0.0)
                ema20 = float(getattr(snapshot, "ema_20", 0.0) or 0.0)
                ema50 = float(getattr(snapshot, "ema_50", 0.0) or 0.0)
                ema20_slope = float(getattr(snapshot, "ema20_slope", 0.0) or 0.0)
                rsi = float(getattr(snapshot, "rsi_14", 0.0) or 0.0)
                macd_hist = float(getattr(snapshot, "macd_histogram", 0.0) or 0.0)

                if side == "BUY":
                    force = close < ema50 and ema20_slope < 0.0 and macd_hist < 0.0
                    if force:
                        return ("FORCE_EXIT", "force_reversal_bearish")

                    early_signal = close < ema20 and (ema20_slope < 0.0 or macd_hist < 0.0 or rsi < 50.0)
                    if profit_points >= min_profit_points and profit_points > 0.0 and early_signal:
                        return ("CLOSE_EARLY", "profit_protect_reversal_bearish")

                    return ("HOLD", "no_exit_signal")

                if side == "SELL":
                    force = close > ema50 and ema20_slope > 0.0 and macd_hist > 0.0
                    if force:
                        return ("FORCE_EXIT", "force_reversal_bullish")

                    early_signal = close > ema20 and (ema20_slope > 0.0 or macd_hist > 0.0 or rsi > 50.0)
                    if profit_points >= min_profit_points and profit_points > 0.0 and early_signal:
                        return ("CLOSE_EARLY", "profit_protect_reversal_bullish")

                    return ("HOLD", "no_exit_signal")

                return ("HOLD", "unknown_side")

            def close_position_by_ticket(
                ticket: Any,
                symbol: str,
                side: str,
                volume: float,
            ) -> tuple[bool, str]:
                if mt5 is None:
                    return (False, "mt5_module_unavailable")

                try:
                    resolved = gateway.ensure_symbol_selected(symbol)
                    tick = mt5.symbol_info_tick(resolved)
                    if tick is None:
                        code, message = mt5.last_error()
                        return (False, f"tick_unavailable code={code} message={message}")

                    order_type = mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY
                    price = float(tick.bid) if side == "BUY" else float(tick.ask)

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": resolved,
                        "volume": float(volume),
                        "type": order_type,
                        "position": int(ticket),
                        "price": price,
                        "deviation": 20,
                        "comment": "position_monitor_close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    if result is None:
                        code, message = mt5.last_error()
                        return (False, f"order_send_failed code={code} message={message}")

                    retcode = getattr(result, "retcode", None)
                    if retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                        return (True, f"retcode={retcode}")
                    return (False, f"retcode={retcode}")
                except Exception as exc:
                    return (False, str(exc))

            monitor_symbols: list[str] = []
            for symbol in runtime.app_settings.symbols:
                try:
                    gateway.ensure_symbol_selected(symbol)
                    monitor_symbols.append(symbol)
                except Exception:
                    if symbol not in reported_position_failures:
                        reported_position_failures.add(symbol)
                        print(
                            f"[MONITOR] {symbol} side=NA action=HOLD ticket=NA "
                            f"reason=symbol_unavailable"
                        )

            if not monitor_symbols:
                print("[MONITOR-SUMMARY] positions=0")
                return 0

            cycle = 0
            try:
                while True:
                    cycle += 1
                    positions_count = 0
                    close_signals = 0
                    executed = 0

                    for symbol in monitor_symbols:
                        try:
                            positions = gateway.get_positions_by_symbol(symbol)
                        except Exception:
                            if symbol not in reported_position_failures:
                                reported_position_failures.add(symbol)
                                print(
                                    f"[MONITOR] {symbol} side=NA action=HOLD ticket=NA "
                                    f"reason=positions_unavailable"
                                )
                            continue

                        if not positions:
                            continue

                        for position in positions:
                            positions_count += 1

                            pos_symbol = str(position.get("symbol") or symbol)
                            ticket = position.get("ticket")
                            pos_type = position.get("type")
                            side = "BUY" if pos_type in (0, "0", "buy", "BUY") else "SELL"
                            volume = float(position.get("volume", 0.0) or 0.0)
                            price_open = float(position.get("price_open", 0.0) or 0.0)
                            sl = position.get("sl")
                            tp = position.get("tp")

                            try:
                                tick = gateway.get_tick(pos_symbol)
                                current_price = float(tick.bid) if side == "BUY" else float(tick.ask)
                                spread_points = float(tick.spread_points)
                            except Exception:
                                if pos_symbol not in reported_market_failures:
                                    reported_market_failures.add(pos_symbol)
                                action = "HOLD"
                                reason = "tick_unavailable"
                                print(
                                    f"[MONITOR] {pos_symbol} side={side} action={action} ticket={ticket} "
                                    f"reason={reason}"
                                )
                                append_monitor_log(
                                    {
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "ticket": ticket,
                                        "symbol": pos_symbol,
                                        "side": side,
                                        "entry_price": price_open,
                                        "current_price": None,
                                        "sl": sl,
                                        "tp": tp,
                                        "pnl": None,
                                        "exit_decision": action,
                                        "exit_reason": reason,
                                        "close_execution_enabled": execution_enabled,
                                        "close_attempted": False,
                                        "close_result": "skipped",
                                        "close_error": "",
                                        "dry_run": runtime.app_settings.dry_run,
                                    }
                                )
                                continue

                            contract = runtime.symbol_registry.symbols.get(pos_symbol) or runtime.symbol_registry.symbols.get(symbol)
                            point_value = float(getattr(contract, "point_value", 0.01) or 0.01)
                            profit_points = (
                                (current_price - price_open) / point_value
                                if side == "BUY"
                                else (price_open - current_price) / point_value
                            )

                            try:
                                market_frame = market_data.load_symbol_frame(pos_symbol)
                                snapshot = feature_engine.build_snapshot(
                                    symbol=pos_symbol,
                                    timeframe=runtime.app_settings.timeframe,
                                    frame=market_frame.data,
                                    spread=spread_points,
                                    open_position_flag=True,
                                )
                                dashboard_writer.update_market(
                                    _build_market_payload(
                                        symbol=pos_symbol,
                                        timeframe=runtime.app_settings.timeframe,
                                        market_frame=market_frame,
                                        tick=tick,
                                        spread=spread_points,
                                    ),
                                    status="position_monitor",
                                )
                                dashboard_writer.update_indicators(
                                    _build_indicator_payload(snapshot),
                                    status="position_monitor",
                                )
                            except Exception:
                                if pos_symbol not in reported_market_failures:
                                    reported_market_failures.add(pos_symbol)
                                action = "HOLD"
                                reason = "snapshot_unavailable"
                                print(
                                    f"[MONITOR] {pos_symbol} side={side} action={action} ticket={ticket} "
                                    f"reason={reason}"
                                )
                                append_monitor_log(
                                    {
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "ticket": ticket,
                                        "symbol": pos_symbol,
                                        "side": side,
                                        "entry_price": price_open,
                                        "current_price": current_price,
                                        "sl": sl,
                                        "tp": tp,
                                        "pnl": None,
                                        "exit_decision": action,
                                        "exit_reason": reason,
                                        "close_execution_enabled": execution_enabled,
                                        "close_attempted": False,
                                        "close_result": "skipped",
                                        "close_error": "",
                                        "dry_run": runtime.app_settings.dry_run,
                                    }
                                )
                                continue

                            action, reason = eval_exit_action(snapshot, side, profit_points)
                            print(
                                f"[MONITOR] {pos_symbol} side={side} action={action} ticket={ticket} "
                                f"reason={reason}"
                            )

                            execution_attempted = False
                            execution_success = False
                            exec_reason = ""

                            if action in ("CLOSE_EARLY", "FORCE_EXIT"):
                                close_signals += 1
                                if execution_enabled and ticket not in closed_tickets:
                                    execution_attempted = True
                                    ok, exec_reason = close_position_by_ticket(
                                        ticket=ticket,
                                        symbol=pos_symbol,
                                        side=side,
                                        volume=volume,
                                    )
                                    status = "SUCCESS" if ok else "FAIL"
                                    print(
                                        f"[MONITOR-EXEC] {pos_symbol} ticket={ticket} action={action} "
                                        f"status={status} reason={exec_reason}"
                                    )
                                    if ok:
                                        closed_tickets.add(ticket)
                                        executed += 1
                                        execution_success = True

                            append_monitor_log(
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "ticket": ticket,
                                    "symbol": pos_symbol,
                                    "side": side,
                                    "entry_price": price_open,
                                    "current_price": current_price,
                                    "sl": sl,
                                    "tp": tp,
                                    "pnl": profit_points,
                                    "exit_decision": action,
                                    "exit_reason": reason,
                                    "close_execution_enabled": execution_enabled,
                                    "close_attempted": execution_attempted,
                                    "close_result": (
                                        "success"
                                        if execution_success
                                        else ("failed" if execution_attempted else "skipped")
                                    ),
                                    "close_error": (
                                        ""
                                        if execution_success or not execution_attempted
                                        else exec_reason
                                    ),
                                    "dry_run": runtime.app_settings.dry_run,
                                }
                            )

                    dashboard_writer.update_report(
                        {
                            "summary": {
                                "mode": "position_monitor",
                                "cycle": cycle,
                                "positions": positions_count,
                                "close_signals": close_signals,
                                "executed": executed,
                            }
                        },
                        status="position_monitor",
                    )

                    print(
                        f"[MONITOR-CYCLE] {cycle} positions={positions_count} "
                        f"close_signals={close_signals} executed={executed}"
                    )

                    if max_cycles is not None and cycle >= max_cycles:
                        break

                    time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("Position monitor interrupted. Shutting down.")
                return 0
            finally:
                gateway.shutdown()

            return 0

        elif os.getenv("RUN_PRODUCTION_RUNTIME") == "1" or os.getenv("RUN_DEMO_RUNTIME") == "1":
            if os.getenv("RUN_PRODUCTION_RUNTIME") == "1":
                print("RUN_PRODUCTION_RUNTIME=1 - production runtime mode enabled")
            else:
                print("RUN_DEMO_RUNTIME=1 - production runtime mode enabled (legacy alias)")

            def load_production_raw_candidates(
                symbols_to_scan: list[str],
                reported_failures: set[str],
                market_data: Any,
                feature_engine: Any,
                candidate_engine: Any,
                gateway: Any,
            ) -> list[dict[str, Any]]:
                timeframe = runtime.app_settings.timeframe
                raw_candidates: list[dict[str, Any]] = []
                for symbol in symbols_to_scan:
                    try:
                        market_frame = market_data.load_symbol_frame(symbol)
                        tick = gateway.get_tick(symbol)
                        snapshot = feature_engine.build_snapshot(
                            symbol=symbol,
                            timeframe=timeframe,
                            frame=market_frame.data,
                            spread=float(getattr(tick, "spread_points", 0.0) or 0.0),
                        )
                        dashboard_writer.update_market(
                            _build_market_payload(
                                symbol=symbol,
                                timeframe=timeframe,
                                market_frame=market_frame,
                                tick=tick,
                                spread=_safe_float(getattr(tick, "spread_points", None)),
                            ),
                            status="production_runtime",
                        )
                        dashboard_writer.update_indicators(
                            _build_indicator_payload(snapshot),
                            status="production_runtime",
                        )

                        candidate = candidate_engine.detect_candidate(snapshot)
                        if candidate is None:
                            dashboard_writer.update_signal(
                                {
                                    "status": "NO_CANDIDATE",
                                    "side": None,
                                    "score": None,
                                    "entry": None,
                                    "sl": None,
                                    "tp": None,
                                    "reason": "no_candidate_detected",
                                    "request_id": None,
                                },
                                status="production_runtime",
                            )
                            continue

                        raw_candidate = {
                            "symbol": candidate.symbol,
                            "timeframe": candidate.timeframe,
                            "bar_time": candidate.bar_time,
                            "decision": candidate.direction,
                            "score": candidate.score,
                            "entry": candidate.entry_hint,
                            "sl": candidate.stop_hint,
                            "tp": candidate.target_hint,
                            "guard": "allowed",
                        }
                        dashboard_writer.update_signal(
                            _build_signal_payload(
                                raw_candidate,
                                status="CANDIDATE",
                                reason="candidate_detected",
                            ),
                            status="production_runtime",
                        )
                        dashboard_writer.update_guard(
                            _build_guard_payload(
                                status="allowed",
                                reason="candidate_scan_allowed",
                            ),
                            status="production_runtime",
                        )
                        raw_candidates.append(raw_candidate)
                    except Exception as exc:
                        if symbol not in reported_failures:
                            reported_failures.add(symbol)
                            print(f"Production runtime skip symbol: {symbol} ({exc})")
                        dashboard_writer.append_log(
                            {
                                "event": "production_runtime_symbol_error",
                                "symbol": symbol,
                                "reason": str(exc),
                            },
                            status="production_runtime",
                        )
                        continue
                return raw_candidates

            from core.candidate_engine import CandidateEngine
            from core.candidate_scan_finalize import finalize_candidate_scan
            from core.feature_engine import FeatureEngine
            from core.groq_client import GroqClient
            from core.groq_prompt_builder import GroqPromptBuilder
            from core.groq_response_parser import GroqResponseParser
            from core.market_data import MarketDataService
            from core.mt5_gateway import MT5Gateway
            import time
            from datetime import datetime

            interval_raw = os.getenv("PRODUCTION_RUNTIME_INTERVAL_SECONDS", "").strip()
            if not interval_raw:
                interval_raw = os.getenv("DEMO_RUNTIME_INTERVAL_SECONDS", "").strip()
            interval_seconds = 10
            if interval_raw:
                try:
                    interval_seconds = max(int(interval_raw), 1)
                except ValueError:
                    interval_seconds = 10

            max_cycles_raw = os.getenv("PRODUCTION_RUNTIME_MAX_CYCLES", "").strip()
            if not max_cycles_raw:
                max_cycles_raw = os.getenv("DEMO_RUNTIME_MAX_CYCLES", "").strip()
            max_cycles: int | None = None
            if max_cycles_raw:
                try:
                    max_cycles = max(int(max_cycles_raw), 1)
                except ValueError:
                    max_cycles = None

            groq_client = GroqClient()
            groq_configured = groq_client.is_configured()
            if not groq_configured:
                print("Production runtime warning: GROQ_API_KEY not configured")
            parser = GroqResponseParser()

            max_processed_setups_raw = os.getenv(
                "PRODUCTION_RUNTIME_MAX_PROCESSED_SETUPS", ""
            ).strip()
            if not max_processed_setups_raw:
                max_processed_setups_raw = os.getenv(
                    "DEMO_RUNTIME_MAX_PROCESSED_SETUPS", ""
                ).strip()
            max_processed_setups = 5000
            if max_processed_setups_raw:
                try:
                    max_processed_setups = max(int(max_processed_setups_raw), 1)
                except ValueError:
                    max_processed_setups = 5000

            processed_setup_state_path = PROJECT_ROOT / "storage" / "processed_setups.json"
            processed_setup_order, processed_setups = _load_processed_setup_state(
                processed_setup_state_path
            )
            print(
                f"[DEDUP-STATE] loaded={len(processed_setup_order)} "
                f"path={processed_setup_state_path}"
            )

            ai_log_path = PROJECT_ROOT / "storage" / "ai_decisions.jsonl"
            ai_log_path.parent.mkdir(parents=True, exist_ok=True)

            reported_failures: set[str] = set()
            available_symbols: list[str] = []

            gateway = MT5Gateway()
            gateway.initialize()
            gateway.ensure_connection()
            try:
                market_data = MarketDataService(
                    gateway=gateway,
                    timeframe=runtime.app_settings.timeframe,
                    max_bars_fetch=runtime.app_settings.max_bars_fetch,
                )
                feature_engine = FeatureEngine()
                candidate_engine = CandidateEngine()

                for symbol in runtime.app_settings.symbols:
                    try:
                        market_data.load_symbol_frame(symbol)
                        available_symbols.append(symbol)
                    except Exception:
                        reported_failures.add(symbol)
                        print(f"Production runtime skip symbol: {symbol}")

                if not available_symbols:
                    print("Production runtime fatal: no available MT5 symbols. Shutting down.")
                    return 0

                print(f"[ACTIVE-SYMBOLS] {','.join(available_symbols)}")
                dashboard_writer.update_runtime(status="production_runtime", active_symbol=available_symbols[0])

                cycle = 0
                try:
                    while True:
                        cycle += 1
                        raw_candidates = load_production_raw_candidates(
                            available_symbols,
                            reported_failures,
                            market_data,
                            feature_engine,
                            candidate_engine,
                            gateway,
                        )

                        finalized = finalize_candidate_scan(
                            raw_candidates=raw_candidates,
                            timeframe=runtime.app_settings.timeframe,
                            processed_symbols=len(available_symbols),
                        )
                        accepted = finalized["accepted"]
                        rejected = finalized["rejected"]

                        runtime_accepted: list[dict[str, Any]] = []
                        runtime_duplicate_count = 0

                        for candidate in accepted:
                            symbol = (
                                candidate.get("display_symbol")
                                or candidate.get("runtime_symbol")
                                or candidate.get("canonical_symbol")
                                or "unknown_symbol"
                            )
                            timeframe = candidate.get("timeframe") or runtime.app_settings.timeframe
                            setup_id = _extract_runtime_setup_id(
                                candidate=candidate,
                                default_symbol=symbol,
                                default_timeframe=timeframe,
                            )
                            if setup_id in processed_setups:
                                runtime_duplicate_count += 1
                                print(f"[SKIP-DUPLICATE] {symbol} {setup_id}")
                                dashboard_writer.update_guard(
                                    _build_guard_payload(
                                        status="blocked",
                                        reason="duplicate_setup_blocked",
                                        metadata=candidate.get("metadata", {}) or {},
                                    ),
                                    status="production_runtime",
                                )
                                dashboard_writer.append_log(
                                    {
                                        "event": "duplicate_setup_blocked",
                                        "symbol": symbol,
                                        "setup_id": setup_id,
                                    },
                                    status="production_runtime",
                                )
                                continue
                            runtime_accepted.append(candidate)

                        for candidate in runtime_accepted:
                            print(_format_runtime_accept_line(candidate))
                            dashboard_writer.update_signal(
                                _build_signal_payload(
                                    candidate,
                                    status="APPROVED",
                                    reason="finalize_candidate_scan_accepted",
                                ),
                                status="production_runtime",
                            )
                            dashboard_writer.update_guard(
                                _build_guard_payload(
                                    status=str(candidate.get("guard") or "allowed"),
                                    reason="finalize_candidate_scan_accepted",
                                    metadata=candidate.get("metadata", {}) or {},
                                ),
                                status="production_runtime",
                            )

                        for candidate in rejected:
                            print(_format_runtime_reject_line(candidate))
                            dashboard_writer.update_signal(
                                _build_signal_payload(
                                    candidate,
                                    status=str(candidate.get("status") or "REJECTED"),
                                    reason=str(candidate.get("reason") or "candidate_rejected"),
                                ),
                                status="production_runtime",
                            )
                            dashboard_writer.update_guard(
                                _build_guard_payload(
                                    status="blocked",
                                    reason=str(candidate.get("reason") or "candidate_rejected"),
                                    metadata=candidate.get("metadata", {}) or {},
                                ),
                                status="production_runtime",
                            )

                        summary_line = _format_runtime_summary_line(
                            processed_symbols=len(available_symbols),
                            input_candidates=len(raw_candidates),
                            approved_candidates=len(runtime_accepted),
                            rejected_candidates=len(rejected) + runtime_duplicate_count,
                            duplicates_blocked=int(
                                (finalized.get("summary", {}) or {}).get("duplicates_blocked", 0)
                            )
                            + runtime_duplicate_count,
                            unique_underlying_setups=len(runtime_accepted),
                            timeframe=runtime.app_settings.timeframe,
                        )
                        print(summary_line)
                        dashboard_writer.update_report(
                            {
                                "summary": {
                                    "cycle": cycle,
                                    "processed_symbols": len(available_symbols),
                                    "input_candidates": len(raw_candidates),
                                    "approved_candidates": len(runtime_accepted),
                                    "rejected_candidates": len(rejected) + runtime_duplicate_count,
                                    "duplicates_blocked": int(
                                        (finalized.get("summary", {}) or {}).get("duplicates_blocked", 0)
                                    )
                                    + runtime_duplicate_count,
                                    "unique_underlying_setups": len(runtime_accepted),
                                    "timeframe": runtime.app_settings.timeframe,
                                    "summary_line": summary_line,
                                }
                            },
                            status="production_runtime",
                        )

                        ai_processed = 0
                        for candidate in runtime_accepted:
                            symbol = (
                                candidate.get("display_symbol")
                                or candidate.get("runtime_symbol")
                                or candidate.get("canonical_symbol")
                                or "unknown_symbol"
                            )
                            direction = candidate.get("decision") or candidate.get("direction") or ""
                            timeframe = candidate.get("timeframe") or runtime.app_settings.timeframe
                            entry = float(candidate.get("entry", 0.0) or 0.0)
                            sl = float(candidate.get("sl", 0.0) or 0.0)
                            tp = float(candidate.get("tp", 0.0) or 0.0)
                            score = float(candidate.get("score", 0.0) or 0.0)
                            bar_time = candidate.get("bar_time", "")

                            metadata = candidate.get("metadata", {}) or {}
                            setup_id = _extract_runtime_setup_id(
                                candidate=candidate,
                                default_symbol=symbol,
                                default_timeframe=timeframe,
                            )
                            candidate_id = (
                                metadata.get("candidate_id")
                                or candidate.get("setup_key")
                                or "unknown_candidate"
                            )

                            _remember_processed_setup(
                                setup_id=setup_id,
                                ordered_setup_ids=processed_setup_order,
                                processed_setup_ids=processed_setups,
                                file_path=processed_setup_state_path,
                                max_items=max_processed_setups,
                            )

                            prompt_data = GroqPromptBuilder.build_decision_prompt(
                                symbol=symbol,
                                timeframe=timeframe,
                                direction=direction,
                                score=score,
                                entry_hint=entry,
                                stop_hint=sl,
                                target_hint=tp,
                                reasons=metadata.get("reasons", []),
                                features=metadata.get("features", {}),
                                bar_time=bar_time,
                            )

                            if groq_configured:
                                try:
                                    groq_response = groq_client.chat_completion(
                                        system_prompt=prompt_data["system"],
                                        user_prompt=prompt_data["user"],
                                    )
                                except Exception as exc:
                                    groq_response = {
                                        "success": False,
                                        "content": "",
                                        "model_name": "unknown",
                                        "latency_ms": 0,
                                        "error": str(exc),
                                        "finish_reason": "error",
                                    }
                            else:
                                groq_response = {
                                    "success": False,
                                    "content": "",
                                    "model_name": "skipped",
                                    "latency_ms": 0,
                                    "error": "API key not configured",
                                    "finish_reason": "skipped",
                                }

                            parser_candidate_data: dict[str, Any] = {
                                "candidate_id": candidate_id,
                                "direction": direction,
                                "entry_hint": entry,
                                "stop_hint": sl,
                                "target_hint": tp,
                                "setup_quality": float(metadata.get("setup_quality", 0.0) or 0.0),
                                "trend_alignment": float(metadata.get("trend_alignment", 0.0) or 0.0),
                                "regime_fit": float(metadata.get("regime_fit", 0.0) or 0.0),
                                "exhaustion_risk": float(metadata.get("exhaustion_risk", 0.0) or 0.0),
                            }
                            ai_decision = parser.parse(
                                groq_response=groq_response,
                                candidate_data=parser_candidate_data,
                                prompt_version=prompt_data["prompt_version"],
                            )

                            print(
                                f"[AI] {symbol} {ai_decision.decision} "
                                f"approved={ai_decision.approved} conf={ai_decision.confidence:.2f} "
                                f"valid={ai_decision.valid_response} latency={ai_decision.latency_ms} "
                                f"reason={ai_decision.reason}"
                            )

                            if ai_decision.confidence == 0.0 or ai_decision.valid_response is False:
                                raw_content = str(groq_response.get("content", "") or "").strip()
                                raw_preview = raw_content.replace("\n", " ")[:800]
                                print(
                                    f"[AI-DEBUG] {symbol} model={ai_decision.model_name} "
                                    f"reason={ai_decision.reason}"
                                )
                                print(
                                    f"[AI-RAW] success={groq_response.get('success', False)} "
                                    f"finish_reason={groq_response.get('finish_reason', '')} "
                                    f"latency_ms={groq_response.get('latency_ms', 0)} "
                                    f"content={raw_preview}"
                                )

                            logged_decision = str(ai_decision.decision)
                            if "." in logged_decision:
                                logged_decision = logged_decision.split(".")[-1]

                            decision_payload = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "bar_time": bar_time,
                                "setup_id": setup_id,
                                "candidate_id": ai_decision.candidate_id,
                                "score": score,
                                "entry": entry,
                                "sl": sl,
                                "tp": tp,
                                "decision": logged_decision,
                                "approved": ai_decision.approved,
                                "confidence": ai_decision.confidence,
                                "valid_response": ai_decision.valid_response,
                                "reason": ai_decision.reason,
                                "model_name": ai_decision.model_name,
                                "prompt_version": prompt_data["prompt_version"],
                                "raw_success": bool(groq_response.get("success", False)),
                                "raw_error": str(groq_response.get("error", "") or ""),
                                "finish_reason": groq_response.get("finish_reason", ""),
                                "response_preview": str(groq_response.get("content", "") or "").replace("\n", " ")[:300],
                                "response_length": len(str(groq_response.get("content", "") or "")),
                                "dry_run": runtime.app_settings.dry_run,
                                "latency_ms": ai_decision.latency_ms,
                            }

                            with ai_log_path.open("a", encoding="utf-8") as file:
                                file.write(json.dumps(decision_payload, ensure_ascii=False) + "\n")

                            dashboard_writer.append_log(
                                {"event": "ai_decision", **decision_payload},
                                status="production_runtime",
                            )
                            dashboard_writer.update_guard(
                                _build_guard_payload(
                                    status="allowed" if ai_decision.approved else "blocked",
                                    reason=str(ai_decision.reason or "ai_decision"),
                                ),
                                status="production_runtime",
                            )
                            dashboard_writer.update_report(
                                {
                                    "last_decision_at": decision_payload["timestamp"],
                                    "summary": {
                                        "cycle": cycle,
                                        "symbol": symbol,
                                        "setup_id": setup_id,
                                        "ai_decision": logged_decision,
                                        "approved": ai_decision.approved,
                                        "confidence": ai_decision.confidence,
                                        "reason": ai_decision.reason,
                                    },
                                },
                                status="production_runtime",
                            )
                            ai_processed += 1

                        print(
                            f"[CYCLE] {cycle} raw={len(raw_candidates)} "
                            f"accepted={len(runtime_accepted)} "
                            f"rejected={len(rejected) + runtime_duplicate_count} "
                            f"ai={ai_processed}"
                        )

                        if max_cycles is not None and cycle >= max_cycles:
                            break

                        time.sleep(interval_seconds)
                except KeyboardInterrupt:
                    print("Production runtime interrupted. Shutting down.")
                    return 0
            finally:
                gateway.shutdown()

        elif os.getenv("RUN_CANDIDATE_TO_GROQ_SMOKE") == "1":
            print("RUN_CANDIDATE_TO_GROQ_SMOKE=1 - candidate-to-groq smoke test enabled")
            from app.smoke_test_candidate_to_groq import run_integration_test

            run_integration_test()
        print("Foundation layer initialized successfully.")
        return 0
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
