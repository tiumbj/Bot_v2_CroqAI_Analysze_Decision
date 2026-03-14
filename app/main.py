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


def main() -> int:
    try:
        runtime = build_runtime_context()
        initialize_storage(runtime)
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
                        except Exception as exc:
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
                            except Exception as exc:
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
                                        "symbol": pos_symbol,
                                        "ticket": ticket,
                                        "side": side,
                                        "action": action,
                                        "reason": reason,
                                        "price_open": price_open,
                                        "current_price": None,
                                        "sl": sl,
                                        "tp": tp,
                                        "volume": volume,
                                        "execution_enabled": execution_enabled,
                                        "execution_attempted": False,
                                        "execution_success": False,
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
                            except Exception as exc:
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
                                        "symbol": pos_symbol,
                                        "ticket": ticket,
                                        "side": side,
                                        "action": action,
                                        "reason": reason,
                                        "price_open": price_open,
                                        "current_price": current_price,
                                        "sl": sl,
                                        "tp": tp,
                                        "volume": volume,
                                        "execution_enabled": execution_enabled,
                                        "execution_attempted": False,
                                        "execution_success": False,
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
                                    "symbol": pos_symbol,
                                    "ticket": ticket,
                                    "side": side,
                                    "action": action,
                                    "reason": reason,
                                    "price_open": price_open,
                                    "current_price": current_price,
                                    "sl": sl,
                                    "tp": tp,
                                    "volume": volume,
                                    "execution_enabled": execution_enabled,
                                    "execution_attempted": execution_attempted,
                                    "execution_success": execution_success,
                                }
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
            ) -> list[dict[str, Any]]:
                timeframe = runtime.app_settings.timeframe
                raw_candidates: list[dict[str, Any]] = []
                for symbol in symbols_to_scan:
                    try:
                        market_frame = market_data.load_symbol_frame(symbol)
                        snapshot = feature_engine.build_snapshot(
                            symbol=symbol,
                            timeframe=timeframe,
                            frame=market_frame.data,
                        )
                        candidate = candidate_engine.detect_candidate(snapshot)
                        if candidate is None:
                            continue
                        raw_candidates.append(
                            {
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
                        )
                    except Exception as exc:
                        if symbol not in reported_failures:
                            reported_failures.add(symbol)
                            print(f"Production runtime skip symbol: {symbol} ({exc})")
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
            processed_setups: set[str] = set()
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
                        )

                        finalized = finalize_candidate_scan(
                            raw_candidates=raw_candidates,
                            timeframe=runtime.app_settings.timeframe,
                            processed_symbols=len(available_symbols),
                        )
                        accepted = finalized["accepted"]
                        rejected = finalized["rejected"]
                        log_lines = finalized["log_lines"]
                        for line in log_lines:
                            print(line)

                        ai_processed = 0
                        for candidate in accepted:
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
                            setup_id = (
                                candidate.get("setup_key")
                                or metadata.get("candidate_id")
                                or f"{symbol}|{timeframe}|{bar_time}|{direction}"
                            )
                            candidate_id = (
                                metadata.get("candidate_id")
                                or candidate.get("setup_key")
                                or "unknown_candidate"
                            )
                            if setup_id in processed_setups:
                                print(f"[SKIP-DUPLICATE] {symbol} {setup_id}")
                                continue
                            processed_setups.add(setup_id)

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
                                print(
                                    f"[AI-DEBUG] {symbol} model={ai_decision.model_name} "
                                    f"reason={ai_decision.reason}"
                                )

                            with ai_log_path.open("a", encoding="utf-8") as file:
                                file.write(
                                    json.dumps(
                                        {
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "symbol": symbol,
                                            "candidate_id": ai_decision.candidate_id,
                                            "decision": str(ai_decision.decision),
                                            "approved": ai_decision.approved,
                                            "confidence": ai_decision.confidence,
                                            "valid_response": ai_decision.valid_response,
                                            "reason": ai_decision.reason,
                                            "model_name": ai_decision.model_name,
                                            "latency_ms": ai_decision.latency_ms,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            ai_processed += 1

                        print(
                            f"[CYCLE] {cycle} raw={len(raw_candidates)} "
                            f"accepted={len(accepted)} rejected={len(rejected)} ai={ai_processed}"
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
