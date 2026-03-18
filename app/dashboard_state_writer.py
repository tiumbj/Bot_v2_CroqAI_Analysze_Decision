# ============================================================
# ชื่อโค้ด: Dashboard State Writer
# ที่อยู่ไฟล์: app/dashboard_state_writer.py
# คำสั่งรัน: ถูกเรียกจาก python app/main.py
# เวอร์ชัน: v1.2.1
# ============================================================

"""
dashboard_state_writer.py
Version: v1.2.1

Purpose:
    Production dashboard state writer for runtime/dashboard_state.json

CHANGELOG (v1.2.1)
------------------------------------------------------------
- KEEP: atomic JSON write via temp file + os.replace
- KEEP: no demo fallback / no sample loop / no fake data
- KEEP: strict section field filtering for stable schema reuse
- ADD: retry on Windows file-lock / transient replace failures
- ADD: no-crash writer behavior so dashboard output layer never kills runtime
- KEEP: monitoring/output layer only; no trading logic
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

WRITER_VERSION = "v1.2.1"
SCHEMA_VERSION = "dashboard_state.schema.v1"

WRITE_RETRY_ATTEMPTS = 8
WRITE_RETRY_DELAY_SECONDS = 0.15

SECTION_FIELDS: dict[str, set[str]] = {
    "runtime": {
        "app_name",
        "environment",
        "mode",
        "status",
        "timeframe",
        "symbols",
        "active_symbol",
        "dry_run",
    },
    "market": {
        "symbol",
        "timeframe",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "bid",
        "ask",
        "spread",
    },
    "signal": {
        "status",
        "side",
        "score",
        "entry",
        "sl",
        "tp",
        "reason",
        "request_id",
    },
    "guard": {
        "status",
        "reason",
        "spread_ok",
        "rr_ok",
        "cooldown_ok",
    },
    "report": {
        "summary",
        "last_event",
        "last_decision_at",
    },
    "position": {
        "ticket",
        "symbol",
        "side",
        "entry_price",
        "current_price",
        "sl",
        "tp",
        "pnl",
        "exit_decision",
        "exit_reason",
        "close_execution_enabled",
        "close_attempted",
        "close_result",
        "close_error",
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    try:
        return float(value)
    except Exception:
        return str(value)


def _clean_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {str(k): _json_safe(v) for k, v in payload.items()}


class DashboardStateWriter:
    def __init__(
        self,
        state_path: str | Path = "runtime/dashboard_state.json",
        max_logs: int = 200,
    ) -> None:
        self.state_path = Path(state_path)
        self.max_logs = max_logs
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "writer_version": WRITER_VERSION,
                "updated_at": None,
            },
            "runtime": {
                "app_name": None,
                "environment": None,
                "mode": "production",
                "status": "startup",
                "timeframe": None,
                "symbols": [],
                "active_symbol": None,
                "dry_run": None,
            },
            "market": {
                "symbol": None,
                "timeframe": None,
                "bar_time": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "bid": None,
                "ask": None,
                "spread": None,
            },
            "indicators": {},
            "signal": {
                "status": None,
                "side": None,
                "score": None,
                "entry": None,
                "sl": None,
                "tp": None,
                "reason": None,
                "request_id": None,
            },
            "guard": {
                "status": None,
                "reason": None,
                "spread_ok": None,
                "rr_ok": None,
                "cooldown_ok": None,
            },
            "report": {
                "summary": None,
                "last_event": None,
                "last_decision_at": None,
            },
            "position": {
                "ticket": None,
                "symbol": None,
                "side": None,
                "entry_price": None,
                "current_price": None,
                "sl": None,
                "tp": None,
                "pnl": None,
                "exit_decision": None,
                "exit_reason": None,
                "close_execution_enabled": None,
                "close_attempted": None,
                "close_result": None,
                "close_error": None,
            },
            "logs": {
                "last_message": None,
                "recent": [],
            },
        }

    def _filter_section_payload(
        self,
        section_name: str,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        clean = _clean_mapping(payload)
        allowed = SECTION_FIELDS.get(section_name)
        if not allowed:
            return clean
        return {key: value for key, value in clean.items() if key in allowed}

    def _merge_section(self, section_name: str, payload: Mapping[str, Any] | None) -> None:
        clean = self._filter_section_payload(section_name, payload)
        if not clean:
            return

        section = self._state.get(section_name)
        if not isinstance(section, MutableMapping):
            raise TypeError(f"State section '{section_name}' is not mutable")

        section.update(clean)

    def _set_runtime_status(self, status: str | None) -> None:
        if status:
            self._state["runtime"]["status"] = status

    def _ensure_parent_dir(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self) -> None:
        self._ensure_parent_dir()
        payload = deepcopy(self._state)
        payload["meta"]["updated_at"] = _utc_now_iso()
        tmp_name: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.state_path.parent),
                prefix=".dashboard_state.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_name = tmp_file.name

            last_error: Exception | None = None
            for attempt in range(1, WRITE_RETRY_ATTEMPTS + 1):
                try:
                    os.replace(tmp_name, self.state_path)
                    tmp_name = None
                    return
                except (PermissionError, OSError) as exc:
                    last_error = exc
                    if attempt >= WRITE_RETRY_ATTEMPTS:
                        break
                    time.sleep(WRITE_RETRY_DELAY_SECONDS)

            if last_error is not None:
                raise last_error
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _write_locked(self) -> None:
        try:
            self._atomic_write()
        except Exception as exc:
            print(f"[DASHBOARD-WRITER] write skipped: {exc}")

    def bootstrap(
        self,
        *,
        app_name: str,
        environment: str,
        timeframe: str,
        symbols: list[str],
        dry_run: bool,
        status: str = "running",
    ) -> None:
        with self._lock:
            self._merge_section(
                "runtime",
                {
                    "app_name": app_name,
                    "environment": environment,
                    "mode": "production",
                    "status": status,
                    "timeframe": timeframe,
                    "symbols": symbols,
                    "dry_run": dry_run,
                },
            )
            self._write_locked()

    def update_runtime(
        self,
        *,
        active_symbol: str | None = None,
        status: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            runtime_payload: dict[str, Any] = {}
            if active_symbol is not None:
                runtime_payload["active_symbol"] = active_symbol
            if extra:
                runtime_payload.update(_clean_mapping(extra))
            self._merge_section("runtime", runtime_payload)
            self._set_runtime_status(status)
            self._write_locked()

    def update_market(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = "market_updated",
    ) -> None:
        with self._lock:
            self._merge_section("market", payload)
            symbol = payload.get("symbol")
            if symbol is not None:
                self._merge_section("runtime", {"active_symbol": symbol})
            self._set_runtime_status(status)
            self._write_locked()

    def update_indicators(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = "indicators_updated",
    ) -> None:
        with self._lock:
            clean = _clean_mapping(payload)
            self._state["indicators"].update(clean)
            self._set_runtime_status(status)
            self._write_locked()

    def update_signal(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = "signal_updated",
    ) -> None:
        with self._lock:
            self._merge_section("signal", payload)
            self._state["report"]["last_decision_at"] = _utc_now_iso()
            self._set_runtime_status(status)
            self._write_locked()

    def update_guard(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = "guard_updated",
    ) -> None:
        with self._lock:
            self._merge_section("guard", payload)
            self._set_runtime_status(status)
            self._write_locked()

    def update_report(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = "report_updated",
    ) -> None:
        with self._lock:
            self._merge_section("report", payload)
            self._set_runtime_status(status)
            self._write_locked()

    def update_position(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = "position_updated",
    ) -> None:
        with self._lock:
            self._merge_section("position", payload)
            symbol = payload.get("symbol")
            if symbol is not None:
                self._merge_section("runtime", {"active_symbol": symbol})
            self._set_runtime_status(status)
            self._write_locked()

    def append_log(
        self,
        payload: Mapping[str, Any],
        *,
        status: str | None = None,
    ) -> None:
        with self._lock:
            clean = _clean_mapping(payload)
            recent_logs = self._state["logs"]["recent"]
            recent_logs.append(clean)

            if len(recent_logs) > self.max_logs:
                del recent_logs[:-self.max_logs]

            self._state["logs"]["last_message"] = clean
            self._state["report"]["last_event"] = clean

            if status:
                self._set_runtime_status(status)

            self._write_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)
