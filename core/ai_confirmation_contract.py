from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


@dataclass(frozen=True)
class ConfirmationResult:
    approved: bool
    reason: str
    risk_flags: List[str]
    context: Dict[str, Any]


class AIConfirmationContract:
    def __init__(self) -> None:
        self.calendar_url = os.getenv("ECON_CALENDAR_API_URL", "").strip()
        self.calendar_timeout_seconds = int(os.getenv("ECON_CALENDAR_TIMEOUT_SECONDS", "4") or 4)
        self.lookahead_minutes = int(os.getenv("ECON_CALENDAR_LOOKAHEAD_MINUTES", "15") or 15)
        self.correlation_window = int(os.getenv("CORRELATION_WINDOW_BARS", "24") or 24)
        self.session = requests.Session()

    def evaluate(self, candidate_data: Dict[str, Any], decision: str) -> ConfirmationResult:
        risk_flags: List[str] = []
        context: Dict[str, Any] = {}

        symbol = str(candidate_data.get("symbol", "") or "")
        features = candidate_data.get("features", {}) or {}
        direction = str(decision or candidate_data.get("direction", "")).upper().strip()

        event_risk, event_context = self._economic_calendar_risk(symbol=symbol)
        if event_risk:
            risk_flags.append("high_impact_news_within_lookahead")
        context["calendar"] = event_context

        cross_risk, cross_context = self._cross_asset_risk(direction=direction)
        if cross_risk:
            risk_flags.append("cross_asset_conflict")
        context["cross_asset"] = cross_context

        pa_risk, pa_context = self._price_action_risk(
            direction=direction,
            candidate_data=candidate_data,
            features=features,
        )
        if pa_risk:
            risk_flags.append("price_action_risk")
        context["price_action"] = pa_context

        if risk_flags:
            reason = "Risk gates blocked: " + ", ".join(risk_flags)
            return ConfirmationResult(
                approved=False,
                reason=reason,
                risk_flags=risk_flags,
                context=context,
            )

        return ConfirmationResult(
            approved=True,
            reason="Deep confirmation passed",
            risk_flags=[],
            context=context,
        )

    def _economic_calendar_risk(self, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        now_utc = datetime.now(timezone.utc)
        upper_bound = now_utc + timedelta(minutes=self.lookahead_minutes)

        if not self.calendar_url:
            return False, {"status": "calendar_api_not_configured"}

        try:
            response = self.session.get(self.calendar_url, timeout=self.calendar_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return False, {"status": "calendar_api_error", "error": str(exc)}

        if not isinstance(payload, list):
            return False, {"status": "calendar_payload_invalid"}

        relevant_currency = {"USD"}
        high_impact_events: List[Dict[str, Any]] = []

        for row in payload:
            if not isinstance(row, dict):
                continue
            impact = str(row.get("impact", "") or "").upper()
            currency = str(row.get("currency", "") or "").upper()
            event_time_raw = row.get("time") or row.get("datetime") or ""
            if impact not in {"HIGH", "RED", "3"}:
                continue
            if currency not in relevant_currency:
                continue
            parsed = self._parse_event_time(event_time_raw)
            if parsed is None:
                continue
            if now_utc <= parsed <= upper_bound:
                high_impact_events.append(
                    {
                        "currency": currency,
                        "impact": impact,
                        "title": str(row.get("title", "") or row.get("event", "") or ""),
                        "time": parsed.isoformat(),
                    }
                )

        if high_impact_events:
            return True, {"status": "news_risk", "events": high_impact_events[:3]}
        return False, {"status": "clear"}

    def _cross_asset_risk(self, direction: str) -> Tuple[bool, Dict[str, Any]]:
        symbols = {
            "gold": "GC=F",
            "dxy": "DX-Y.NYB",
            "us10y": "^TNX",
        }
        frames: Dict[str, pd.Series] = {}
        context: Dict[str, Any] = {"status": "clear"}

        for key, ticker in symbols.items():
            series = self._fetch_yahoo_close_series(ticker=ticker, interval="5m", period="1d")
            if series is None or len(series) < self.correlation_window:
                context["status"] = "insufficient_market_data"
                return False, context
            frames[key] = series.tail(self.correlation_window)

        data = pd.DataFrame(frames).dropna()
        if len(data) < max(10, self.correlation_window // 2):
            context["status"] = "insufficient_overlap"
            return False, context

        returns = data.pct_change().dropna()
        if returns.empty:
            context["status"] = "returns_empty"
            return False, context

        corr_gold_dxy = float(returns["gold"].corr(returns["dxy"]))
        corr_gold_us10y = float(returns["gold"].corr(returns["us10y"]))
        dxy_momentum = float((data["dxy"].iloc[-1] / data["dxy"].iloc[0]) - 1.0)
        us10y_momentum = float((data["us10y"].iloc[-1] / data["us10y"].iloc[0]) - 1.0)
        context = {
            "status": "ok",
            "corr_gold_dxy": round(corr_gold_dxy, 4),
            "corr_gold_us10y": round(corr_gold_us10y, 4),
            "dxy_momentum": round(dxy_momentum, 5),
            "us10y_momentum": round(us10y_momentum, 5),
        }

        if direction == "BUY":
            conflict = dxy_momentum > 0.0025 and us10y_momentum > 0.0020
            return conflict, context
        if direction == "SELL":
            conflict = dxy_momentum < -0.0025 and us10y_momentum < -0.0020
            return conflict, context
        return False, context

    def _price_action_risk(
        self,
        direction: str,
        candidate_data: Dict[str, Any],
        features: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        entry = float(candidate_data.get("entry_hint", 0.0) or 0.0)
        stop = float(candidate_data.get("stop_hint", 0.0) or 0.0)
        close = float(features.get("close", entry) or entry)
        atr = float(features.get("atr_14", features.get("atr", 0.0)) or 0.0)
        swing_high = float(features.get("swing_high", close) or close)
        swing_low = float(features.get("swing_low", close) or close)
        macd_hist = float(features.get("macd_histogram", features.get("macd_hist", 0.0)) or 0.0)
        rsi = float(features.get("rsi_14", features.get("rsi", 50.0)) or 50.0)
        ema20_slope = float(features.get("ema20_slope", features.get("ema_20_slope", 0.0)) or 0.0)
        spread = float(features.get("spread", 0.0) or 0.0)
        tick_volume = features.get("tick_volume")
        volume_ma = features.get("tick_volume_sma_20")

        stop_hunt_risk = False
        if atr > 0.0:
            if direction == "BUY":
                stop_hunt_risk = abs(entry - swing_low) <= (0.20 * atr) and spread > (0.20 * atr)
            elif direction == "SELL":
                stop_hunt_risk = abs(swing_high - entry) <= (0.20 * atr) and spread > (0.20 * atr)

        momentum_conflict = False
        if direction == "BUY":
            momentum_conflict = macd_hist < 0.0 or rsi < 45.0 or ema20_slope < 0.0
        elif direction == "SELL":
            momentum_conflict = macd_hist > 0.0 or rsi > 55.0 or ema20_slope > 0.0

        abnormal_volume = False
        if tick_volume is not None and volume_ma is not None:
            try:
                abnormal_volume = float(tick_volume) > (float(volume_ma) * 2.4)
            except Exception:
                abnormal_volume = False

        risk = stop_hunt_risk or momentum_conflict or abnormal_volume
        context = {
            "stop_hunt_risk": stop_hunt_risk,
            "momentum_conflict": momentum_conflict,
            "abnormal_volume": abnormal_volume,
            "entry": entry,
            "stop": stop,
            "close": close,
            "atr": atr,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "ema20_slope": ema20_slope,
        }
        return risk, context

    def _fetch_yahoo_close_series(
        self,
        ticker: str,
        interval: str = "5m",
        period: str = "1d",
    ) -> Optional[pd.Series]:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": interval, "range": period}
        try:
            response = self.session.get(url, params=params, timeout=4)
            response.raise_for_status()
            payload = response.json()
            result = payload.get("chart", {}).get("result", [])
            if not result:
                return None
            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            timestamps = result[0].get("timestamp", [])
            if not closes or not timestamps:
                return None
            series = pd.Series(closes, index=pd.to_datetime(timestamps, unit="s", utc=True))
            series = pd.to_numeric(series, errors="coerce").dropna()
            return series if not series.empty else None
        except Exception:
            return None

    @staticmethod
    def _parse_event_time(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = pd.to_datetime(text, utc=True)
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            return None

