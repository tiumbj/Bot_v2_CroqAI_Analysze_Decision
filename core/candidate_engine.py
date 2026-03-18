"""
OracleBot-Pro
File: core/candidate_engine.py
Version: v1.0.0

Purpose
- Detect locked-v1 trade candidates from a FeatureSnapshot-like object
- Keep logic lean, deterministic, and production-safe
- Return one best candidate per symbol/bar at most

Notes
- This file is intentionally schema-tolerant:
  it can read data from dataclass/object/dict snapshots
- No AI logic here
- No execution logic here
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CandidateConfig:
    min_adx: float = 16.0
    min_bb_width: float = 0.0020
    min_ema_spread_ratio: float = 0.0002
    min_room_to_swing_atr: float = 0.80
    min_score_to_emit: float = 0.60
    min_rsi_buy: float = 52.0
    max_rsi_buy: float = 72.0
    min_rsi_sell: float = 28.0
    max_rsi_sell: float = 48.0
    atr_stop_buffer: float = 0.35
    adx_trend_threshold: float = 22.0
    adx_range_threshold: float = 18.0
    bb_width_high_vol: float = 0.0040
    bb_width_low_vol: float = 0.0015
    min_score_trend: float = 0.62
    min_score_range: float = 0.52
    min_score_volatile: float = 0.58
    min_score_quiet: float = 0.50
    sl_atr_trend: float = 2.0    # กว้างขึ้นเพื่อกัน Stop Hunt ใน Trend
    tp_atr_trend: float = 4.0    # หวังผล 1:2 หรือมากกว่า (Let profit run)
    sl_atr_range: float = 1.5    # กลยุทธ์ Mean Reversion ให้ SL แคบลงได้
    tp_atr_range: float = 2.25   # หวังผลกำไรแบบ RR 1:1.5 สำหรับสวิงสั้นๆ
    sl_atr_volatile: float = 3.0 # กว้างมากเพราะตลาดสวิงแรง
    tp_atr_volatile: float = 6.0 # หวังผล 1:2
    sl_atr_quiet: float = 1.2    # ตลาดซึม ตั้งแคบๆ ได้
    tp_atr_quiet: float = 1.8    # หวังผล 1:1.5


@dataclass(frozen=True)
class CandidateSignal:
    symbol: str
    timeframe: str
    bar_time: str
    direction: str
    score: float
    entry_hint: float
    stop_hint: float
    target_hint: float
    reasons: List[str]
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CandidateEngine:
    """
    Deterministic candidate detector for locked v1 feature set.
    Emits at most one candidate per snapshot.
    """

    def __init__(self, config: Optional[CandidateConfig] = None) -> None:
        self.config = config or CandidateConfig()

    def detect_candidate(self, snapshot: Any) -> Optional[CandidateSignal]:
        symbol = str(self._read(snapshot, "symbol", "UNKNOWN"))
        timeframe = str(self._read(snapshot, "timeframe", "UNKNOWN"))
        bar_time = self._normalize_time(
            self._read(snapshot, "bar_time", self._read(snapshot, "time", datetime.utcnow()))
        )

        regime = self._detect_market_regime(snapshot)
        adaptive_params = self._resolve_adaptive_params(regime)

        long_eval = self._evaluate_long(snapshot, adaptive_params)
        short_eval = self._evaluate_short(snapshot, adaptive_params)

        best = long_eval if long_eval["score"] >= short_eval["score"] else short_eval

        if best["score"] < adaptive_params["min_score"]:
            return None

        entry_hint = float(self._read(snapshot, "close", 0.0))
        atr = float(self._read(snapshot, "atr_14", self._read(snapshot, "atr", 0.0)))
        swing_low = float(self._read(snapshot, "swing_low", entry_hint))
        swing_high = float(self._read(snapshot, "swing_high", entry_hint))

        # Dynamic SL/TP based on Volatility + Market Structure
        if best["direction"] == "BUY":
            # หาว่าอะไรอยู่ไกลกว่ากันระหว่าง SL แบบตายตัว (ATR) กับ SL แบบโครงสร้าง (Swing Low)
            atr_sl_dist = atr * adaptive_params["sl_atr"]
            struct_sl_dist = entry_hint - swing_low if swing_low > 0 else 0
            
            # ใช้จุดที่ไกลกว่านิดหน่อยเพื่อกันโดนกวาด
            actual_sl_dist = max(atr_sl_dist, struct_sl_dist)
            
            # ไม่ควรให้ SL ไกลเกินไปจนรับความเสี่ยงไม่ไหว (Cap at 4 ATR)
            max_allowed_sl = atr * 4.0
            actual_sl_dist = min(actual_sl_dist, max_allowed_sl)
            
            stop_hint = entry_hint - actual_sl_dist
            
            # TP คำนวณจากระยะ SL จริง เพื่อรักษา RR (Dynamic RR)
            rr_ratio = adaptive_params["tp_atr"] / adaptive_params["sl_atr"]
            target_hint = entry_hint + (actual_sl_dist * rr_ratio)
            
        else:
            atr_sl_dist = atr * adaptive_params["sl_atr"]
            struct_sl_dist = swing_high - entry_hint if swing_high > 0 else 0
            
            actual_sl_dist = max(atr_sl_dist, struct_sl_dist)
            
            max_allowed_sl = atr * 4.0
            actual_sl_dist = min(actual_sl_dist, max_allowed_sl)
            
            stop_hint = entry_hint + actual_sl_dist
            
            rr_ratio = adaptive_params["tp_atr"] / adaptive_params["sl_atr"]
            target_hint = entry_hint - (actual_sl_dist * rr_ratio)

        return CandidateSignal(
            symbol=symbol,
            timeframe=timeframe,
            bar_time=bar_time,
            direction=best["direction"],
            score=round(best["score"], 4),
            entry_hint=round(entry_hint, 5),
            stop_hint=round(stop_hint, 5),
            target_hint=round(target_hint, 5),
            reasons=best["reasons"],
            features=self._extract_feature_subset(snapshot),
        )

    def _detect_market_regime(self, snapshot: Any) -> str:
        adx = float(self._read(snapshot, "adx_14", self._read(snapshot, "adx", 0.0)))
        bb_width = float(self._read(snapshot, "bb_width", self._read(snapshot, "bollinger_width", 0.0)))
        ema_spread_ratio = float(self._read(snapshot, "ema_spread_ratio", 0.0))
        atr = float(self._read(snapshot, "atr_14", self._read(snapshot, "atr", 0.0)))
        close = float(self._read(snapshot, "close", 0.0))

        # Detect high volatility based on ATR relative to price
        atr_to_price_ratio = atr / close if close > 0 else 0
        high_volatility = atr_to_price_ratio > 0.002  # 0.2% ATR/Price ratio

        if high_volatility or bb_width >= self.config.bb_width_high_vol:
            return "VOLATILE"

        if adx >= self.config.adx_trend_threshold and abs(ema_spread_ratio) >= self.config.min_ema_spread_ratio:
            return "TREND"

        if adx <= self.config.adx_range_threshold:
            if bb_width <= self.config.bb_width_low_vol:
                return "QUIET"
            return "RANGE"

        return "RANGE"

    def _resolve_adaptive_params(self, regime: str) -> Dict[str, float]:
        if regime == "TREND":
            return {
                "min_score": self.config.min_score_trend,
                "min_adx": max(self.config.min_adx, 20.0),
                "min_bb_width": self.config.min_bb_width,
                "min_room_atr": self.config.min_room_to_swing_atr,
                "sl_atr": self.config.sl_atr_trend,
                "tp_atr": self.config.tp_atr_trend,
                "min_rsi_buy": 50.0,
                "max_rsi_buy": 75.0,
                "min_rsi_sell": 25.0,
                "max_rsi_sell": 50.0,
            }
        if regime == "VOLATILE":
            return {
                "min_score": self.config.min_score_volatile,
                "min_adx": max(self.config.min_adx, 22.0),
                "min_bb_width": max(self.config.min_bb_width, self.config.bb_width_high_vol),
                "min_room_atr": max(self.config.min_room_to_swing_atr, 1.0),
                "sl_atr": self.config.sl_atr_volatile,
                "tp_atr": self.config.tp_atr_volatile,
                "min_rsi_buy": self.config.min_rsi_buy,
                "max_rsi_buy": self.config.max_rsi_buy,
                "min_rsi_sell": self.config.min_rsi_sell,
                "max_rsi_sell": self.config.max_rsi_sell,
            }
        if regime == "QUIET":
            return {
                "min_score": self.config.min_score_quiet,
                "min_adx": 12.0,
                "min_bb_width": self.config.bb_width_low_vol,
                "min_room_atr": 0.55,
                "sl_atr": self.config.sl_atr_quiet,
                "tp_atr": self.config.tp_atr_quiet,
                "min_rsi_buy": 48.0,
                "max_rsi_buy": 68.0,
                "min_rsi_sell": 32.0,
                "max_rsi_sell": 52.0,
            }
        return {
            "min_score": self.config.min_score_range,
            "min_adx": 12.0,
            "min_bb_width": self.config.bb_width_low_vol,
            "min_room_atr": 0.60,
            "sl_atr": self.config.sl_atr_range,
            "tp_atr": self.config.tp_atr_range,
            "min_rsi_buy": 45.0,
            "max_rsi_buy": 70.0,
            "min_rsi_sell": 30.0,
            "max_rsi_sell": 55.0,
        }

    def _evaluate_long(self, snapshot: Any, adaptive_params: Dict[str, float]) -> Dict[str, Any]:
        checks: List[bool] = []
        reasons: List[str] = []

        close = float(self._read(snapshot, "close", 0.0))
        ema20 = float(self._read(snapshot, "ema_20", 0.0))
        ema50 = float(self._read(snapshot, "ema_50", 0.0))
        ema200 = float(self._read(snapshot, "ema_200", 0.0))
        ema20_slope = float(self._read(snapshot, "ema20_slope", self._read(snapshot, "ema_20_slope", 0.0)))
        ema_spread_ratio = float(self._read(snapshot, "ema_spread_ratio", 0.0))
        rsi = float(self._read(snapshot, "rsi_14", self._read(snapshot, "rsi", 50.0)))
        macd_hist = float(self._read(snapshot, "macd_histogram", self._read(snapshot, "macd_hist", 0.0)))
        adx = float(self._read(snapshot, "adx_14", self._read(snapshot, "adx", 0.0)))
        di_plus = float(self._read(snapshot, "di_plus", self._read(snapshot, "plus_di", 0.0)))
        di_minus = float(self._read(snapshot, "di_minus", self._read(snapshot, "minus_di", 0.0)))
        bb_width = float(self._read(snapshot, "bb_width", self._read(snapshot, "bollinger_width", 0.0)))
        room_to_high = float(
            self._read(
                snapshot,
                "distance_to_swing_high_atr",
                self._read(snapshot, "dist_to_swing_high_atr", 999.0),
            )
        )
        breakout_state = str(self._read(snapshot, "breakout_state", "")).lower()
        retest_state = str(self._read(snapshot, "retest_state", "")).lower()

        trend_ok = close > ema20 > ema50 > ema200
        if trend_ok:
            reasons.append("trend_alignment_bullish")
        checks.append(trend_ok)

        slope_ok = ema20_slope > 0.0 and ema_spread_ratio >= self.config.min_ema_spread_ratio
        if slope_ok:
            reasons.append("ema_slope_positive")
        checks.append(slope_ok)

        momentum_ok = adaptive_params["min_rsi_buy"] <= rsi <= adaptive_params["max_rsi_buy"] and macd_hist > 0.0
        if momentum_ok:
            reasons.append("momentum_bullish")
        checks.append(momentum_ok)

        strength_ok = adx >= adaptive_params["min_adx"] and di_plus > di_minus
        if strength_ok:
            reasons.append("trend_strength_confirmed")
        checks.append(strength_ok)

        volatility_ok = bb_width >= adaptive_params["min_bb_width"]
        if volatility_ok:
            reasons.append("volatility_sufficient")
        checks.append(volatility_ok)

        structure_ok = (
            breakout_state in {"bullish", "long", "up"}
            or retest_state in {"bullish", "long", "support", "up"}
        )
        if structure_ok:
            reasons.append("structure_bullish")
        checks.append(structure_ok)

        room_ok = room_to_high >= adaptive_params["min_room_atr"]
        if room_ok:
            reasons.append("room_to_high_available")
        checks.append(room_ok)

        score = sum(1 for item in checks if item) / len(checks)
        return {
            "direction": "BUY",
            "score": score,
            "reasons": reasons[:4],
        }

    def _evaluate_short(self, snapshot: Any, adaptive_params: Dict[str, float]) -> Dict[str, Any]:
        checks: List[bool] = []
        reasons: List[str] = []

        close = float(self._read(snapshot, "close", 0.0))
        ema20 = float(self._read(snapshot, "ema_20", 0.0))
        ema50 = float(self._read(snapshot, "ema_50", 0.0))
        ema200 = float(self._read(snapshot, "ema_200", 0.0))
        ema20_slope = float(self._read(snapshot, "ema20_slope", self._read(snapshot, "ema_20_slope", 0.0)))
        ema_spread_ratio = float(self._read(snapshot, "ema_spread_ratio", 0.0))
        rsi = float(self._read(snapshot, "rsi_14", self._read(snapshot, "rsi", 50.0)))
        macd_hist = float(self._read(snapshot, "macd_histogram", self._read(snapshot, "macd_hist", 0.0)))
        adx = float(self._read(snapshot, "adx_14", self._read(snapshot, "adx", 0.0)))
        di_plus = float(self._read(snapshot, "di_plus", self._read(snapshot, "plus_di", 0.0)))
        di_minus = float(self._read(snapshot, "di_minus", self._read(snapshot, "minus_di", 0.0)))
        bb_width = float(self._read(snapshot, "bb_width", self._read(snapshot, "bollinger_width", 0.0)))
        room_to_low = float(
            self._read(
                snapshot,
                "distance_to_swing_low_atr",
                self._read(snapshot, "dist_to_swing_low_atr", 999.0),
            )
        )
        breakout_state = str(self._read(snapshot, "breakout_state", "")).lower()
        retest_state = str(self._read(snapshot, "retest_state", "")).lower()

        trend_ok = close < ema20 < ema50 < ema200
        if trend_ok:
            reasons.append("trend_alignment_bearish")
        checks.append(trend_ok)

        slope_ok = ema20_slope < 0.0 and ema_spread_ratio <= (-1.0 * self.config.min_ema_spread_ratio)
        if slope_ok:
            reasons.append("ema_slope_negative")
        checks.append(slope_ok)

        momentum_ok = adaptive_params["min_rsi_sell"] <= rsi <= adaptive_params["max_rsi_sell"] and macd_hist < 0.0
        if momentum_ok:
            reasons.append("momentum_bearish")
        checks.append(momentum_ok)

        strength_ok = adx >= adaptive_params["min_adx"] and di_minus > di_plus
        if strength_ok:
            reasons.append("trend_strength_confirmed")
        checks.append(strength_ok)

        volatility_ok = bb_width >= adaptive_params["min_bb_width"]
        if volatility_ok:
            reasons.append("volatility_sufficient")
        checks.append(volatility_ok)

        structure_ok = (
            breakout_state in {"bearish", "short", "down"}
            or retest_state in {"bearish", "short", "resistance", "down"}
        )
        if structure_ok:
            reasons.append("structure_bearish")
        checks.append(structure_ok)

        room_ok = room_to_low >= adaptive_params["min_room_atr"]
        if room_ok:
            reasons.append("room_to_low_available")
        checks.append(room_ok)

        score = sum(1 for item in checks if item) / len(checks)
        return {
            "direction": "SELL",
            "score": score,
            "reasons": reasons[:4],
        }

    def _extract_feature_subset(self, snapshot: Any) -> Dict[str, Any]:
        keys = [
            "close",
            "ema_20",
            "ema_50",
            "ema_200",
            "ema20_slope",
            "ema_spread_ratio",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "atr_14",
            "adx_14",
            "di_plus",
            "di_minus",
            "bb_width",
            "swing_high",
            "swing_low",
            "distance_to_swing_high_atr",
            "distance_to_swing_low_atr",
            "breakout_state",
            "retest_state",
            "spread",
            "session",
            "open_position_flag",
        ]

        subset: Dict[str, Any] = {}
        for key in keys:
            value = self._read(snapshot, key, None)
            if value is not None:
                subset[key] = value
        return subset

    @staticmethod
    def _normalize_time(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _read(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        if is_dataclass(obj):
            data = asdict(obj)
            return data.get(key, default)

        if hasattr(obj, key):
            return getattr(obj, key, default)

        return default
