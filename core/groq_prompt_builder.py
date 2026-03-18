"""
Groq Prompt Builder
Version: 1.0.2
Purpose: Build deterministic prompts from candidate data for Groq AI decision analysis
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class GroqPromptBuilder:
    """
    Builds structured prompts for Groq API from candidate/postprocessed data.
    Keeps prompts concise, deterministic, and decision-focused.
    """

    PROMPT_VERSION = "v1.0.2"

    @classmethod
    def build_decision_prompt(
        cls,
        symbol: str,
        timeframe: str,
        direction: str,
        score: float,
        entry_hint: Optional[float] = None,
        stop_hint: Optional[float] = None,
        target_hint: Optional[float] = None,
        reasons: Optional[list] = None,
        features: Optional[Dict[str, Any]] = None,
        bar_time: Optional[str] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Build a complete prompt for Groq decision analysis.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            direction: Trade direction ('BUY' or 'SELL')
            score: Candidate confidence score (0.0-1.0)
            entry_hint: Suggested entry price
            stop_hint: Suggested stop loss price
            target_hint: Suggested take profit price
            reasons: List of detection reasons
            features: Additional feature data
            bar_time: Timestamp of the bar

        Returns:
            Dict with 'system', 'user', and 'prompt_version'
        """
        system_prompt = cls._build_system_prompt()
        user_prompt = cls._build_user_prompt(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            score=score,
            entry_hint=entry_hint,
            stop_hint=stop_hint,
            target_hint=target_hint,
            reasons=reasons,
            features=features,
            bar_time=bar_time,
            market_context=market_context,
        )
        return {
            "system": system_prompt,
            "user": user_prompt,
            "prompt_version": cls.PROMPT_VERSION,
        }

    @classmethod
    def _build_system_prompt(cls) -> str:
        """Build the system-level instruction prompt."""
        return """You are a professional trading decision validator for market trade candidates.

Goal:
- Return compact JSON decision for execution safety.
- Prefer deny when risk signals are inconsistent.

Response format (strict JSON only):
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "decision": "APPROVE" or "DENY",
  "reasoning": "brief explanation",
  "risk_assessment": "LOW/MEDIUM/HIGH",
  "suggested_position_size": 0.0-1.0,
  "warnings": ["warning1", "warning2"] or []
}

Rules:
- BUY requires stop_loss < entry_price
- SELL requires stop_loss > entry_price
- Use supplied validation facts as source-of-truth
- Deny if confidence < 0.60
- Keep reasoning <= 160 chars
- Return JSON only"""

    @classmethod
    def _build_user_prompt(
        cls,
        symbol: str,
        timeframe: str,
        direction: str,
        score: float,
        entry_hint: Optional[float],
        stop_hint: Optional[float],
        target_hint: Optional[float],
        reasons: Optional[list],
        features: Optional[Dict[str, Any]],
        bar_time: Optional[str],
        market_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the user-specific candidate data prompt."""
        prompt_parts = [
            "=== TRADE CANDIDATE ANALYSIS REQUEST ===",
            f"Symbol: {symbol}",
            f"Timeframe: {timeframe}",
            f"Direction: {direction}",
            f"Candidate Score: {score:.4f}",
        ]

        if bar_time:
            prompt_parts.append(f"Bar Time: {bar_time}")
        if entry_hint is not None:
            prompt_parts.append(f"Entry Price: {entry_hint}")
        if stop_hint is not None:
            prompt_parts.append(f"Stop Loss: {stop_hint}")
        if target_hint is not None:
            prompt_parts.append(f"Take Profit: {target_hint}")

        stop_loss_rule = "UNKNOWN"
        stop_loss_validity = "UNKNOWN"
        stop_loss_distance = "UNKNOWN"

        if entry_hint is not None and stop_hint is not None:
            if direction == "BUY":
                stop_loss_rule = "BUY requires stop_loss < entry_price"
                stop_loss_validity = "VALID" if stop_hint < entry_hint else "INVALID"
                stop_loss_distance = f"{abs(entry_hint - stop_hint):.2f}"
            elif direction == "SELL":
                stop_loss_rule = "SELL requires stop_loss > entry_price"
                stop_loss_validity = "VALID" if stop_hint > entry_hint else "INVALID"
                stop_loss_distance = f"{abs(stop_hint - entry_hint):.2f}"

        prompt_parts.append("")
        prompt_parts.append("Precomputed Validation Facts:")
        prompt_parts.append(f"Stop Loss Rule: {stop_loss_rule}")
        prompt_parts.append(f"Stop Loss Validity: {stop_loss_validity}")
        prompt_parts.append(f"Stop Loss Distance: {stop_loss_distance}")
        prompt_parts.append("Use these validation facts as source-of-truth.")
        prompt_parts.append("Do not recalculate stop-loss validity differently from these facts.")

        if entry_hint is not None and stop_hint is not None and target_hint is not None:
            if direction == "BUY":
                risk = abs(entry_hint - stop_hint)
                reward = abs(target_hint - entry_hint)
            else:
                risk = abs(stop_hint - entry_hint)
                reward = abs(entry_hint - target_hint)

            if risk > 0:
                rr_ratio = reward / risk
                prompt_parts.append(f"Risk-Reward Ratio: {rr_ratio:.2f}")

        if reasons:
            prompt_parts.append("")
            prompt_parts.append("Detection Reasons:")
            for idx, reason in enumerate(reasons[:4], 1):
                prompt_parts.append(f"{idx}. {reason}")

        if features:
            feature_keys = [
                "close",
                "ema_20",
                "ema_50",
                "ema_200",
                "ema20_slope",
                "ema_spread_ratio",
                "rsi_14",
                "macd_histogram",
                "atr_14",
                "adx_14",
                "di_plus",
                "di_minus",
                "bb_width",
                "distance_to_swing_high_atr",
                "distance_to_swing_low_atr",
                "breakout_state",
                "retest_state",
                "spread",
                "session",
            ]
            prompt_parts.append("")
            prompt_parts.append("Technical Features:")
            for key in feature_keys:
                if key not in features:
                    continue
                value = features.get(key)
                if isinstance(value, float):
                    prompt_parts.append(f"{key}: {value:.4f}")
                else:
                    prompt_parts.append(f"{key}: {value}")

        if market_context:
            prompt_parts.append("")
            prompt_parts.append("Market Context:")
            for key in ["news_risk", "correlation_risk", "price_action_risk", "notes"]:
                if key in market_context:
                    prompt_parts.append(f"{key}: {market_context[key]}")

        prompt_parts.append("")
        prompt_parts.append("=== VALIDATE AND DECIDE ===")
        prompt_parts.append(
            "Analyze this candidate and return your decision in strict JSON format."
        )
        prompt_parts.append(
            "If Stop Loss Validity is VALID, do not deny the trade for stop-loss placement."
        )

        return "\n".join(prompt_parts)

    @classmethod
    def get_prompt_version(cls) -> str:
        """Return current prompt version for tracking."""
        return cls.PROMPT_VERSION
