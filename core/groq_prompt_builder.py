"""
Groq Prompt Builder
Version: 1.0.0
Purpose: Build deterministic prompts from candidate data for Groq AI decision analysis
"""

from typing import Dict, Any, Optional
from datetime import datetime


class GroqPromptBuilder:
    """
    Builds structured prompts for Groq API from candidate/postprocessed data.
    Keeps prompts concise, deterministic, and decision-focused.
    """

    PROMPT_VERSION = "v1.0.0"

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
    ) -> Dict[str, str]:
        """
        Build a complete prompt for Groq decision analysis.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h')
            direction: Trade direction ('BUY' or 'SELL')
            score: Candidate confidence score (0.0-1.0)
            entry_hint: Suggested entry price
            stop_hint: Suggested stop loss price
            target_hint: Suggested take profit price
            reasons: List of detection reasons
            features: Additional feature data
            bar_time: Timestamp of the bar

        Returns:
            Dict with 'system' and 'user' prompt messages
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
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
            "prompt_version": cls.PROMPT_VERSION,
        }

    @classmethod
    def _build_system_prompt(cls) -> str:
        """Build the system-level instruction prompt."""
        return """You are a professional trading decision validator for cryptocurrency markets.

Your role:
1. Analyze candidate trade signals with technical and risk parameters
2. Validate stop-loss placement logic (BUY: stop < entry, SELL: stop > entry)
3. Assess risk-reward ratio and trade quality
4. Return a structured JSON decision

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

Critical rules:
- BUY trades: stop_loss MUST be below entry_price
- SELL trades: stop_loss MUST be above entry_price
- If stop-loss logic is invalid, always deny (approved=false)
- If confidence < 0.6, recommend deny
- If risk is HIGH without strong reasoning, recommend deny
- Always return valid JSON only, no markdown, no explanations outside JSON"""

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
    ) -> str:
        """Build the user-specific candidate data prompt."""
        prompt_parts = [
            "=== TRADE CANDIDATE ANALYSIS REQUEST ===\n",
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

        # Calculate risk-reward if we have all prices
        if entry_hint and stop_hint and target_hint:
            if direction == "BUY":
                risk = abs(entry_hint - stop_hint)
                reward = abs(target_hint - entry_hint)
            else:  # SELL
                risk = abs(stop_hint - entry_hint)
                reward = abs(entry_hint - target_hint)

            if risk > 0:
                rr_ratio = reward / risk
                prompt_parts.append(f"Risk-Reward Ratio: {rr_ratio:.2f}")

        if reasons:
            prompt_parts.append(f"\nDetection Reasons:")
            for idx, reason in enumerate(reasons, 1):
                prompt_parts.append(f"  {idx}. {reason}")

        if features:
            prompt_parts.append(f"\nTechnical Features:")
            for key, value in features.items():
                if isinstance(value, float):
                    prompt_parts.append(f"  {key}: {value:.4f}")
                else:
                    prompt_parts.append(f"  {key}: {value}")

        prompt_parts.append("\n=== VALIDATE AND DECIDE ===")
        prompt_parts.append(
            "Analyze this candidate and return your decision in strict JSON format."
        )

        return "\n".join(prompt_parts)

    @classmethod
    def get_prompt_version(cls) -> str:
        """Return current prompt version for tracking."""
        return cls.PROMPT_VERSION