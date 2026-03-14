"""
Groq Response Parser
Version: 1.2.0
Purpose: Parse Groq API responses into AIDecision schema with strict validation
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from models.schemas import AIDecision


class GroqResponseParser:
    """
    Parses Groq API responses into AIDecision objects.
    Implements fail-closed logic: deny on any parsing/validation failure.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize parser with configuration.

        Args:
            config_path: Path to ai.yaml config file. If None, uses default location.
        """
        self.config = self._load_config(config_path)
        self.minimum_confidence = self.config.get("groq", {}).get("minimum_confidence", 0.6)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from ai.yaml."""
        if config_path is None:
            repo_root = Path(__file__).parent.parent
            config_path = repo_root / "config" / "ai.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            return {"groq": {"minimum_confidence": 0.6}}

        if yaml is None:
            return {"groq": {"minimum_confidence": 0.6}}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except Exception:
            return {"groq": {"minimum_confidence": 0.6}}

    def parse(
        self,
        groq_response: Optional[Dict[str, Any]] = None,
        candidate: Optional[Dict[str, Any]] = None,
        candidate_data: Optional[Dict[str, Any]] = None,
        prompt_version: str = "v1",
        **kwargs: Any,
    ) -> AIDecision:
        """
        Parse Groq API response into AIDecision object.

        Args:
            groq_response: Response dict from GroqClient.chat_completion()
            candidate: Candidate data (alternative to candidate_data)
            candidate_Original candidate data with entry/stop/target hints
            prompt_version: Version of prompt used for this request
            **kwargs: Additional arguments (minimum_confidence, model_name, latency_ms, etc.)

        Returns:
            AIDecision object (fail-closed on errors)
        """
        if groq_response is None:
            groq_response = {}

        candidate_info = candidate_data if candidate_data is not None else candidate
        if candidate_info is None:
            candidate_info = {}

        model_name = kwargs.get("model_name") or groq_response.get("model_name", "unknown")
        latency_ms = kwargs.get("latency_ms") or groq_response.get("latency_ms", 0)
        min_confidence = kwargs.get("minimum_confidence", self.minimum_confidence)

        if not groq_response.get("success", False):
            error_msg = groq_response.get("error", "unknown error")
            return self._create_deny_decision(
                reason=f"API call failed: {error_msg}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        content = groq_response.get("content", "").strip()
        if not content:
            return self._create_deny_decision(
                reason="Empty response from model",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        parsed_json = self._extract_json(content)
        if parsed_json is None:
            return self._create_deny_decision(
                reason="Invalid JSON response",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        required_fields = ["approved", "confidence", "decision", "reasoning"]
        missing_fields = [f for f in required_fields if f not in parsed_json]
        if missing_fields:
            return self._create_deny_decision(
                reason=f"Missing required fields: {missing_fields}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        try:
            approved = bool(parsed_json.get("approved", False))
            confidence = float(parsed_json.get("confidence", 0.0))
            decision_str = str(parsed_json.get("decision", "")).upper()
            reasoning = str(parsed_json.get("reasoning", ""))
            risk_assessment = str(parsed_json.get("risk_assessment", "UNKNOWN"))
            suggested_position_size = float(parsed_json.get("suggested_position_size", 0.0))
            warnings = parsed_json.get("warnings", [])
            if not isinstance(warnings, list):
                warnings = []
        except (ValueError, TypeError) as e:
            return self._create_deny_decision(
                reason=f"Field type conversion error: {str(e)}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        if confidence < min_confidence:
            approved = False
            reasoning = f"Confidence {confidence:.2f} below minimum {min_confidence}"

        direction = candidate_info.get("direction", "BUY").upper()
        if direction not in ["BUY", "SELL"]:
            direction = "BUY"

        if decision_str == "APPROVE":
            decision = direction
        elif decision_str == "DENY":
            approved = False
            decision = direction
        elif decision_str in ["BUY", "SELL"]:
            decision = decision_str
        else:
            approved = False
            decision = direction
            reasoning = f"Invalid decision value: {decision_str}"

        entry_hint = candidate_info.get("entry_hint")
        stop_hint = candidate_info.get("stop_hint")
        target_hint = candidate_info.get("target_hint")

        if entry_hint is None:
            entry_hint = 0.0
        if stop_hint is None:
            stop_hint = 0.0
        if target_hint is None:
            target_hint = 0.0

        entry_min = float(entry_hint)
        entry_max = float(entry_hint)
        stop_loss = float(stop_hint)

        valid_response = True
        if entry_min > 0 and stop_loss > 0:
            if decision == "BUY":
                if stop_loss >= entry_min:
                    valid_response = False
                    approved = False
                    reasoning = f"Invalid BUY stop-loss: stop ({stop_loss}) must be < entry ({entry_min})"
            elif decision == "SELL":
                if stop_loss <= entry_max:
                    valid_response = False
                    approved = False
                    reasoning = f"Invalid SELL stop-loss: stop ({stop_loss}) must be > entry ({entry_max})"

        candidate_id = candidate_info.get("candidate_id", "unknown_candidate")
        setup_quality = candidate_info.get("setup_quality", 0.0)
        trend_alignment = candidate_info.get("trend_alignment", 0.0)
        regime_fit = candidate_info.get("regime_fit", 0.0)
        exhaustion_risk = candidate_info.get("exhaustion_risk", 0.0)

        try:
            ai_decision = AIDecision(
                candidate_id=candidate_id,
                decision=decision,
                entry_min=entry_min,
                entry_max=entry_max,
                stop_loss=stop_loss,
                setup_quality=setup_quality,
                trend_alignment=trend_alignment,
                regime_fit=regime_fit,
                exhaustion_risk=exhaustion_risk,
                reason=reasoning,
                approved=approved,
                valid_response=valid_response,
                confidence=confidence,
                risk_assessment=risk_assessment,
                suggested_position_size=suggested_position_size,
                warnings=warnings,
                model_name=model_name,
                prompt_version=prompt_version,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
            )
            return ai_decision
        except Exception as e:
            return self._create_deny_decision(
                reason=f"Schema validation failed: {str(e)}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from content string.

        Args:
            content: Raw content string that may contain JSON

        Returns:
            Parsed JSON dict or None if invalid
        """
        try:
            cleaned = content.strip()

            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                filtered_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped.startswith("```") and stripped != "json":
                        filtered_lines.append(line)
                cleaned = "\n".join(filtered_lines).strip()

            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _create_deny_decision(
        self,
        reason: str,
        candidate_info: Dict[str, Any],
        model_name: str,
        latency_ms: int,
        prompt_version: str,
    ) -> AIDecision:
        """
        Create a safe deny-style AIDecision for error cases.

        Args:
            reason: Explanation for denial
            candidate_info: Candidate data for fallback values
            model_name: Model name for tracking
            latency_ms: Request latency
            prompt_version: Prompt version used

        Returns:
            AIDecision with approved=False, valid_response=False
        """
        direction = candidate_info.get("direction", "BUY").upper()
        if direction not in ["BUY", "SELL"]:
            direction = "BUY"

        candidate_id = candidate_info.get("candidate_id", "unknown_candidate")
        entry_hint = candidate_info.get("entry_hint", 0.0)
        stop_hint = candidate_info.get("stop_hint", 0.0)
        setup_quality = candidate_info.get("setup_quality", 0.0)
        trend_alignment = candidate_info.get("trend_alignment", 0.0)
        regime_fit = candidate_info.get("regime_fit", 0.0)
        exhaustion_risk = candidate_info.get("exhaustion_risk", 0.0)

        return AIDecision(
            candidate_id=candidate_id,
            decision=direction,
            entry_min=float(entry_hint),
            entry_max=float(entry_hint),
            stop_loss=float(stop_hint),
            setup_quality=float(setup_quality),
            trend_alignment=float(trend_alignment),
            regime_fit=float(regime_fit),
            exhaustion_risk=float(exhaustion_risk),
            reason=reason,
            approved=False,
            valid_response=False,
            confidence=0.0,
            risk_assessment="UNKNOWN",
            suggested_position_size=0.0,
            warnings=["Parser error - fail-closed deny"],
            model_name=model_name,
            prompt_version=prompt_version,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )

    def validate_stop_loss_logic(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> bool:
        """
        Validate stop-loss placement logic.

        Args:
            direction: Trade direction ('BUY' or 'SELL')
            entry_price: Entry price level
            stop_loss: Stop loss price level

        Returns:
            True if stop-loss logic is valid, False otherwise
        """
        direction_upper = direction.upper()

        if direction_upper == "BUY":
            return stop_loss < entry_price
        elif direction_upper == "SELL":
            return stop_loss > entry_price
        else:
            return False