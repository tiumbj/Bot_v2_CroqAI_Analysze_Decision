"""
Groq Response Parser
Version: 1.3.0
Purpose: Parse Groq API responses into AIDecision schema with strict validation
         and production-safe normalization for confidence / decision / wrapped JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

from models.schemas import AIDecision
from core.ai_confirmation_contract import AIConfirmationContract


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
        self.minimum_confidence = float(
            self.config.get("groq", {}).get("minimum_confidence", 0.6)
        )
        self.deep_confirmation = AIConfirmationContract()

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
            with config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            return config if isinstance(config, dict) else {}
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
            groq_response: Response dict from Groq client
            candidate: Candidate data (alternative to candidate_data)
            candidate_data: Original candidate data with entry/stop/target hints
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

        model_name = str(
            kwargs.get("model_name") or groq_response.get("model_name", "unknown")
        )
        latency_ms = self._normalize_latency(
            kwargs.get("latency_ms", groq_response.get("latency_ms", 0))
        )
        min_confidence = self._normalize_fraction(
            kwargs.get("minimum_confidence", self.minimum_confidence)
        )

        if not groq_response.get("success", False):
            error_msg = str(groq_response.get("error", "unknown error"))
            return self._create_deny_decision(
                reason=f"API call failed: {error_msg}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        content = str(groq_response.get("content", "") or "").strip()
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

        normalized = self._normalize_payload(parsed_json)

        required_fields = ["approved", "confidence", "decision", "reasoning"]
        missing_fields = [field for field in required_fields if field not in normalized]
        if missing_fields:
            return self._create_deny_decision(
                reason=f"Missing required fields: {missing_fields}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        try:
            approved = self._normalize_bool(normalized.get("approved", False))
            confidence = self._normalize_confidence(normalized.get("confidence", 0.0))
            decision_str = self._normalize_decision(normalized.get("decision", ""))
            reasoning = str(normalized.get("reasoning", "") or "").strip()
        except (TypeError, ValueError) as exc:
            return self._create_deny_decision(
                reason=f"Field type conversion error: {exc}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

        direction = str(candidate_info.get("direction", "BUY") or "BUY").upper()
        if direction not in {"BUY", "SELL"}:
            direction = "BUY"

        if decision_str == "APPROVE":
            decision = direction
        elif decision_str == "DENY":
            decision = direction
            approved = False
        elif decision_str in {"BUY", "SELL"}:
            decision = decision_str
        else:
            decision = direction
            approved = False
            reasoning = f"Invalid decision value: {decision_str}"

        if confidence < min_confidence:
            approved = False
            reasoning = f"Confidence {confidence:.2f} below minimum {min_confidence:.1f}"

        entry_hint = self._normalize_positive_price(candidate_info.get("entry_hint", 0.0))
        stop_hint = self._normalize_positive_price(candidate_info.get("stop_hint", 0.0))

        valid_response = True
        if entry_hint > 0.0 and stop_hint > 0.0:
            if decision == "BUY" and stop_hint >= entry_hint:
                valid_response = False
                approved = False
                reasoning = (
                    f"Invalid BUY stop-loss: stop ({stop_hint}) must be < entry ({entry_hint})"
                )
            elif decision == "SELL" and stop_hint <= entry_hint:
                valid_response = False
                approved = False
                reasoning = (
                    f"Invalid SELL stop-loss: stop ({stop_hint}) must be > entry ({entry_hint})"
                )

        if approved and valid_response:
            confirmation = self.deep_confirmation.evaluate(
                candidate_data=candidate_info,
                decision=decision,
            )
            if not confirmation.approved:
                approved = False
                valid_response = False
                context_text = json.dumps(confirmation.context, ensure_ascii=False)
                reasoning = f"{confirmation.reason}; context={context_text[:240]}"

        candidate_id = str(candidate_info.get("candidate_id", "unknown_candidate"))
        setup_quality = self._normalize_fraction(candidate_info.get("setup_quality", 0.0))
        trend_alignment = self._normalize_fraction(
            candidate_info.get("trend_alignment", 0.0)
        )
        regime_fit = self._normalize_fraction(candidate_info.get("regime_fit", 0.0))
        exhaustion_risk = self._normalize_fraction(
            candidate_info.get("exhaustion_risk", 0.0)
        )

        try:
            return AIDecision(
                candidate_id=candidate_id,
                decision=decision,
                approved=approved,
                confidence=confidence,
                entry_min=entry_hint,
                entry_max=entry_hint,
                stop_loss=stop_hint,
                setup_quality=setup_quality,
                trend_alignment=trend_alignment,
                regime_fit=regime_fit,
                exhaustion_risk=exhaustion_risk,
                reason=reasoning[:300] if reasoning else "No reasoning provided",
                model_name=model_name,
                prompt_version=prompt_version,
                latency_ms=latency_ms,
                valid_response=valid_response,
            )
        except Exception as exc:
            return self._create_deny_decision(
                reason=f"Schema validation failed: {exc}",
                candidate_info=candidate_info,
                model_name=model_name,
                latency_ms=latency_ms,
                prompt_version=prompt_version,
            )

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from content string.

        Supports:
        - raw JSON
        - fenced ```json ... ```
        - text wrapped around a JSON object
        """
        try:
            cleaned = content.strip()

            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                filtered_lines: list[str] = []
                for line in lines:
                    stripped = line.strip().lower()
                    if stripped.startswith("```"):
                        continue
                    if stripped == "json":
                        continue
                    filtered_lines.append(line)
                cleaned = "\n".join(filtered_lines).strip()

            try:
                parsed = json.loads(cleaned)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = cleaned[start : end + 1]
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else None

            return None
        except Exception:
            return None

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)

        if (
            isinstance(normalized.get("result"), dict)
            and "approved" not in normalized
            and "confidence" not in normalized
        ):
            normalized = dict(normalized["result"])

        field_aliases = {
            "approved": ["approved", "approve", "is_approved", "valid_trade"],
            "confidence": ["confidence", "confidence_score", "score", "probability"],
            "decision": ["decision", "action", "signal", "verdict"],
            "reasoning": ["reasoning", "reason", "rationale", "analysis", "explanation"],
            "risk_assessment": ["risk_assessment", "risk", "risk_level"],
            "suggested_position_size": [
                "suggested_position_size",
                "position_size",
                "size_fraction",
                "risk_fraction",
            ],
            "warnings": ["warnings", "warning", "alerts", "notes"],
        }

        output: Dict[str, Any] = {}
        for target_field, aliases in field_aliases.items():
            for alias in aliases:
                if alias in normalized:
                    output[target_field] = normalized[alias]
                    break

        return output

    def _normalize_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) != 0.0

        text = str(value or "").strip().lower()
        if text in {"true", "1", "yes", "y", "approve", "approved", "buy", "allow"}:
            return True
        if text in {"false", "0", "no", "n", "deny", "denied", "reject", "sell", "block"}:
            return False

        return False

    def _normalize_fraction(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            numeric = float(value)
        else:
            text = str(value or "").strip().replace("%", "")
            if not text:
                return 0.0
            numeric = float(text)

        if numeric > 1.0:
            numeric = numeric / 100.0

        if numeric < 0.0:
            return 0.0
        if numeric > 1.0:
            return 1.0
        return numeric

    def _normalize_confidence(self, value: Any) -> float:
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"high", "strong", "very high"}:
                return 0.85
            if text in {"medium", "moderate"}:
                return 0.65
            if text in {"low", "weak", "very low"}:
                return 0.35

        return self._normalize_fraction(value)

    def _normalize_decision(self, value: Any) -> str:
        text = str(value or "").strip().upper()
        if text in {"APPROVE", "APPROVED", "ALLOW", "VALID"}:
            return "APPROVE"
        if text in {"DENY", "DENIED", "REJECT", "REJECTED", "INVALID"}:
            return "DENY"
        if text in {"BUY", "SELL"}:
            return text
        return text

    def _normalize_positive_price(self, value: Any) -> float:
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0

        if numeric <= 0.0:
            return 0.00001
        return numeric

    def _normalize_latency(self, value: Any) -> int:
        try:
            latency = int(float(value or 0))
        except (TypeError, ValueError):
            latency = 0
        return max(latency, 0)

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
        """
        direction = str(candidate_info.get("direction", "BUY") or "BUY").upper()
        if direction not in {"BUY", "SELL"}:
            direction = "BUY"

        candidate_id = str(candidate_info.get("candidate_id", "unknown_candidate"))
        entry_hint = self._normalize_positive_price(candidate_info.get("entry_hint", 0.0))
        stop_hint = self._normalize_positive_price(candidate_info.get("stop_hint", 0.0))
        setup_quality = self._normalize_fraction(candidate_info.get("setup_quality", 0.0))
        trend_alignment = self._normalize_fraction(
            candidate_info.get("trend_alignment", 0.0)
        )
        regime_fit = self._normalize_fraction(candidate_info.get("regime_fit", 0.0))
        exhaustion_risk = self._normalize_fraction(
            candidate_info.get("exhaustion_risk", 0.0)
        )

        return AIDecision(
            candidate_id=candidate_id,
            decision=direction,
            approved=False,
            confidence=0.0,
            entry_min=entry_hint,
            entry_max=entry_hint,
            stop_loss=stop_hint,
            setup_quality=setup_quality,
            trend_alignment=trend_alignment,
            regime_fit=regime_fit,
            exhaustion_risk=exhaustion_risk,
            reason=(reason[:300] if reason else "Parser error") or "Parser error",
            model_name=model_name or "unknown",
            prompt_version=prompt_version or "v1",
            latency_ms=max(int(latency_ms), 0),
            valid_response=False,
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
        direction_upper = str(direction or "").upper()

        if direction_upper == "BUY":
            return stop_loss < entry_price
        if direction_upper == "SELL":
            return stop_loss > entry_price
        return False
