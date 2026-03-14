"""
Groq Decision Smoke Test
Version: 1.0.0
Purpose: Independent smoke test for Groq adapter layer (prompt builder, client, parser)
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.groq_prompt_builder import GroqPromptBuilder
from core.groq_client import GroqClient, GroqClientError
from core.groq_response_parser import GroqResponseParser
from models.schemas import AIDecision


class GroqSmokeTest:
    """Smoke test suite for Groq integration layer."""

    def __init__(self):
        self.results = []
        self.parser = GroqResponseParser()
        self.client = None
        try:
            self.client = GroqClient()
        except GroqClientError as e:
            print(f"⚠️  Groq client initialization warning: {e}")

    def run_all_tests(self):
        """Run all smoke tests and print summary."""
        print("=" * 70)
        print("GROQ DECISION SMOKE TEST")
        print("=" * 70)
        print()

        # Test 1: Safe deny path
        self.test_safe_deny_path()

        # Test 2: Malformed response
        self.test_malformed_response()

        # Test 3: BUY valid stop loss
        self.test_buy_valid_stop_loss()

        # Test 4: BUY invalid stop loss
        self.test_buy_invalid_stop_loss()

        # Test 5: SELL valid stop loss
        self.test_sell_valid_stop_loss()

        # Test 6: SELL invalid stop loss
        self.test_sell_invalid_stop_loss()

        # Test 7: Low confidence rejection
        self.test_low_confidence_rejection()

        # Test 8: Missing required fields
        self.test_missing_required_fields()

        # Test 9: Prompt builder
        self.test_prompt_builder()

        # Test 10: AIDecision schema validation
        self.test_schema_validation()

        # Test 11: Live Groq API call (if configured)
        self.test_live_groq_call()

        # Print summary
        self.print_summary()

    def test_safe_deny_path(self):
        """Test 1: Safe deny on API failure."""
        print("Test 1: Safe Deny Path (API Failure)")
        print("-" * 70)

        groq_response = {
            "success": False,
            "content": "",
            "model_name": "test-model",
            "latency_ms": 100,
            "error": "Connection timeout",
        }

        candidate_data = {
            "direction": "BUY",
            "entry_hint": 50000.0,
            "stop_hint": 49500.0,
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is False
            and decision.valid_response is False
            and "API call failed" in decision.reason
        )

        self._record_result("Safe Deny Path", passed, decision)
        print()

    def test_malformed_response(self):
        """Test 2: Deny on malformed JSON."""
        print("Test 2: Malformed JSON Response")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": "This is not valid JSON at all!",
            "model_name": "test-model",
            "latency_ms": 150,
        }

        candidate_data = {
            "direction": "BUY",
            "entry_hint": 50000.0,
            "stop_hint": 49500.0,
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is False
            and decision.valid_response is False
            and "Invalid JSON" in decision.reason
        )

        self._record_result("Malformed Response", passed, decision)
        print()

    def test_buy_valid_stop_loss(self):
        """Test 3: BUY with valid stop loss (stop < entry)."""
        print("Test 3: BUY Valid Stop Loss")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": """{
                "approved": true,
                "confidence": 0.85,
                "decision": "APPROVE",
                "reasoning": "Strong bullish setup with valid risk management",
                "risk_assessment": "MEDIUM",
                "suggested_position_size": 0.7,
                "warnings": []
            }""",
            "model_name": "test-model",
            "latency_ms": 200,
        }

        candidate_data = {
            "direction": "BUY",
            "entry_hint": 50000.0,
            "stop_hint": 49500.0,  # Valid: stop < entry
            "target_hint": 51000.0,
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is True
            and decision.valid_response is True
            and decision.confidence == 0.85
        )

        self._record_result("BUY Valid Stop Loss", passed, decision)
        print()

    def test_buy_invalid_stop_loss(self):
        """Test 4: BUY with invalid stop loss (stop >= entry)."""
        print("Test 4: BUY Invalid Stop Loss")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": """{
                "approved": true,
                "confidence": 0.85,
                "decision": "APPROVE",
                "reasoning": "Looks good",
                "risk_assessment": "LOW",
                "suggested_position_size": 0.8,
                "warnings": []
            }""",
            "model_name": "test-model",
            "latency_ms": 180,
        }

        candidate_data = {
            "direction": "BUY",
            "entry_hint": 50000.0,
            "stop_hint": 50500.0,  # Invalid: stop > entry for BUY
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is False
            and decision.valid_response is False
            and "Invalid BUY stop-loss" in decision.reason
        )

        self._record_result("BUY Invalid Stop Loss", passed, decision)
        print()

    def test_sell_valid_stop_loss(self):
        """Test 5: SELL with valid stop loss (stop > entry)."""
        print("Test 5: SELL Valid Stop Loss")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": """{
                "approved": true,
                "confidence": 0.78,
                "decision": "APPROVE",
                "reasoning": "Bearish momentum confirmed",
                "risk_assessment": "MEDIUM",
                "suggested_position_size": 0.6,
                "warnings": ["High volatility expected"]
            }""",
            "model_name": "test-model",
            "latency_ms": 190,
        }

        candidate_data = {
            "direction": "SELL",
            "entry_hint": 50000.0,
            "stop_hint": 50500.0,  # Valid: stop > entry for SELL
            "target_hint": 49000.0,
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is True
            and decision.valid_response is True
            and decision.confidence == 0.78
        )

        self._record_result("SELL Valid Stop Loss", passed, decision)
        print()

    def test_sell_invalid_stop_loss(self):
        """Test 6: SELL with invalid stop loss (stop <= entry)."""
        print("Test 6: SELL Invalid Stop Loss")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": """{
                "approved": true,
                "confidence": 0.90,
                "decision": "APPROVE",
                "reasoning": "Perfect setup",
                "risk_assessment": "LOW",
                "suggested_position_size": 0.9,
                "warnings": []
            }""",
            "model_name": "test-model",
            "latency_ms": 175,
        }

        candidate_data = {
            "direction": "SELL",
            "entry_hint": 50000.0,
            "stop_hint": 49500.0,  # Invalid: stop < entry for SELL
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is False
            and decision.valid_response is False
            and "Invalid SELL stop-loss" in decision.reason
        )

        self._record_result("SELL Invalid Stop Loss", passed, decision)
        print()

    def test_low_confidence_rejection(self):
        """Test 7: Deny on low confidence."""
        print("Test 7: Low Confidence Rejection")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": """{
                "approved": true,
                "confidence": 0.45,
                "decision": "APPROVE",
                "reasoning": "Weak signal but trying",
                "risk_assessment": "HIGH",
                "suggested_position_size": 0.3,
                "warnings": ["Low confidence"]
            }""",
            "model_name": "test-model",
            "latency_ms": 160,
        }

        candidate_data = {
            "direction": "BUY",
            "entry_hint": 50000.0,
            "stop_hint": 49500.0,
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is False
            and "below minimum" in decision.reason
        )

        self._record_result("Low Confidence Rejection", passed, decision)
        print()

    def test_missing_required_fields(self):
        """Test 8: Deny on missing required fields."""
        print("Test 8: Missing Required Fields")
        print("-" * 70)

        groq_response = {
            "success": True,
            "content": """{
                "approved": true,
                "confidence": 0.80
            }""",
            "model_name": "test-model",
            "latency_ms": 140,
        }

        candidate_data = {
            "direction": "BUY",
            "entry_hint": 50000.0,
            "stop_hint": 49500.0,
        }

        decision = self.parser.parse(
            groq_response=groq_response,
            candidate_data=candidate_data,
            prompt_version="v1.0.0",
        )

        passed = (
            decision.approved is False
            and decision.valid_response is False
            and "Missing required fields" in decision.reason
        )

        self._record_result("Missing Required Fields", passed, decision)
        print()

    def test_prompt_builder(self):
        """Test 9: Prompt builder generates valid prompts."""
        print("Test 9: Prompt Builder")
        print("-" * 70)

        prompt_data = GroqPromptBuilder.build_decision_prompt(
            symbol="BTCUSDT",
            timeframe="15m",
            direction="BUY",
            score=0.87,
            entry_hint=50000.0,
            stop_hint=49500.0,
            target_hint=51500.0,
            reasons=["Bullish divergence", "Support bounce"],
            features={"rsi": 45.5, "volume_ratio": 1.8},
            bar_time="2024-01-15 10:30:00",
        )

        passed = (
            "system" in prompt_data
            and "user" in prompt_data
            and "prompt_version" in prompt_data
            and "BTCUSDT" in prompt_data["user"]
            and "BUY" in prompt_data["user"]
            and "JSON" in prompt_data["system"]
        )

        if passed:
            print("✅ PASS")
            print(f"System prompt length: {len(prompt_data['system'])} chars")
            print(f"User prompt length: {len(prompt_data['user'])} chars")
            print(f"Prompt version: {prompt_data['prompt_version']}")
        else:
            print("❌ FAIL")
            print(f"Generated prompt: {prompt_data}")

        self.results.append(("Prompt Builder", passed))
        print()

    def test_schema_validation(self):
        """Test 10: AIDecision schema validation."""
        print("Test 10: AIDecision Schema Validation")
        print("-" * 70)

        try:
            # Create valid AIDecision
            decision = AIDecision(
                candidate_id="test_cand_01",
                decision="BUY",
                approved=True,
                confidence=0.85,
                entry_min=50000.0,
                entry_max=50000.0,
                stop_loss=49500.0,
                setup_quality=0.7,
                trend_alignment=0.6,
                regime_fit=0.5,
                exhaustion_risk=0.2,
                reason="Test decision",
                model_name="test-model",
                prompt_version="v1.0.0",
                latency_ms=200,
                valid_response=True,
            )

            # Validate using Pydantic
            validated = AIDecision.model_validate(decision.model_dump())

            passed = (
                validated.approved is True
                and validated.confidence == 0.85
                and validated.decision == "BUY"
            )

            if passed:
                print("✅ PASS")
                print(f"Schema validation successful")
                print(f"Decision: {validated.decision}")
                print(f"Confidence: {validated.confidence}")
                print(f"Approved: {validated.approved}")
            else:
                print("❌ FAIL")
                print(f"Validation mismatch")

            self.results.append(("Schema Validation", passed))

        except Exception as e:
            print("❌ FAIL")
            print(f"Schema validation error: {e}")
            self.results.append(("Schema Validation", False))

        print()

    def test_live_groq_call(self):
        """Test 11: Live Groq API call (if configured)."""
        print("Test 11: Live Groq API Call")
        print("-" * 70)

        if not self.client or not self.client.is_configured():
            print("⏭️  SKIP - Groq API key not configured")
            print("   Set GROQ_API_KEY environment variable to enable live test")
            self.results.append(("Live Groq Call", None))
            print()
            return

        try:
            # Build prompt
            prompt_data = GroqPromptBuilder.build_decision_prompt(
                symbol="BTCUSDT",
                timeframe="1h",
                direction="BUY",
                score=0.82,
                entry_hint=45000.0,
                stop_hint=44500.0,
                target_hint=46000.0,
                reasons=["Strong support level", "RSI oversold"],
                features={"rsi": 32.5, "macd": 0.015},
            )

            print("Calling Groq API...")
            # Execute live API call
            response = self.client.chat_completion(
                system_prompt=prompt_data["system"],
                user_prompt=prompt_data["user"],
            )

            print(f"API Response received (latency: {response['latency_ms']}ms)")

            # Parse response
            candidate_data = {
                "direction": "BUY",
                "entry_hint": 45000.0,
                "stop_hint": 44500.0,
                "target_hint": 46000.0,
            }

            decision = self.parser.parse(
                groq_response=response,
                candidate_data=candidate_data,
                prompt_version=prompt_data["prompt_version"],
            )

            passed = (
                response["success"] is True
                and decision is not None
                and hasattr(decision, "approved")
                and hasattr(decision, "confidence")
            )

            if passed:
                print("✅ PASS")
                print(f"Decision: {decision.decision}")
                print(f"Approved: {decision.approved}")
                print(f"Confidence: {decision.confidence:.2f}")
                print(f"Reason: {decision.reason[:100]}...")
                print(f"Model: {decision.model_name}")
                print(f"Latency: {decision.latency_ms}ms")
            else:
                print("❌ FAIL")
                print(f"Response: {response}")

            self.results.append(("Live Groq Call", passed))

        except Exception as e:
            print("❌ FAIL")
            print(f"Live API call error: {e}")
            self.results.append(("Live Groq Call", False))

        print()

    def _record_result(self, test_name: str, passed: bool, decision: AIDecision):
        """Record test result and print details."""
        if passed:
            print("✅ PASS")
        else:
            print("❌ FAIL")

        print(f"Approved: {decision.approved}")
        print(f"Valid Response: {decision.valid_response}")
        print(f"Confidence: {decision.confidence}")
        print(f"Decision: {decision.decision}")
        print(f"Reason: {decision.reason}")

        self.results.append((test_name, passed))

    def print_summary(self):
        """Print test summary."""
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed_count = sum(1 for _, result in self.results if result is True)
        failed_count = sum(1 for _, result in self.results if result is False)
        skipped_count = sum(1 for _, result in self.results if result is None)
        total_count = len(self.results)

        for test_name, result in self.results:
            if result is True:
                status = "✅ PASS"
            elif result is False:
                status = "❌ FAIL"
            else:
                status = "⏭️  SKIP"
            print(f"{status:12} {test_name}")

        print("-" * 70)
        print(f"Total Tests: {total_count}")
        print(f"Passed:      {passed_count}")
        print(f"Failed:      {failed_count}")
        print(f"Skipped:     {skipped_count}")
        print("=" * 70)

        if failed_count == 0 and passed_count > 0:
            print("🎉 ALL TESTS PASSED!")
        elif failed_count > 0:
            print("⚠️  SOME TESTS FAILED - Review output above")
        else:
            print("⚠️  NO TESTS RAN")

        print()


def main():
    """Main entry point for smoke test."""
    test_suite = GroqSmokeTest()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
