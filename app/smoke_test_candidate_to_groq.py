"""
Candidate to Groq Integration Smoke Test
Version: 1.0.0
Purpose: Validate end-to-end flow from raw candidates to AIDecision via Groq
"""

import os
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.candidate_scan_finalize import finalize_candidate_scan
from core.groq_prompt_builder import GroqPromptBuilder
from core.groq_client import GroqClient
from core.groq_response_parser import GroqResponseParser


def build_sample_candidates():
    """Build sample raw candidate batch for testing."""
    return [
        {
            "symbol": "GOLD",
            "timeframe": "M15",
            "bar_time": "2026-03-10 14:30:00",
            "decision": "BUY",
            "score": 0.86,
            "entry": 5186.91,
            "sl": 5158.58,
            "tp": 5243.57,
            "guard": "allowed",
        },
        {
            "symbol": "XAUUSDm",
            "timeframe": "M15",
            "bar_time": "2026-03-10 14:30:00",
            "decision": "BUY",
            "score": 0.86,
            "entry": 5186.91,
            "sl": 5158.58,
            "tp": 5243.57,
            "guard": "allowed",
        },
        {
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "bar_time": "2026-03-10 14:45:00",
            "decision": "BUY",
            "score": 0.82,
            "entry": 5190.10,
            "sl": 5160.00,
            "tp": 5250.00,
            "guard": "allowed",
        },
    ]


def run_integration_test():
    """Execute candidate to Groq integration smoke test."""
    print("=" * 80)
    print("CANDIDATE TO GROQ INTEGRATION SMOKE TEST")
    print("=" * 80)
    print()

    print("Step 1: Building sample candidate batch...")
    raw_candidates = build_sample_candidates()
    print(f"✓ Created {len(raw_candidates)} raw candidates")
    for c in raw_candidates:
        print(f"  - {c['symbol']} {c['decision']} (score={c['score']:.2f})")
    print()

    print("Step 2: Finalizing candidate scan...")
    try:
        finalized = finalize_candidate_scan(
            raw_candidates=raw_candidates,
            timeframe="M15",
            processed_symbols=3,
        )
        approved = finalized.get("accepted", [])
        rejected = finalized.get("rejected", [])
        print(f"✓ Finalization complete")
        print(f"  - Approved: {len(approved)}")
        print(f"  - Rejected: {len(rejected)}")
    except Exception as e:
        print(f"✗ Finalization failed: {e}")
        return
    print()

    if not approved:
        print("⚠️  No approved candidates to process through Groq")
        print("=" * 80)
        return

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("⚠️  GROQ_API_KEY not configured - skipping live API calls")
        print("   Set GROQ_API_KEY environment variable to enable live testing")
        live_mode = False
    else:
        print("✓ GROQ_API_KEY configured - enabling live API calls")
        live_mode = True
    print()

    groq_client = GroqClient() if live_mode else None
    parser = GroqResponseParser()

    print("Step 3: Processing approved candidates through Groq...")
    print("-" * 80)

    results = []
    for idx, candidate in enumerate(approved, 1):
        candidate_id = (
            candidate.get("metadata", {}).get("candidate_id")
            or candidate.get("setup_key")
            or "unknown_candidate"
        )
        symbol = (
            candidate.get("display_symbol")
            or candidate.get("runtime_symbol")
            or candidate.get("canonical_symbol")
            or "unknown_symbol"
        )
        decision = candidate.get("decision", "")
        print(f"\nCandidate {idx}/{len(approved)}: {candidate_id}")
        print(f"  Symbol: {symbol}")
        print(f"  Direction: {decision}")
        print(f"  Score: {candidate['score']:.2f}")
        print(f"  Entry: {candidate['entry']}, Stop: {candidate['sl']}")

        prompt_data = GroqPromptBuilder.build_decision_prompt(
            symbol=symbol,
            timeframe=candidate["timeframe"],
            direction=decision,
            score=candidate["score"],
            entry_hint=candidate["entry"],
            stop_hint=candidate["sl"],
            target_hint=candidate.get("tp"),
            reasons=candidate.get("reasons", []),
            features=candidate.get("features", {}),
            bar_time=candidate.get("bar_time"),
        )
        print(f"  ✓ Prompt built (version: {prompt_data['prompt_version']})")

        if live_mode:
            try:
                print(f"  → Calling Groq API...")
                groq_response = groq_client.chat_completion(
                    system_prompt=prompt_data["system"],
                    user_prompt=prompt_data["user"],
                )
                print(f"  ✓ API call completed ({groq_response.get('latency_ms', 0)}ms)")
            except Exception as e:
                print(f"  ✗ API call failed: {e}")
                groq_response = {
                    "success": False,
                    "error": str(e),
                    "model_name": "unknown",
                    "latency_ms": 0,
                }
        else:
            print(f"  ⏭️  Skipping live API call (no API key)")
            groq_response = {
                "success": False,
                "error": "API key not configured",
                "model_name": "skipped",
                "latency_ms": 0,
            }

        print(f"  → Parsing response into AIDecision...")
        parser_candidate_data = dict(candidate)
        parser_candidate_data["candidate_id"] = candidate_id
        parser_candidate_data["direction"] = decision
        parser_candidate_data["entry_hint"] = candidate.get("entry")
        parser_candidate_data["stop_hint"] = candidate.get("sl")
        parser_candidate_data["target_hint"] = candidate.get("tp")
        ai_decision = parser.parse(
            groq_response=groq_response,
            candidate_data=parser_candidate_data,
            prompt_version=prompt_data["prompt_version"],
        )

        print(f"  ✓ AIDecision created")
        print(f"    - Candidate ID: {ai_decision.candidate_id}")
        print(f"    - Decision: {ai_decision.decision}")
        print(f"    - Approved: {ai_decision.approved}")
        print(f"    - Confidence: {ai_decision.confidence:.2f}")
        print(f"    - Valid Response: {ai_decision.valid_response}")
        print(f"    - Reason: {ai_decision.reason}")
        print(f"    - Model: {ai_decision.model_name}")
        print(f"    - Latency: {ai_decision.latency_ms}ms")

        results.append({
            "candidate_id": candidate_id,
            "symbol": symbol,
            "direction": decision,
            "ai_decision": ai_decision,
        })

    print()
    print("-" * 80)
    print()

    print("Step 4: Summary")
    print("-" * 80)
    print(f"Total raw candidates:      {len(raw_candidates)}")
    print(f"Approved by finalization:  {len(approved)}")
    print(f"Rejected by finalization:  {len(rejected)}")
    print(f"Processed through Groq:    {len(results)}")
    print()

    if results:
        print("AIDecision Results:")
        for r in results:
            ai_dec = r["ai_decision"]
            status = "✓ APPROVED" if ai_dec.approved else "✗ DENIED"
            valid = "VALID" if ai_dec.valid_response else "INVALID"
            print(f"  {status:12} {r['symbol']:10} {r['direction']:4} "
                  f"conf={ai_dec.confidence:.2f} {valid:7} {r['candidate_id']}")

    print()
    print("=" * 80)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 80)


def main():
    """Main entry point."""
    try:
        run_integration_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
