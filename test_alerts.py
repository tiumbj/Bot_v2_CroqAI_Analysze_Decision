#!/usr/bin/env python3
"""
Test script for the alert system
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.alert_system import alert_system

def test_alert_system():
    """Test all alert types"""
    print("Testing Alert System...")
    print("=" * 50)
    
    # Test candidate detection alert
    print("\n1. Testing Candidate Detection Alert:")
    alert_system.candidate_detected("GOLD", "BUY", 0.85)
    
    # Test AI approval alert
    print("\n2. Testing AI Approval Alert:")
    alert_system.ai_approved("GOLD", "BUY", 0.78)
    
    # Test AI rejection alert
    print("\n3. Testing AI Rejection Alert:")
    alert_system.ai_rejected("GOLD", "BUY", "Risk gates blocked")
    
    # Test order execution alert
    print("\n4. Testing Order Execution Alert:")
    alert_system.order_executed("GOLD", "BUY", 123456, "SUCCESS")
    
    # Test failed execution alert
    print("\n5. Testing Failed Execution Alert:")
    alert_system.order_executed("GOLD", "BUY", None, "FAILED: Connection error")
    
    print("\n" + "=" * 50)
    print("Alert System Test Completed!")

if __name__ == "__main__":
    test_alert_system()