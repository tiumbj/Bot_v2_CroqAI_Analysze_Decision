"""
Alert System for OracleBot-Pro
Provides visual and audible alerts for trade signals and executions
"""

import os
import time
import winsound
from datetime import datetime
from typing import Dict, Any, Optional


class AlertSystem:
    """Alert system for trade signals and executions"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
    
    def _can_alert(self) -> bool:
        """Check if alerting is allowed (cooldown and enabled)"""
        if not self.enabled:
            return False
        current_time = time.time()
        return (current_time - self.last_alert_time) >= self.alert_cooldown
    
    def candidate_detected(self, symbol: str, direction: str, score: float) -> None:
        """Alert for candidate detection"""
        if not self._can_alert():
            return
            
        message = f"CANDIDATE DETECTED: {symbol} {direction} Score: {score:.2f}"
        self._visual_alert(message, "CANDIDATE")
        self._sound_alert("candidate")
        self.last_alert_time = time.time()
        
    def ai_approved(self, symbol: str, direction: str, confidence: float) -> None:
        """Alert for AI approval"""
        if not self._can_alert():
            return
            
        message = f"AI APPROVED: {symbol} {direction} Confidence: {confidence:.2f}"
        self._visual_alert(message, "APPROVED")
        self._sound_alert("approved")
        self.last_alert_time = time.time()
        
    def ai_rejected(self, symbol: str, direction: str, reason: str) -> None:
        """Alert for AI rejection"""
        if not self._can_alert():
            return
            
        message = f"AI REJECTED: {symbol} {direction} Reason: {reason}"
        self._visual_alert(message, "REJECTED")
        self._sound_alert("rejected")
        self.last_alert_time = time.time()
        
    def order_executed(self, symbol: str, direction: str, ticket: Any, status: str) -> None:
        """Alert for order execution"""
        if not self._can_alert():
            return
            
        message = f"ORDER EXECUTED: {symbol} {direction} Ticket: {ticket} Status: {status}"
        self._visual_alert(message, "EXECUTION")
        self._sound_alert("execution")
        self.last_alert_time = time.time()
        
    def _visual_alert(self, message: str, alert_type: str) -> None:
        """Display visual alert"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on alert type
        colors = {
            "CANDIDATE": "\033[93m",  # Yellow
            "APPROVED": "\033[92m",   # Green
            "REJECTED": "\033[91m",   # Red
            "EXECUTION": "\033[96m"   # Cyan
        }
        
        color = colors.get(alert_type, "\033[0m")
        reset = "\033[0m"
        
        # Create a prominent visual alert
        border = "=" * 80
        print(f"\n{border}")
        print(f"{color}🚨 {alert_type} ALERT - {timestamp} 🚨{reset}")
        print(f"{color}{message}{reset}")
        print(f"{border}\n")
        
    def _sound_alert(self, alert_type: str) -> None:
        """Play sound alert"""
        try:
            if alert_type == "candidate":
                # Short beep for candidate detection
                winsound.Beep(1000, 200)
            elif alert_type == "approved":
                # Double beep for approval
                winsound.Beep(1200, 150)
                time.sleep(0.1)
                winsound.Beep(1200, 150)
            elif alert_type == "rejected":
                # Low beep for rejection
                winsound.Beep(800, 300)
            elif alert_type == "execution":
                # Triple beep for execution
                winsound.Beep(1500, 100)
                time.sleep(0.05)
                winsound.Beep(1500, 100)
                time.sleep(0.05)
                winsound.Beep(1500, 100)
        except Exception:
            # Fallback if winsound is not available
            pass


# Global alert system instance
alert_system = AlertSystem()


def enable_alerts() -> None:
    """Enable the alert system"""
    alert_system.enabled = True


def disable_alerts() -> None:
    """Disable the alert system"""
    alert_system.enabled = False


def get_alert_system() -> AlertSystem:
    """Get the global alert system instance"""
    return alert_system