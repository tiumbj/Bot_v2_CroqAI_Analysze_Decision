"""
Groq API Client
Version: 1.0.0
Purpose: Execute Groq chat completion requests with proper error handling and telemetry
"""

import time
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class GroqClientError(Exception):
    """Base exception for Groq client errors."""
    pass


class GroqClient:
    """
    Groq API client for chat completion requests.
    Handles configuration loading, API calls, timeout, and telemetry.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Groq client with configuration.

        Args:
            config_path: Path to ai.yaml config file. If None, uses default location.
        """
        self.config = self._load_config(config_path)
        self.api_key = self._get_api_key()
        self.model_name = self.config.get("groq", {}).get("model", "llama-3.3-70b-versatile")
        self.temperature = self.config.get("groq", {}).get("temperature", 0.2)
        self.max_tokens = self.config.get("groq", {}).get("max_tokens", 1024)
        self.timeout = self.config.get("groq", {}).get("timeout_seconds", 10)

        # Initialize Groq SDK client if API key is available
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                raise GroqClientError(
                    "Groq SDK not installed. Install with: pip install groq"
                )

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from ai.yaml."""
        if config_path is None:
            # Default to config/ai.yaml relative to repo root
            repo_root = Path(__file__).parent.parent
            config_path = repo_root / "config" / "ai.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise GroqClientError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def _get_api_key(self) -> Optional[str]:
        """
        Get Groq API key from config or environment.
        Returns None if not configured (allows testing without live API).
        """
        import os

        # Try config first
        api_key = self.config.get("groq", {}).get("api_key")

        # Fall back to environment variable
        if not api_key or api_key == "${GROQ_API_KEY}":
            api_key = os.getenv("GROQ_API_KEY")

        return api_key

    def is_configured(self) -> bool:
        """Check if client is properly configured with API key."""
        return self.client is not None

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a chat completion request to Groq API.

        Args:
            system_prompt: System-level instruction prompt
            user_prompt: User-specific query/data prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Dict containing:
                - success: bool
                - content: str (model response text)
                - model_name: str
                - latency_ms: int
                - error: Optional[str]
                - finish_reason: Optional[str]

        Raises:
            GroqClientError: On configuration or critical errors
        """
        if not self.is_configured():
            raise GroqClientError("Groq client not configured. API key missing.")

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                timeout=self.timeout,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response content
            if not response.choices:
                return {
                    "success": False,
                    "content": "",
                    "model_name": self.model_name,
                    "latency_ms": latency_ms,
                    "error": "No choices returned from model",
                    "finish_reason": None,
                }

            choice = response.choices[0]
            content = choice.message.content if choice.message else ""
            finish_reason = choice.finish_reason

            return {
                "success": True,
                "content": content,
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "error": None,
                "finish_reason": finish_reason,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_msg = f"{type(e).__name__}: {str(e)}"

            return {
                "success": False,
                "content": "",
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "error": error_msg,
                "finish_reason": None,
            }

    def get_config_summary(self) -> Dict[str, Any]:
        """Return current configuration summary for debugging."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
            "is_configured": self.is_configured(),
        }