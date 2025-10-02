"""
Configuration Settings Module

This module loads and validates configuration from environment variables.
"""

import os
import logging
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigurationError(Exception):
    """Exception raised for configuration errors"""
    pass


class Settings:
    """Application settings loaded from environment variables"""

    # Dify Configuration
    DIFY_API_KEY: str
    DIFY_API_URL: str
    DIFY_TIMEOUT: int

    # LangChain / LLM Configuration
    OPENAI_API_KEY: Optional[str]
    LLM_PROVIDER: str
    LLM_MODEL_NAME: str
    LLM_TEMPERATURE: float
    LLM_MAX_TOKENS: Optional[int]

    # Application Configuration
    APP_ENV: str
    APP_HOST: str
    APP_PORT: int
    DEBUG: bool

    # Logging Configuration
    LOG_LEVEL: str
    LOG_FORMAT: str

    # Memory Configuration
    MEMORY_TYPE: str
    MEMORY_WINDOW_SIZE: int

    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED: bool
    RATE_LIMIT_REQUESTS: int
    RATE_LIMIT_PERIOD: int

    def __init__(self):
        """Initialize settings from environment variables"""
        self._load_settings()
        self._validate_required_settings()
        self._configure_logging()

    def _load_settings(self):
        """Load all settings from environment variables"""
        # Dify Configuration
        self.DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")
        self.DIFY_API_URL = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1")
        self.DIFY_TIMEOUT = int(os.getenv("DIFY_TIMEOUT", "30"))

        # LangChain / LLM Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

        max_tokens = os.getenv("LLM_MAX_TOKENS", "")
        self.LLM_MAX_TOKENS = int(max_tokens) if max_tokens else None

        # Application Configuration
        self.APP_ENV = os.getenv("APP_ENV", "development")
        self.APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
        self.APP_PORT = int(os.getenv("APP_PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        self.LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Memory Configuration
        self.MEMORY_TYPE = os.getenv("MEMORY_TYPE", "buffer")
        self.MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))

        # Rate Limiting Configuration
        self.RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() in ("true", "1", "yes")
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

    def _validate_required_settings(self):
        """
        Validate that required settings are present

        Raises:
            ConfigurationError: If required settings are missing
        """
        required_settings = []

        # Dify API key is required
        if not self.DIFY_API_KEY:
            required_settings.append("DIFY_API_KEY")

        # OpenAI API key is required if using OpenAI provider
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            required_settings.append("OPENAI_API_KEY (required for OpenAI provider)")

        if required_settings:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(required_settings)}\n"
                f"Please set these variables in your .env file or environment."
            )

    def _configure_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format=self.LOG_FORMAT
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key

        Args:
            key: Setting key name
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        return getattr(self, key, default)

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.APP_ENV.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.APP_ENV.lower() == "development"

    def to_dict(self) -> dict:
        """
        Convert settings to dictionary (excludes sensitive data)

        Returns:
            Dictionary of non-sensitive settings
        """
        return {
            "DIFY_API_URL": self.DIFY_API_URL,
            "DIFY_TIMEOUT": self.DIFY_TIMEOUT,
            "LLM_PROVIDER": self.LLM_PROVIDER,
            "LLM_MODEL_NAME": self.LLM_MODEL_NAME,
            "LLM_TEMPERATURE": self.LLM_TEMPERATURE,
            "LLM_MAX_TOKENS": self.LLM_MAX_TOKENS,
            "APP_ENV": self.APP_ENV,
            "APP_HOST": self.APP_HOST,
            "APP_PORT": self.APP_PORT,
            "DEBUG": self.DEBUG,
            "LOG_LEVEL": self.LOG_LEVEL,
            "MEMORY_TYPE": self.MEMORY_TYPE,
            "MEMORY_WINDOW_SIZE": self.MEMORY_WINDOW_SIZE,
            "RATE_LIMIT_ENABLED": self.RATE_LIMIT_ENABLED,
            "RATE_LIMIT_REQUESTS": self.RATE_LIMIT_REQUESTS,
            "RATE_LIMIT_PERIOD": self.RATE_LIMIT_PERIOD,
        }

    def __repr__(self) -> str:
        """String representation of settings (excludes sensitive data)"""
        return f"Settings({self.to_dict()})"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern)

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables

    Returns:
        New Settings instance
    """
    global _settings
    load_dotenv(override=True)
    _settings = Settings()
    return _settings


# Example usage
if __name__ == "__main__":
    try:
        settings = get_settings()
        print("Configuration loaded successfully!")
        print(f"\nEnvironment: {settings.APP_ENV}")
        print(f"Dify API URL: {settings.DIFY_API_URL}")
        print(f"LLM Provider: {settings.LLM_PROVIDER}")
        print(f"LLM Model: {settings.LLM_MODEL_NAME}")
        print(f"\nAll settings (non-sensitive):")
        for key, value in settings.to_dict().items():
            print(f"  {key}: {value}")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
