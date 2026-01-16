"""Configuration manager for runtime config overrides.

Handles loading config from .env and applying per-request overrides.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv, set_key
from pydantic import ValidationError

from rag_app.config import Settings

# Ensure .env is loaded from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env", override=True)


class ConfigManager:
    """Manages configuration with runtime overrides.
    
    Loads config from .env file + Settings defaults, allows temporary overrides
    for per-request customization.
    """
    
    def __init__(self):
        self._default_config: Optional[Settings] = None
    
    def load_default_config(self) -> Settings:
        """Load configuration from .env file and Settings defaults.
        
        Returns:
            Settings object with .env overrides applied to defaults
        """
        # Create Settings - pydantic-settings will load from .env automatically
        self._default_config = Settings()
        return self._default_config
    
    def get_default_config(self) -> Settings:
        """Get the default configuration.
        
        Returns:
            Default Settings object
        """
        if self._default_config is None:
            return self.load_default_config()
        return self._default_config
    
    def get_config_with_overrides(self, overrides: Dict[str, Any]) -> Settings:
        """Create a Settings object with runtime overrides applied.
        
        This allows per-request config customization without modifying .env.
        
        Args:
            overrides: Dictionary of config keys to override
            
        Returns:
            Settings object with overrides applied
            
        Raises:
            ValidationError: If overrides contain invalid values
        """
        # Get base config as dict
        base_config = self.get_default_config().model_dump()
        
        # Apply overrides
        base_config.update(overrides)
        
        # Create new Settings instance with merged values
        return Settings(**base_config)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the configuration as a dictionary.
        
        Returns:
            Dictionary of current config values
        """
        if self._default_config is None:
            self.load_default_config()
        
        return self._default_config.model_dump()
    
    def validate_overrides(self, overrides: Dict[str, Any]) -> Dict[str, str]:
        """Validate config overrides without applying them.
        
        Args:
            overrides: Dictionary of config keys to validate
            
        Returns:
            Dictionary of validation errors (empty if valid)
        """
        try:
            self.get_config_with_overrides(overrides)
            return {}
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                errors[field] = error["msg"]
            return errors
    
    def save_to_env(self, overrides: Dict[str, Any]) -> None:
        """Save configuration changes to .env file.
        
        Args:
            overrides: Dictionary of config keys to save
            
        Raises:
            ValidationError: If overrides contain invalid values
        """
        # Fields that should NOT be saved to .env (internal/computed fields)
        EXCLUDED_FIELDS = {
            'config',           # Recursive config field (from extra="allow")
            'source',           # Internal metadata field
            'timestamp_utc',    # Computed timestamp field
            'prompt_version',   # Runtime version info
        }
        
        # Validate first
        validated_settings = self.get_config_with_overrides(overrides)
        
        # Get path to .env file
        env_path = project_root / ".env"
        
        # Update .env file
        for key, value in overrides.items():
            # Skip excluded fields
            if key in EXCLUDED_FIELDS:
                continue
            
            # Convert to uppercase for .env format
            env_key = key.upper()
            
            # Handle special types
            if isinstance(value, dict):
                env_value = json.dumps(value)
            elif isinstance(value, bool):
                env_value = "true" if value else "false"
            else:
                env_value = str(value)
            
            # Write to .env file with error checking
            success, _, _ = set_key(
                str(env_path), 
                env_key, 
                env_value,
                quote_mode='never',  # Don't auto-quote - we control the format
                encoding='utf-8'
            )
            
            if not success:
                raise IOError(f"Failed to write {env_key}={env_value} to .env file at {env_path}")
        
        # Reload config to pick up changes
        load_dotenv(env_path, override=True)
        self._default_config = Settings()
