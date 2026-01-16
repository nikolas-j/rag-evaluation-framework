"""Configuration management endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

router = APIRouter()


class ConfigUpdateRequest(BaseModel):
    """Request model for config updates."""
    overrides: Dict[str, Any]


class ConfigResponse(BaseModel):
    """Response model for config data."""
    config: Dict[str, Any]
    source: str = "default"


@router.get("", response_model=ConfigResponse)
async def get_config(request: Request):
    """Get current default configuration.
    
    Returns the configuration loaded from .env file
    with Field defaults for any values not specified.
    """
    config_manager = request.app.state.config_manager
    config_dict = config_manager.get_config_dict()
    
    return ConfigResponse(
        config=config_dict,
        source=".env + defaults"
    )


@router.post("/validate")
async def validate_config(
    request: Request,
    config_update: ConfigUpdateRequest
):
    """Validate configuration overrides without applying them.
    
    Useful for UI validation before submitting changes.
    """
    config_manager = request.app.state.config_manager
    errors = config_manager.validate_overrides(config_update.overrides)
    
    if errors:
        return {
            "valid": False,
            "errors": errors
        }
    
    return {
        "valid": True,
        "errors": {}
    }


@router.post("/save")
async def save_config(
    request: Request,
    config_update: ConfigUpdateRequest
):
    """Save configuration changes to .env file.
    
    Validates the config and persists it to .env for future use.
    """
    config_manager = request.app.state.config_manager
    
    try:
        config_manager.save_to_env(config_update.overrides)
        return {
            "success": True,
            "message": "Configuration saved to .env file"
        }
    except ValidationError as e:
        errors = {}
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            errors[field] = error["msg"]
        raise HTTPException(
            status_code=400,
            detail={"message": "Validation failed", "errors": errors}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save configuration: {str(e)}"
        )


@router.get("/schema")
async def get_config_schema():
    """Get the configuration schema for UI form generation.
    
    Returns field names, types, defaults, and descriptions.
    """
    from rag_app.config import Settings
    
    schema = Settings.model_json_schema()
    
    # Extract useful info for UI
    fields = {}
    for field_name, field_info in schema.get("properties", {}).items():
        fields[field_name] = {
            "type": field_info.get("type"),
            "default": field_info.get("default"),
            "description": field_info.get("description", ""),
            "title": field_info.get("title", field_name),
        }
        
        # Add enum options if present
        if "enum" in field_info:
            fields[field_name]["options"] = field_info["enum"]
    
    return {
        "fields": fields,
        "required": schema.get("required", [])
    }
