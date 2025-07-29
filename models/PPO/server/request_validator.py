from typing import Dict, Any, Tuple
from flask import Request

def validate_optimize_request(request: Request) -> Tuple[Dict[str, Any], str, int]:
    """Validate optimization request and return (data, error_message, status_code)"""
    
    if not request.is_json:
        return {}, "Request must be JSON", 400
        
    data = request.get_json()
    
    if not isinstance(data, dict):
        return {}, "Invalid JSON format", 400
        
    if "prompt" not in data:
        return {}, "Missing 'prompt' field", 400
        
    if not data["prompt"] or not isinstance(data["prompt"], str):
        return {}, "Invalid prompt value", 400
        
    return data, "", 200