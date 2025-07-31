import os
from typing import Optional
import uuid
import base64
from utils.config import settings

def save_base64_image(base64_str: str) -> Optional[str]:
    """Save base64 image to file and return the file path"""
    try:
        if not base64_str:
            return None
            
        # Extract the base64 binary data
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        # Generate unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(settings.IMAGE_UPLOAD_DIR, filename)
        
        # Decode and save
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(base64_str))
            
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None