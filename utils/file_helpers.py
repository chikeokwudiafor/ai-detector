
import os
from config import SUPPORTED_IMAGE_FORMATS, SUPPORTED_TEXT_FORMATS, SUPPORTED_VIDEO_FORMATS, ERROR_MESSAGES

def validate_file_extension(filename):
    """Validate file extension and return file type"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith(SUPPORTED_IMAGE_FORMATS):
        return True, "image", None
    elif filename_lower.endswith(SUPPORTED_TEXT_FORMATS):
        return True, "text", None
    elif filename_lower.endswith(SUPPORTED_VIDEO_FORMATS):
        return False, None, "🎬 Video Detection Coming Soon\n\nVideo analysis is in development. We're working on this feature!\n\nTry images or text for now."
    else:
        return False, None, ERROR_MESSAGES["unsupported_format"]

def validate_file_size(file, max_size_mb=10):
    """Validate file size"""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size_bytes = max_size_mb * 1024 * 1024
    if size > max_size_bytes:
        return False, f"File too large. Maximum size is {max_size_mb}MB."
    
    return True, None

def get_safe_filename(filename):
    """Generate a safe filename for storage"""
    import re
    import uuid
    
    # Remove unsafe characters
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Add timestamp to avoid conflicts
    timestamp = str(int(datetime.now().timestamp()))
    name, ext = os.path.splitext(safe_name)
    
    return f"{name}_{timestamp}{ext}"
