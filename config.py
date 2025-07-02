
"""
Configuration settings for AI Detector
"""

# Model configurations
TEXT_MODELS = [
    {
        "name": "roberta-base-openai-detector",
        "weight": 0.6,
        "fallback": "Hello-SimpleAI/chatgpt-detector-roberta"
    },
    {
        "name": "Hello-SimpleAI/chatgpt-detector-roberta", 
        "weight": 0.4,
        "fallback": None
    }
]

IMAGE_MODELS = [
    {
        "name": "umm-maybe/AI-image-detector",
        "weight": 0.7,
        "fallback": None
    }
]

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to show results
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
LOW_CONFIDENCE_THRESHOLD = 0.25

# File processing
MAX_TEXT_LENGTH = 1000
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
SUPPORTED_TEXT_FORMATS = ('.txt',)
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

# UI Messages
RESULT_MESSAGES = {
    "very_high_ai": {
        "message": "AI-Generated",
        "class": "ai-high", 
        "icon": "🤖",
        "description": "Very high confidence - almost certainly AI-created"
    },
    "high_ai": {
        "message": "Likely AI-Generated", 
        "class": "ai-high",
        "icon": "🤖", 
        "description": "Strong indicators of artificial intelligence"
    },
    "medium": {
        "message": "Uncertain Origin",
        "class": "ai-medium",
        "icon": "🧐",
        "description": "Mixed signals - requires manual review"
    },
    "low_human": {
        "message": "Likely Human-Created",
        "class": "ai-low", 
        "icon": "🧠",
        "description": "Strong indicators of human authorship"
    },
    "very_low_human": {
        "message": "Human-Created",
        "class": "ai-low",
        "icon": "🧠", 
        "description": "Very high confidence - almost certainly human-made"
    },
    "insufficient": {
        "message": "Insufficient Data",
        "class": "ai-medium",
        "icon": "❓",
        "description": "Unable to determine with sufficient confidence"
    }
}

# Error messages
ERROR_MESSAGES = {
    "no_file": "⚠️ Please select a file to analyze.",
    "unsupported_format": "🚫 Unsupported file type. Please upload an image, text file, or video.",
    "model_unavailable": "🔧 Detection model is currently unavailable. Please try again later.",
    "processing_error": "❌ Error processing file. Please check the file format and try again.",
    "file_too_large": "📏 File is too large. Please use a smaller file.",
    "corrupted_file": "💾 File appears to be corrupted or unreadable."
}
