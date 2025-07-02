
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

# Detection thresholds for 5-tier system
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to show results

# File processing
MAX_TEXT_LENGTH = 1000
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
SUPPORTED_TEXT_FORMATS = ('.txt',)
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

# Error messages
ERROR_MESSAGES = {
    "no_file": "Please select a file to analyze.",
    "unsupported_format": "Unsupported file type. Please upload an image (.jpg, .png), text file (.txt), or video (.mp4, .mov, .avi).",
    "processing_error": "An error occurred while processing your file. Please try again.",
    "model_unavailable": "AI detection models are currently unavailable. Please try again later."
}

# UI Messages - New 5-tier system
RESULT_MESSAGES = {
    "almost_certainly_human": {
        "message": "✅ Almost Certainly Human",
        "class": "confidence-tier-1", 
        "icon": "✅",
        "description": "This looks and feels naturally made with little to no signs of AI.",
        "footer": "Very low chance it was AI-generated.",
        "color": "🟩"
    },
    "likely_human": {
        "message": "🧠 Likely Human",
        "class": "confidence-tier-2",
        "icon": "🧠", 
        "description": "Some things raised flags, but it mostly seems human-created.",
        "footer": "Small chance of AI involvement.",
        "color": "🟨"
    },
    "unsure": {
        "message": "🤔 Unsure – Needs a Closer Look",
        "class": "confidence-tier-3",
        "icon": "🤔",
        "description": "Hard to tell — it shares traits with both AI and human-made content.",
        "footer": "Too close to call confidently.",
        "color": "🟧"
    },
    "possibly_ai": {
        "message": "⚠️ Possibly AI-Generated",
        "class": "confidence-tier-4", 
        "icon": "⚠️",
        "description": "There's a noticeable pattern that matches how AI typically creates content.",
        "footer": "Leaning AI, but not 100% sure.",
        "color": "🟥"
    },
    "likely_ai": {
        "message": "🤖 Likely AI-Generated",
        "class": "confidence-tier-5",
        "icon": "🤖", 
        "description": "Strong signals that this was created by AI tools or models.",
        "footer": "High chance of being AI-made.",
        "color": "🔴"
    },
    "insufficient": {
        "message": "❓ Undetermined",
        "class": "confidence-tier-3",
        "icon": "❓",
        "description": "Unable to determine with sufficient confidence - more data needed.",
        "footer": "Analysis inconclusive.",
        "color": "🟧"
    }
}
