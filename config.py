"""
Configuration settings for AI Detector
"""

# Model configurations - Multiple models for better ensemble accuracy
TEXT_MODELS = [
    {
        "name": "Hello-SimpleAI/chatgpt-detector-roberta", 
        "weight": 1.0,
        "fallback": None
    },
    {
        "name": "roberta-base-openai-detector",
        "weight": 0.8,
        "fallback": "Hello-SimpleAI/chatgpt-detector-roberta"
    }
]

IMAGE_MODELS = [
    {
        "name": "Organika/sdxl-detector",
        "weight": 2.0,  # Most accurate for AI detection - 100% CONFIDENCE = ABSOLUTE OVERRIDE
        "fallback": "umm-maybe/AI-image-detector",
        "absolute_override": True  # When this model hits 1.0, bypass everything
    },
    {
        "name": "microsoft/resnet-50",
        "weight": 1.8,  # Strong general classifier - better human detection
        "fallback": "umm-maybe/AI-image-detector"
    },
    {
        "name": "umm-maybe/AI-image-detector",
        "weight": 1.2,  # Baseline model
        "fallback": None
    },
    {
        "name": "saltacc/anime-ai-detect",
        "weight": 1.0,  # Specialized for anime/art
        "fallback": "umm-maybe/AI-image-detector"
    }
]

# Detection thresholds for 5-tier system
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to show results

# File processing
MAX_TEXT_LENGTH = 1000
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
SUPPORTED_TEXT_FORMATS = ('.txt',)
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

# Heuristic configurations
HEURISTICS = {
    "text_features": {
        "short_text_penalty": 0.85,  # Confidence multiplier for text < 50 chars
        "uniform_sentences_boost": 1.15,  # Boost when sentences are very uniform
        "uniform_threshold": 10  # Std dev threshold for sentence length uniformity
    },
    "image_features": {
        "small_image_penalty": 0.85,  # Confidence multiplier for images < 256px
        "min_size_threshold": 256  # Minimum width/height for full confidence
    },
    "filename_semantic": {
        "ai_keywords": [
            'chatgpt', 'gpt', 'dalle', 'midjourney', 'stable diffusion', 'stablediffusion',
            'ai generated', 'ai_generated', 'artificial', 'generated', 'synthetic',
            'deepfake', 'gan', 'diffusion', 'neural', 'model',
            'openai', 'anthropic', 'claude', 'bard', 'gemini',
            'leonardo', 'firefly', 'kandinsky', 'playground',
            'aiart', 'ai_art', 'machinelearning', 'ml_generated',
            'copilot', 'runway', 'pika', 'suno', 'luma', 'kling',
            'flux', 'imagen', 'sd', 'comfyui', 'automatic1111',
            'civitai', 'huggingface', 'replicate'
        ],
        "ai_boost_factor": 2.2  # Confidence boost when AI keywords found
    },
    "ensemble": {
        "disagreement_threshold": 0.3,
        "max_disagreement_penalty": 0.25,
        "high_confidence_threshold": 0.8,
        # Organika 100% Override System
        "organika_override": {
            "enabled": True,
            "absolute_confidence_threshold": 1.0,  # ONLY 100% exactly - not 99.99%
            "bypass_all_processing": True,  # Skip ensemble, adjustments, everything
            "override_message": "🎯 ORGANIKA ABSOLUTE OVERRIDE"
        },
        # Dynamic weighting system
        "dynamic_weighting": {
            "enabled": True,
            "organika_ai_boost": 1.4,      # Boost Organika when other models also say AI
            "organika_human_penalty": 0.6,  # Reduce Organika when others say human
            "consensus_threshold": 0.6,     # Agreement threshold for dynamic adjustment
            "outlier_penalty": 0.7          # Penalty for models that disagree with consensus
        }
    }
}

# Error messages
ERROR_MESSAGES = {
    "no_file": "Please select a file to analyze.",
    "unsupported_format": "Unsupported file type. Please upload an image (.jpg, .png), text file (.txt), or video (.mp4, .mov, .avi).",
    "processing_error": "An error occurred while processing your file. Please try again.",
    "model_unavailable": "AI detection models are currently unavailable. Please try again later.",
    "video_not_implemented": "Video detection is coming soon! We're working on adding this feature."
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
    },
    "video_not_implemented": {
        "message": "🎬 Video Detection Coming Soon",
        "class": "confidence-tier-3",
        "icon": "🎬",
        "description": "Video analysis is in development. We're working on this feature!",
        "footer": "Try images or text for now.",
        "color": "🟦"
    }
}