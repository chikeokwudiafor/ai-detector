
"""
AI Detection Module with ensemble support
"""

from PIL import Image
import torch
from transformers import pipeline
import numpy as np
from config import *
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of AI detection models"""
    
    def __init__(self):
        self.text_models = []
        self.image_models = []
        self._load_models()
    
    def _load_models(self):
        """Load all configured models with fallbacks"""
        logger.info("Loading AI detection models...")
        
        # Load text models
        for model_config in TEXT_MODELS:
            model = self._load_text_model(model_config)
            if model:
                self.text_models.append({
                    'model': model,
                    'weight': model_config['weight'],
                    'name': model_config['name']
                })
        
        # Load image models  
        for model_config in IMAGE_MODELS:
            model = self._load_image_model(model_config)
            if model:
                self.image_models.append({
                    'model': model,
                    'weight': model_config['weight'],
                    'name': model_config['name']
                })
        
        logger.info(f"✓ Loaded {len(self.text_models)} text models and {len(self.image_models)} image models")
    
    def _load_text_model(self, model_config):
        """Load a text classification model with fallback"""
        try:
            model = pipeline("text-classification", model=model_config['name'])
            logger.info(f"✓ Text model loaded: {model_config['name']}")
            return model
        except Exception as e:
            logger.warning(f"✗ Failed to load {model_config['name']}: {e}")
            
            # Try fallback
            if model_config['fallback']:
                try:
                    model = pipeline("text-classification", model=model_config['fallback'])
                    logger.info(f"✓ Text model loaded (fallback): {model_config['fallback']}")
                    return model
                except Exception as e2:
                    logger.error(f"✗ Fallback also failed: {e2}")
            
            return None
    
    def _load_image_model(self, model_config):
        """Load an image classification model with fallback"""
        try:
            model = pipeline("image-classification", model=model_config['name'])
            logger.info(f"✓ Image model loaded: {model_config['name']}")
            return model
        except Exception as e:
            logger.warning(f"✗ Failed to load {model_config['name']}: {e}")
            return None

# Global model manager instance
model_manager = ModelManager()

class AIDetector:
    """Main AI detection class with ensemble support"""
    
    @staticmethod
    def detect_text(text_content):
        """
        Detect AI-generated text using ensemble of models
        Returns: (result_type, confidence, raw_scores)
        """
        if not model_manager.text_models:
            return "model_unavailable", 0.0, []
        
        try:
            # Truncate text if too long
            if len(text_content) > MAX_TEXT_LENGTH:
                text_content = text_content[:MAX_TEXT_LENGTH]
            
            # Get predictions from all models
            predictions = []
            weights = []
            
            for model_info in model_manager.text_models:
                try:
                    result = model_info['model'](text_content)
                    confidence = AIDetector._parse_text_result(result)
                    predictions.append(confidence)
                    weights.append(model_info['weight'])
                    logger.info(f"Model {model_info['name']}: {confidence:.3f}")
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
            
            if not predictions:
                return "processing_error", 0.0, []
            
            # Calculate weighted ensemble score
            ensemble_confidence = np.average(predictions, weights=weights)
            
            # Apply heuristic adjustments
            ensemble_confidence = AIDetector._apply_text_heuristics(
                text_content, ensemble_confidence, predictions
            )
            
            result_type = AIDetector._classify_confidence(ensemble_confidence)
            
            return result_type, ensemble_confidence, predictions
            
        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return "processing_error", 0.0, []
    
    @staticmethod
    def detect_image(image_file):
        """
        Detect AI-generated images using ensemble of models
        Returns: (result_type, confidence, raw_scores)
        """
        if not model_manager.image_models:
            return "model_unavailable", 0.0, []
        
        try:
            # Load and preprocess image
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get predictions from all models
            predictions = []
            weights = []
            
            for model_info in model_manager.image_models:
                try:
                    results = model_info['model'](image)
                    confidence = AIDetector._parse_image_result(results)
                    predictions.append(confidence)
                    weights.append(model_info['weight'])
                    logger.info(f"Model {model_info['name']}: {confidence:.3f}")
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
            
            if not predictions:
                return "processing_error", 0.0, []
            
            # Calculate weighted ensemble score
            ensemble_confidence = np.average(predictions, weights=weights)
            
            # Apply heuristic adjustments
            ensemble_confidence = AIDetector._apply_image_heuristics(
                image, ensemble_confidence, predictions
            )
            
            result_type = AIDetector._classify_confidence(ensemble_confidence)
            
            return result_type, ensemble_confidence, predictions
            
        except Exception as e:
            logger.error(f"Image detection error: {e}")
            return "processing_error", 0.0, []
    
    @staticmethod
    def detect_video(video_file):
        """
        Placeholder for video detection
        Returns: (result_type, confidence, raw_scores)
        """
        return "insufficient", 0.0, []
    
    @staticmethod
    def _parse_text_result(result):
        """Parse text classification result to AI confidence score"""
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        label = result['label'].lower()
        score = result['score']
        
        # Check for AI-indicating labels
        if any(keyword in label for keyword in ['fake', 'ai', 'generated', 'machine', 'label_1', '1']):
            return score
        elif any(keyword in label for keyword in ['real', 'human', 'authentic', 'label_0', '0']):
            return 1.0 - score
        else:
            # Default: assume higher score means AI
            return score
    
    @staticmethod
    def _parse_image_result(results):
        """Parse image classification result to AI confidence score"""
        ai_confidence = 0.0
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            # Check for AI-indicating labels
            if any(keyword in label for keyword in ['ai', 'artificial', 'generated', 'fake', 'synthetic']):
                ai_confidence = max(ai_confidence, score)
            elif any(keyword in label for keyword in ['real', 'human', 'natural', 'authentic']):
                ai_confidence = max(ai_confidence, 1.0 - score)
        
        # If no specific labels found, use first result
        if ai_confidence == 0.0 and results:
            first_result = results[0]
            ai_confidence = first_result['score'] if 'ai' in first_result['label'].lower() else 1.0 - first_result['score']
        
        return ai_confidence
    
    @staticmethod
    def _apply_text_heuristics(text_content, confidence, predictions):
        """Apply heuristic adjustments for text"""
        # If models disagree significantly, reduce confidence
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            if std_dev > 0.3:  # High disagreement
                confidence *= 0.8
        
        # Very short text is harder to classify
        if len(text_content) < 50:
            confidence *= 0.9
        
        # Perfect grammar/no typos might indicate AI
        if len(text_content) > 100:
            # Simple heuristic: check for variation in sentence length
            sentences = text_content.split('.')
            if len(sentences) > 3:
                lengths = [len(s.strip()) for s in sentences if s.strip()]
                if lengths and np.std(lengths) < 10:  # Very uniform sentence lengths
                    confidence *= 1.1
        
        return min(confidence, 1.0)
    
    @staticmethod
    def _apply_image_heuristics(image, confidence, predictions):
        """Apply heuristic adjustments for images"""
        # If models disagree, reduce confidence
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            if std_dev > 0.3:
                confidence *= 0.8
        
        # Very small images are harder to classify
        width, height = image.size
        if width < 256 or height < 256:
            confidence *= 0.9
        
        return min(confidence, 1.0)
    
    @staticmethod
    def _classify_confidence(confidence):
        """Classify confidence into result categories"""
        # Check if confidence meets minimum threshold
        if confidence < CONFIDENCE_THRESHOLD and confidence > (1.0 - CONFIDENCE_THRESHOLD):
            return "insufficient"
        
        # Classify based on thresholds
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return "very_high_ai"
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return "high_ai"
        elif confidence >= LOW_CONFIDENCE_THRESHOLD:
            return "medium"
        elif confidence >= (1.0 - LOW_CONFIDENCE_THRESHOLD):
            return "low_human"
        else:
            return "very_low_human"

def get_result_classification(result_type):
    """
    Get display information for a result type
    Returns: (message, css_class, icon, description)
    """
    if result_type in RESULT_MESSAGES:
        msg_info = RESULT_MESSAGES[result_type]
        return (
            msg_info["message"],
            msg_info["class"], 
            msg_info["icon"],
            msg_info["description"]
        )
    else:
        # Fallback
        return ("Unknown Result", "ai-medium", "❓", "Unable to classify")
