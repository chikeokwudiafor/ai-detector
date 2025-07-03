
"""
Test AI Detection Module for custom model experimentation
"""

import os
import csv
import json
import logging
from datetime import datetime
from PIL import Image
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomModelManager:
    """Manages loading of custom fine-tuned models alongside base models"""
    
    def __init__(self):
        self.text_models = []
        self.image_models = []
        self.custom_models = []
        self.filename_classifier = None
        self._load_models()
    
    def _load_models(self):
        """Load base models and any custom models"""
        logger.info("Loading AI detection models (TEST MODE)...")
        
        # Load base text models
        for model_config in TEXT_MODELS:
            model = self._load_text_model(model_config)
            if model:
                self.text_models.append({
                    'model': model,
                    'weight': model_config['weight'],
                    'name': model_config['name'],
                    'type': 'base'
                })
        
        # Load base image models  
        for model_config in IMAGE_MODELS:
            model = self._load_image_model(model_config)
            if model:
                self.image_models.append({
                    'model': model,
                    'weight': model_config['weight'],
                    'name': model_config['name'],
                    'type': 'base'
                })
        
        # Try to load custom models if they exist
        self._load_custom_models()
        
        logger.info(f"✓ Loaded {len(self.text_models)} text models, {len(self.image_models)} image models, {len(self.custom_models)} custom models")
    
    def _load_custom_models(self):
        """Load custom fine-tuned models from custom_models directory"""
        custom_dir = "custom_models"
        if not os.path.exists(custom_dir):
            os.makedirs(custom_dir)
            logger.info("Created custom_models directory for future use")
            return
        
        # Look for custom model directories
        for model_dir in os.listdir(custom_dir):
            model_path = os.path.join(custom_dir, model_dir)
            if os.path.isdir(model_path):
                try:
                    # Try to load as text model
                    if os.path.exists(os.path.join(model_path, "config.json")):
                        model = pipeline("text-classification", model=model_path)
                        self.custom_models.append({
                            'model': model,
                            'weight': 0.8,  # Higher weight for custom models
                            'name': f"custom_{model_dir}",
                            'type': 'custom',
                            'model_type': 'text'
                        })
                        logger.info(f"✓ Loaded custom text model: {model_dir}")
                except Exception as e:
                    logger.warning(f"Failed to load custom model {model_dir}: {e}")
        
        # Try to load filename classifier
        filename_model_path = os.path.join(custom_dir, "filename_classifier")
        if os.path.exists(filename_model_path):
            try:
                self.filename_classifier = self._load_filename_classifier(filename_model_path)
                logger.info("✓ Loaded custom filename classifier")
            except Exception as e:
                logger.warning(f"Failed to load filename classifier: {e}")
    
    def _load_filename_classifier(self, model_path):
        """Load custom filename classifier"""
        # This would load your trained filename classification model
        # For now, return None - we'll implement this after training
        return None
    
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

# Global instances for test module
test_model_manager = CustomModelManager()

class AIDetectorTest:
    """Test AI detection class with custom model support"""
    
    @staticmethod
    def detect_text(text_content, filename="unknown.txt"):
        """
        Detect AI-generated text using base + custom models
        Returns: (result_type, confidence, raw_scores)
        """
        start_time = datetime.now()
        
        all_models = test_model_manager.text_models + [m for m in test_model_manager.custom_models if m.get('model_type') == 'text']
        
        if not all_models:
            return "model_unavailable", 0.0, []
        
        try:
            # Truncate text if too long
            if len(text_content) > MAX_TEXT_LENGTH:
                text_content = text_content[:MAX_TEXT_LENGTH]
            
            # Get predictions from all models
            predictions = []
            weights = []
            predictions_data = []
            
            for model_info in all_models:
                try:
                    result = model_info['model'](text_content)
                    confidence = AIDetectorTest._parse_text_result(result)
                    predictions.append(confidence)
                    weights.append(model_info['weight'])
                    
                    predictions_data.append({
                        'model_name': model_info['name'],
                        'confidence': confidence,
                        'weight': model_info['weight'],
                        'model_type': model_info['type'],
                        'raw_result': result
                    })
                    
                    logger.info(f"[TEST] Model {model_info['name']} ({model_info['type']}): {confidence:.3f} (weight: {model_info['weight']})")
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
            
            if not predictions:
                return "processing_error", 0.0, []
            
            # Calculate weighted ensemble score
            ensemble_confidence = np.average(predictions, weights=weights)
            
            # Apply enhanced filename analysis
            filename_boost = AIDetectorTest._enhanced_filename_analysis(filename)
            final_confidence = min(ensemble_confidence * filename_boost, 1.0)
            
            # Classify result
            result_type = AIDetectorTest._classify_confidence(final_confidence)
            
            logger.info(f"[TEST] Ensemble result: {result_type} ({final_confidence:.3f}) - Filename boost: {filename_boost:.3f}")
            
            return result_type, final_confidence, predictions
            
        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return "processing_error", 0.0, []
    
    @staticmethod
    def detect_image(image_file, filename="unknown.jpg"):
        """
        Detect AI-generated images using base + custom models
        Returns: (result_type, confidence, raw_scores)
        """
        start_time = datetime.now()
        
        all_models = test_model_manager.image_models + [m for m in test_model_manager.custom_models if m.get('model_type') == 'image']
        
        if not all_models:
            return "model_unavailable", 0.0, []
        
        try:
            # Load and preprocess image
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get predictions from all models
            predictions = []
            weights = []
            predictions_data = []
            
            for model_info in all_models:
                try:
                    results = model_info['model'](image)
                    confidence = AIDetectorTest._parse_image_result(results)
                    predictions.append(confidence)
                    weights.append(model_info['weight'])
                    
                    predictions_data.append({
                        'model_name': model_info['name'],
                        'confidence': confidence,
                        'weight': model_info['weight'],
                        'model_type': model_info['type'],
                        'raw_result': results
                    })
                    
                    logger.info(f"[TEST] Model {model_info['name']} ({model_info['type']}): {confidence:.3f} (weight: {model_info['weight']})")
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
            
            if not predictions:
                return "processing_error", 0.0, []
            
            # Calculate weighted ensemble score
            ensemble_confidence = np.average(predictions, weights=weights)
            
            # Apply enhanced filename analysis
            filename_boost = AIDetectorTest._enhanced_filename_analysis(filename)
            final_confidence = min(ensemble_confidence * filename_boost, 1.0)
            
            # Classify result
            result_type = AIDetectorTest._classify_confidence(final_confidence)
            
            logger.info(f"[TEST] Ensemble result: {result_type} ({final_confidence:.3f}) - Filename boost: {filename_boost:.3f}")
            
            return result_type, final_confidence, predictions
            
        except Exception as e:
            logger.error(f"Image detection error: {e}")
            return "processing_error", 0.0, []
    
    @staticmethod
    def detect_video(video_file, filename="unknown.mp4"):
        """
        Placeholder for video detection
        Returns: (result_type, confidence, raw_scores)
        """
        return "insufficient", 0.0, []
    
    @staticmethod
    def _enhanced_filename_analysis(filename):
        """Enhanced filename analysis using discovered patterns"""
        if not filename:
            return 1.0
        
        name_without_ext = os.path.splitext(filename)[0].lower()
        boost_factor = 1.0
        
        # Load discovered patterns
        try:
            pattern_file = "pattern_analysis/discovered_patterns.json"
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    patterns = json.load(f)
                
                # Apply AI indicators
                ai_keywords = patterns.get('filename_analysis', {}).get('ai_indicators', {})
                for keyword, count in ai_keywords.items():
                    if keyword in name_without_ext:
                        # Boost factor based on frequency and keyword strength
                        if keyword in ['chatgpt', 'gpt', 'dalle', 'midjourney']:
                            boost_factor *= 1.8  # Strong AI indicators
                        elif keyword in ['ai', 'generated', 'artificial']:
                            boost_factor *= 1.5  # Medium AI indicators
                        else:
                            boost_factor *= 1.2  # Weak AI indicators
                        
                        logger.info(f"[TEST] AI keyword '{keyword}' found, boost: {boost_factor:.3f}")
                        break
                
                # Apply human indicators (reduce confidence)
                human_keywords = patterns.get('filename_analysis', {}).get('human_indicators', {})
                for keyword, count in human_keywords.items():
                    if keyword in name_without_ext:
                        boost_factor *= 0.7  # Reduce AI confidence
                        logger.info(f"[TEST] Human keyword '{keyword}' found, reducing AI confidence")
                        break
        
        except Exception as e:
            logger.warning(f"Could not load patterns for filename analysis: {e}")
        
        return boost_factor
    
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
    def _classify_confidence(confidence):
        """Classify confidence into result categories"""
        if (confidence > (1.0 - CONFIDENCE_THRESHOLD) and 
            confidence < CONFIDENCE_THRESHOLD):
            return "insufficient"
        
        if confidence >= 0.85:
            return "likely_ai"
        elif confidence >= 0.65:
            return "possibly_ai"
        elif confidence >= 0.45:
            return "unsure"
        elif confidence >= 0.21:
            return "likely_human"
        else:
            return "almost_certainly_human"

def get_result_classification(result_type):
    """
    Get display information for a result type
    Returns: (message, css_class, icon, description, footer)
    """
    if result_type in RESULT_MESSAGES:
        msg_info = RESULT_MESSAGES[result_type]
        return (
            msg_info["message"],
            msg_info["class"], 
            msg_info["icon"],
            msg_info["description"],
            msg_info["footer"]
        )
    else:
        # Fallback
        return ("Unknown Result", "confidence-tier-3", "❓", "Unable to classify", "Analysis inconclusive")
