
"""
AI Detection Module with ensemble support and logging
"""

import os
import csv
import json
import logging
from datetime import datetime
from PIL import Image
import torch
from transformers import pipeline
import numpy as np
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLogger:
    """Handles logging of model predictions to CSV and JSON"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.csv_file = os.path.join(log_dir, "model_predictions.csv")
        self.json_file = os.path.join(log_dir, "detailed_predictions.json")
        self._ensure_log_dir()
        self._init_csv()
    
    def _ensure_log_dir(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'file_type', 'filename', 'model_name', 
                    'individual_confidence', 'ensemble_confidence', 'final_result',
                    'processing_time_ms'
                ])
    
    def log_prediction(self, file_type, filename, predictions_data, ensemble_result, processing_time):
        """Log prediction to both CSV and JSON"""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV (one row per model)
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for pred in predictions_data:
                writer.writerow([
                    timestamp, file_type, filename, pred['model_name'],
                    pred['confidence'], ensemble_result['confidence'], 
                    ensemble_result['result_type'], processing_time
                ])
        
        # Log detailed info to JSON
        log_entry = {
            'timestamp': timestamp,
            'file_type': file_type,
            'filename': filename,
            'individual_predictions': predictions_data,
            'ensemble_result': ensemble_result,
            'processing_time_ms': processing_time
        }
        
        # Append to JSON file
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        data.append(log_entry)
        
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)

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

# Global instances
model_manager = ModelManager()
model_logger = ModelLogger()

class EnsembleVoter:
    """Handles weighted voting and confidence calculations"""
    
    @staticmethod
    def weighted_vote(predictions, weights):
        """Calculate weighted ensemble score with confidence metrics"""
        if not predictions or not weights:
            return 0.0, {'std_dev': 0.0, 'agreement': 0.0}
        
        # Calculate weighted average
        ensemble_score = np.average(predictions, weights=weights)
        
        # Calculate agreement metrics
        std_dev = np.std(predictions) if len(predictions) > 1 else 0.0
        
        # Agreement score: higher when models agree (lower std dev)
        agreement = max(0.0, 1.0 - (std_dev / 0.5))  # Normalize to 0-1
        
        return ensemble_score, {
            'std_dev': std_dev,
            'agreement': agreement,
            'model_count': len(predictions)
        }
    
    @staticmethod
    def apply_confidence_adjustments(base_confidence, metrics, content_features=None):
        """Apply heuristic adjustments based on model agreement and content"""
        adjusted_confidence = base_confidence
        
        # Reduce confidence if models disagree significantly
        disagreement_threshold = HEURISTICS["ensemble"]["disagreement_threshold"]
        max_penalty = HEURISTICS["ensemble"]["max_disagreement_penalty"]
        
        if metrics['std_dev'] > disagreement_threshold:
            disagreement_penalty = min(max_penalty, metrics['std_dev'] / 2)
            adjusted_confidence *= (1.0 - disagreement_penalty)
            logger.info(f"Applied disagreement penalty: -{disagreement_penalty:.3f}")
        
        # Apply content-specific adjustments
        if content_features:
            for feature, adjustment in content_features.items():
                adjusted_confidence *= adjustment
                logger.info(f"Applied {feature} adjustment: {adjustment:.3f}")
        
        return min(adjusted_confidence, 1.0)

class AIDetector:
    """Main AI detection class with improved ensemble support"""
    
    @staticmethod
    def detect_text(text_content, filename="unknown.txt"):
        """
        Detect AI-generated text using weighted ensemble
        Returns: (result_type, confidence, raw_scores)
        """
        start_time = datetime.now()
        
        if not model_manager.text_models:
            return "model_unavailable", 0.0, []
        
        try:
            # Truncate text if too long
            if len(text_content) > MAX_TEXT_LENGTH:
                text_content = text_content[:MAX_TEXT_LENGTH]
            
            # Get predictions from all models
            predictions = []
            weights = []
            predictions_data = []
            
            for model_info in model_manager.text_models:
                try:
                    result = model_info['model'](text_content)
                    confidence = AIDetector._parse_text_result(result)
                    predictions.append(confidence)
                    weights.append(model_info['weight'])
                    
                    predictions_data.append({
                        'model_name': model_info['name'],
                        'confidence': confidence,
                        'weight': model_info['weight'],
                        'raw_result': result
                    })
                    
                    logger.info(f"Model {model_info['name']}: {confidence:.3f} (weight: {model_info['weight']})")
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
            
            if not predictions:
                return "processing_error", 0.0, []
            
            # Calculate weighted ensemble score
            ensemble_confidence, metrics = EnsembleVoter.weighted_vote(predictions, weights)
            
            # Apply content-based and filename adjustments
            content_features = AIDetector._analyze_text_features(text_content)
            filename_features = AIDetector._analyze_filename(filename)
            all_features = {**content_features, **filename_features}
            final_confidence = EnsembleVoter.apply_confidence_adjustments(
                ensemble_confidence, metrics, all_features
            )
            
            # Classify result
            result_type = AIDetector._classify_confidence(final_confidence)
            
            # Log prediction
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            ensemble_result = {
                'result_type': result_type,
                'confidence': final_confidence,
                'metrics': metrics
            }
            model_logger.log_prediction("text", filename, predictions_data, ensemble_result, processing_time)
            
            logger.info(f"Ensemble result: {result_type} ({final_confidence:.3f}) - Agreement: {metrics['agreement']:.3f}")
            
            return result_type, final_confidence, predictions
            
        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return "processing_error", 0.0, []
    
    @staticmethod
    def detect_image(image_file, filename="unknown.jpg"):
        """
        Detect AI-generated images using weighted ensemble
        Returns: (result_type, confidence, raw_scores)
        """
        start_time = datetime.now()
        
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
            predictions_data = []
            
            for model_info in model_manager.image_models:
                try:
                    results = model_info['model'](image)
                    confidence = AIDetector._parse_image_result(results)
                    predictions.append(confidence)
                    weights.append(model_info['weight'])
                    
                    predictions_data.append({
                        'model_name': model_info['name'],
                        'confidence': confidence,
                        'weight': model_info['weight'],
                        'raw_result': results
                    })
                    
                    logger.info(f"Model {model_info['name']}: {confidence:.3f} (weight: {model_info['weight']})")
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue
            
            if not predictions:
                return "processing_error", 0.0, []
            
            # Calculate weighted ensemble score
            ensemble_confidence, metrics = EnsembleVoter.weighted_vote(predictions, weights)
            
            # Apply image-based and filename adjustments
            content_features = AIDetector._analyze_image_features(image)
            filename_features = AIDetector._analyze_filename(filename)
            all_features = {**content_features, **filename_features}
            final_confidence = EnsembleVoter.apply_confidence_adjustments(
                ensemble_confidence, metrics, all_features
            )
            
            # Classify result
            result_type = AIDetector._classify_confidence(final_confidence)
            
            # Log prediction
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            ensemble_result = {
                'result_type': result_type,
                'confidence': final_confidence,
                'metrics': metrics
            }
            model_logger.log_prediction("image", filename, predictions_data, ensemble_result, processing_time)
            
            logger.info(f"Ensemble result: {result_type} ({final_confidence:.3f}) - Agreement: {metrics['agreement']:.3f}")
            
            return result_type, final_confidence, predictions
            
        except Exception as e:
            logger.error(f"Image detection error: {e}")
            return "processing_error", 0.0, []
    
    @staticmethod
    def detect_video(video_file, filename="unknown.mp4"):
        """
        Placeholder for video detection - Coming soon!
        Returns: (result_type, confidence, raw_scores)
        """
        logger.warning("Video detection not implemented yet. Coming soon.")
        logger.info(f"Video upload attempted: {filename}")
        
        # Return a specific result type for video not implemented
        return "video_not_implemented", 0.0, []
    
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
    def _analyze_text_features(text_content):
        """Analyze text features for confidence adjustments"""
        features = {}
        text_config = HEURISTICS["text_features"]
        
        # Very short text is harder to classify reliably
        if len(text_content) < 50:
            features['short_text'] = text_config["short_text_penalty"]
        
        # Check sentence length variation (AI often has uniform patterns)
        sentences = [s.strip() for s in text_content.split('.') if s.strip()]
        if len(sentences) > 3:
            lengths = [len(s) for s in sentences]
            if np.std(lengths) < text_config["uniform_threshold"]:
                features['uniform_sentences'] = text_config["uniform_sentences_boost"]
        
        return features
    
    @staticmethod
    def _analyze_filename(filename):
        """Analyze filename for semantic AI-related keywords (not file extensions)"""
        if not filename:
            return {}
        
        # Remove file extension and convert to lowercase
        name_without_ext = os.path.splitext(filename)[0].lower()
        
        # Get AI keywords from config
        ai_keywords = HEURISTICS["filename_semantic"]["ai_keywords"]
        boost_factor = HEURISTICS["filename_semantic"]["ai_boost_factor"]
        
        features = {}
        for keyword in ai_keywords:
            if keyword in name_without_ext:
                # Strong semantic indicator of AI generation
                features['ai_filename_indicator'] = boost_factor
                logger.info(f"AI semantic keyword '{keyword}' found in filename: {filename}")
                break
        
        return features
    
    @staticmethod
    def _analyze_image_features(image):
        """Analyze image features for confidence adjustments"""
        features = {}
        image_config = HEURISTICS["image_features"]
        
        # Very small images are harder to classify
        width, height = image.size
        min_size = image_config["min_size_threshold"]
        if width < min_size or height < min_size:
            features['small_image'] = image_config["small_image_penalty"]
        
        return features
    
    @staticmethod
    def _classify_confidence(confidence):
        """Classify confidence into 5-tier result categories"""
        # Check if confidence meets minimum threshold
        if (confidence > (1.0 - CONFIDENCE_THRESHOLD) and 
            confidence < CONFIDENCE_THRESHOLD):
            return "insufficient"
        
        # 5-tier classification based on confidence percentage
        if confidence >= 0.85:  # 85%-100%
            return "likely_ai"
        elif confidence >= 0.65:  # 65%-84%
            return "possibly_ai"
        elif confidence >= 0.45:  # 45%-64%
            return "unsure"
        elif confidence >= 0.21:  # 21%-44%
            return "likely_human"
        else:  # 0%-20%
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
