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

import threading

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
        self._model_cache = {}
        self._load_models()

    def _load_models(self):
        """Load all configured models with fallbacks"""
        logger.info("Loading AI detection models...")

        # Load text models
        for model_config in TEXT_MODELS:
            model = self._load_text_model(model_config)
            if model:
                # Apply adaptive weight if available
                adaptive_weight = adaptive_weights.get_model_weight(
                    model_config['name'], 
                    model_config['weight']
                )
                self.text_models.append({
                    'model': model,
                    'weight': adaptive_weight,
                    'name': model_config['name']
                })

        # Load image models  
        for model_config in IMAGE_MODELS:
            model = self._load_image_model(model_config)
            if model:
                # Apply adaptive weight if available, but preserve Organika override logic
                adaptive_weight = adaptive_weights.get_model_weight(
                    model_config['name'], 
                    model_config['weight']
                )
                self.image_models.append({
                    'model': model,
                    'weight': adaptive_weight,
                    'name': model_config['name']
                })

        logger.info(f"✓ Loaded {len(self.text_models)} text models and {len(self.image_models)} image models")

    def _load_text_model(self, model_config):
        """Load a text classification model with caching and fallback"""
        model_name = model_config['name']
        
        # Check cache first
        if model_name in self._model_cache:
            logger.info(f"✓ Text model loaded from cache: {model_name}")
            return self._model_cache[model_name]
            
        try:
            # Simple model loading without GPU optimizations
            model = pipeline("text-classification", model=model_name)
            self._model_cache[model_name] = model
            logger.info(f"✓ Text model loaded: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"✗ Failed to load {model_name}: {e}")

            # Try fallback
            if model_config['fallback']:
                fallback_name = model_config['fallback']
                if fallback_name in self._model_cache:
                    return self._model_cache[fallback_name]
                    
                try:
                    model = pipeline("text-classification", model=fallback_name)
                    self._model_cache[fallback_name] = model
                    logger.info(f"✓ Text model loaded (fallback): {fallback_name}")
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

# Global instances with lazy initialization
_model_manager = None
_model_logger = None
_result_cache = {}
_cache_lock = threading.Lock()

def get_model_manager():
    """Get model manager instance with lazy loading"""
    global _model_manager
    if _model_manager is None:
        logger.info("Initializing AI detection models...")
        _model_manager = ModelManager()
    return _model_manager

def get_model_logger():
    """Get model logger instance"""
    global _model_logger
    if _model_logger is None:
        _model_logger = ModelLogger()
    return _model_logger

def _get_cache_key(content_hash, filename):
    """Generate cache key for results"""
    return f"{content_hash}_{filename}"

def _get_cached_result(cache_key):
    """Get cached result if available and fresh"""
    with _cache_lock:
        if cache_key in _result_cache:
            result, timestamp = _result_cache[cache_key]
            # Cache for 10 minutes
            if (datetime.now() - timestamp).total_seconds() < 600:
                return result
            else:
                del _result_cache[cache_key]
    return None

def _cache_result(cache_key, result):
    """Cache a result"""
    with _cache_lock:
        # Keep cache size reasonable
        if len(_result_cache) > 100:
            # Remove oldest entries
            oldest_key = min(_result_cache.keys(), 
                           key=lambda k: _result_cache[k][1])
            del _result_cache[oldest_key]
        _result_cache[cache_key] = (result, datetime.now())

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
    def apply_dynamic_weighting(base_confidence, predictions_data):
        """Adjust confidence based on dynamic weighting."""
        # Example implementation: Increase confidence if models agree
        agreement_count = sum(1 for pred in predictions_data if pred['confidence'] > 0.5)
        if agreement_count > len(predictions_data) / 2:
            base_confidence = min(1.0, base_confidence * 1.1)  # Increase by 10% if more than half agree
            logger.info("Applied dynamic weighting boost due to model agreement.")
        return base_confidence


    @staticmethod
    def apply_confidence_adjustments(base_confidence, metrics, content_features=None, predictions_data=None):
        """Apply heuristic adjustments based on model agreement and content"""
        
        # NOTE: Organika override check is now handled earlier in detect_image/detect_text
        # This method only handles normal adjustments
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

        # Apply Organika trust boost for high confidence (but not 100%)
        if predictions_data:
            organika_confidence = None
            organika_weight = 0

            for pred_data in predictions_data:
                if "Organika" in pred_data['model_name']:
                    organika_confidence = pred_data['confidence']
                    organika_weight = pred_data['weight']
                    break

            if organika_confidence is not None and organika_weight > 1.0:
                # WEIGHTED SYSTEM: For high confidence (but not 100%)
                if organika_confidence >= 0.85 and organika_confidence < 1.0:
                    # Regular trust boost for high confidence
                    trust_boost = 1.3
                    adjusted_confidence *= trust_boost
                    logger.info(f"Applied Organika trust boost: {trust_boost:.3f}")
                elif organika_confidence < 0.6:
                    # Check if there are strong AI semantic indicators in filename that should override Organika human confidence
                    has_strong_ai_indicator = False
                    if content_features:
                        # Don't apply human boost if filename has strong AI indicators like "ChatGPT"
                        has_strong_ai_indicator = 'ai_filename_indicator' in content_features
                    
                    if not has_strong_ai_indicator:
                        # Only apply human confidence boost when Organika is confident it's human AND no conflicting filename evidence
                        human_boost = 0.7
                        adjusted_confidence *= human_boost
                        logger.info(f"Applied Organika human confidence boost: {human_boost:.3f} (Organika: {organika_confidence:.3f})")
                    else:
                        logger.info(f"Skipped Organika human boost due to conflicting AI filename indicator (Organika: {organika_confidence:.3f})")
                # For 0.6-0.999 range, don't apply any Organika-specific adjustments
                # Let the ensemble process handle it normally

        return min(adjusted_confidence, 1.0)

class AIDetector:
    """Main AI detection class with improved ensemble support"""

    @staticmethod
    def _ensure_models_loaded():
        """Ensure models are loaded - for pre-warming if needed"""
        return get_model_manager()

    @staticmethod
    def detect_text(text_content, filename="unknown.txt"):
        """
        Detect AI-generated text using weighted ensemble with caching
        Returns: (result_type, confidence, raw_scores)
        """
        start_time = datetime.now()

        # Check cache first
        content_hash = hash(text_content[:1000])  # Hash first 1000 chars for speed
        cache_key = _get_cache_key(content_hash, filename)
        cached_result = _get_cached_result(cache_key)
        if cached_result:
            logger.info("Returning cached text result")
            return cached_result

        manager = get_model_manager()
        if not manager.text_models:
            return "model_unavailable", 0.0, []

        try:
            # Truncate text if too long
            if len(text_content) > MAX_TEXT_LENGTH:
                text_content = text_content[:MAX_TEXT_LENGTH]

            # Run all text models sequentially
            predictions = []
            weights = []
            predictions_data = []

            for model_info in manager.text_models:
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
                    
                    logger.info(f"Model {model_info['name']}: {confidence:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue

            if not predictions:
                return "processing_error", 0.0, []

            # Calculate ensemble confidence
            ensemble_confidence, metrics = EnsembleVoter.weighted_vote(predictions, weights)

            # Quick feature analysis
            filename_features = AIDetector._analyze_filename(filename)
            final_confidence = EnsembleVoter.apply_confidence_adjustments(
                ensemble_confidence, metrics, filename_features, predictions_data
            )

            # Classify result
            result_type = AIDetector._classify_confidence(final_confidence)

            # Cache the result
            final_result = (result_type, final_confidence, predictions)
            _cache_result(cache_key, final_result)

            # Log the result
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Text result: {result_type} ({final_confidence:.3f}) in {processing_time:.1f}ms")

            try:
                model_logger = get_model_logger()
                ensemble_result = {
                    'result_type': result_type,
                    'confidence': final_confidence
                }
                model_logger.log_prediction("text", filename, predictions_data, ensemble_result, processing_time)
            except Exception as e:
                logger.warning(f"Failed to log text result: {e}")

            return final_result

        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return "processing_error", 0.0, []

    @staticmethod
    def detect_image(image_file, filename="unknown.jpg"):
        """
        Detect AI-generated images using optimized processing with caching
        Returns: (result_type, confidence, raw_scores)
        """
        start_time = datetime.now()

        # Generate cache key from file size and name
        image_file.seek(0, 2)  # Seek to end
        file_size = image_file.tell()
        image_file.seek(0)  # Reset to beginning
        cache_key = _get_cache_key(hash((filename, file_size)), filename)
        
        cached_result = _get_cached_result(cache_key)
        if cached_result:
            logger.info("Returning cached image result")
            
            # Still log cached results for tracking
            try:
                model_logger = get_model_logger()
                processing_time = 0.0  # Cached, so minimal time
                ensemble_result = {
                    'result_type': cached_result[0],
                    'confidence': cached_result[1]
                }
                predictions_data = [{'model_name': 'cached', 'confidence': cached_result[1], 'weight': 1.0, 'raw_result': 'cached'}]
                model_logger.log_prediction("image", filename, predictions_data, ensemble_result, processing_time)
            except Exception as e:
                logger.warning(f"Failed to log cached result: {e}")
            
            return cached_result

        manager = get_model_manager()
        if not manager.image_models:
            return "model_unavailable", 0.0, []

        try:
            # Load and preprocess image
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Run all image models sequentially
            predictions = []
            weights = []
            predictions_data = []

            for model_info in manager.image_models:
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
                    
                    logger.info(f"Model {model_info['name']}: {confidence:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Model {model_info['name']} failed: {e}")
                    continue

            if not predictions:
                return "processing_error", 0.0, []

            # Check for Organika override BEFORE ensemble (PRESERVED - no changes)
            organika_result = None
            for pred_data in predictions_data:
                if "Organika" in pred_data['model_name'] and pred_data['confidence'] >= 0.95:
                    organika_result = pred_data
                    break

            if organika_result:
                logger.info(f"🎯 ORGANIKA HIGH CONFIDENCE OVERRIDE: {organika_result['confidence']:.3f}")
                result_type = AIDetector._classify_confidence(organika_result['confidence'])
                final_result = (result_type, organika_result['confidence'], predictions)
                _cache_result(cache_key, final_result)
                
                # Log the Organika override result
                try:
                    model_logger = get_model_logger()
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    ensemble_result = {
                        'result_type': result_type,
                        'confidence': organika_result['confidence']
                    }
                    model_logger.log_prediction("image", filename, predictions_data, ensemble_result, processing_time)
                except Exception as e:
                    logger.warning(f"Failed to log Organika override: {e}")
                
                return final_result

            # Calculate ensemble confidence
            ensemble_confidence, metrics = EnsembleVoter.weighted_vote(predictions, weights)

            # Apply feature analysis
            filename_features = AIDetector._analyze_filename(filename)
            final_confidence = EnsembleVoter.apply_confidence_adjustments(
                ensemble_confidence, metrics, filename_features, predictions_data
            )

            # Classify result
            result_type = AIDetector._classify_confidence(final_confidence)

            # Cache and return
            final_result = (result_type, final_confidence, predictions)
            _cache_result(cache_key, final_result)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Image result: {result_type} ({final_confidence:.3f}) in {processing_time:.1f}ms")

            # Log the ensemble result
            try:
                model_logger = get_model_logger()
                ensemble_result = {
                    'result_type': result_type,
                    'confidence': final_confidence
                }
                model_logger.log_prediction("image", filename, predictions_data, ensemble_result, processing_time)
            except Exception as e:
                logger.warning(f"Failed to log ensemble result: {e}")

            return final_result

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

        # Get AI keywords from config and convert to lowercase
        ai_keywords = HEURISTICS["filename_semantic"]["ai_keywords"]
        boost_factor = HEURISTICS["filename_semantic"]["ai_boost_factor"]

        features = {}
        for keyword in ai_keywords:
            # Ensure keyword comparison is case-insensitive
            if keyword.lower() in name_without_ext:
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

# Import adaptive weight manager
try:
    from feedback_analyzer import adaptive_weight_manager
    ADAPTIVE_WEIGHTS_AVAILABLE = True
except ImportError:
    ADAPTIVE_WEIGHTS_AVAILABLE = False
    logger.warning("Adaptive weights not available - using static weights")

class AdaptiveWeights:
    """
    Manages adaptive weights for AI detection models based on feedback.
    Now integrated with feedback analysis system.
    """

    def __init__(self):
        # Initialize weights with default values
        self.model_weights = {}
        self._load_adaptive_weights()

    def _load_adaptive_weights(self):
        """Load adaptive weights from feedback analysis"""
        if ADAPTIVE_WEIGHTS_AVAILABLE:
            try:
                adaptive_weights = adaptive_weight_manager.get_updated_weights()
                if adaptive_weights:
                    self.model_weights.update(adaptive_weights)
                    logger.info(f"Loaded {len(adaptive_weights)} adaptive weights")
                
                # Automatically update weights if enough feedback available
                updated = adaptive_weight_manager.update_weights_from_feedback()
                if updated:
                    self.model_weights.update(updated)
                    logger.info("✅ Applied feedback-driven weight updates")
                    
            except Exception as e:
                logger.warning(f"Failed to load adaptive weights: {e}")

    def update_weight(self, model_name, new_weight):
        """
        Updates the weight for a specific model.
        """
        self.model_weights[model_name] = new_weight
        logger.info(f"Updated weight for {model_name} to {new_weight:.2f}")

    def get_current_weights(self):
        """
        Returns the current model weights.
        """
        return self.model_weights
    
    def get_model_weight(self, model_name, default_weight):
        """Get adaptive weight for a model, falling back to default"""
        return self.model_weights.get(model_name, default_weight)

# Initialize AdaptiveWeights instance
adaptive_weights = AdaptiveWeights()