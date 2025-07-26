
import json
import os
import pickle
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
from feedback_analyzer import FeedbackAnalyzer

logger = logging.getLogger(__name__)

class CustomModelTrainer:
    """
    Trains and maintains a custom AI detection model based on user feedback
    """
    
    def __init__(self, model_dir="custom_models"):
        self.model_dir = model_dir
        self.analyzer = FeedbackAnalyzer()
        self.feature_extractor = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = None
        self.model_metadata = {}
        self.training_history = []
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to load existing model
        self._load_existing_model()
    
    def extract_features(self, feedback_data):
        """Extract features from feedback data for training"""
        features = []
        labels = []
        
        for entry in feedback_data:
            if 'true_label' not in entry or 'filename' not in entry:
                continue
            
            # Feature extraction
            feature_vector = self._extract_entry_features(entry)
            features.append(feature_vector)
            
            # Label mapping
            label = 1 if entry['true_label'] == 'ai_generated' else 0
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _extract_entry_features(self, entry):
        """Extract features from a single feedback entry"""
        features = []
        filename = entry.get('filename', '').lower()
        
        # Filename-based features
        ai_keywords = ['chatgpt', 'gpt', 'dalle', 'midjourney', 'ai', 'generated', 'synthetic']
        features.append(sum(1 for keyword in ai_keywords if keyword in filename))
        
        # File type features
        file_type = entry.get('file_type', '')
        features.append(1 if file_type == 'image' else 0)
        features.append(1 if file_type == 'text' else 0)
        
        # Model prediction confidence (reverse engineered from result)
        model_prediction = entry.get('model_prediction', '')
        confidence_score = self._map_prediction_to_confidence(model_prediction)
        features.append(confidence_score)
        
        # Pattern-based features
        features.append(len(filename))  # Filename length
        features.append(filename.count('_'))  # Underscore count
        features.append(filename.count('-'))  # Dash count
        features.append(1 if any(char.isdigit() for char in filename) else 0)  # Has numbers
        
        return features
    
    def _map_prediction_to_confidence(self, prediction):
        """Map model prediction text to confidence score"""
        confidence_mapping = {
            "🤖 Likely AI-Generated": 0.9,
            "⚠️ Possibly AI-Generated": 0.7,
            "🤔 Unsure – Needs a Closer Look": 0.5,
            "🧠 Likely Human": 0.3,
            "✅ Almost Certainly Human": 0.1
        }
        return confidence_mapping.get(prediction, 0.5)
    
    def train_model(self, min_samples=10):
        """Train the custom model on available feedback data"""
        feedback_data = self.analyzer.load_feedback_data()
        
        if len(feedback_data) < min_samples:
            logger.warning(f"Not enough feedback data for training (need {min_samples}, have {len(feedback_data)})")
            return False
        
        try:
            # Extract features and labels
            X, y = self.extract_features(feedback_data)
            
            if len(X) < min_samples:
                logger.warning(f"Not enough valid entries for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Update metadata
            self.model_metadata = {
                'training_date': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'feature_count': X.shape[1],
                'total_feedback_entries': len(feedback_data)
            }
            
            # Record training history
            self.training_history.append(self.model_metadata.copy())
            
            # Save model
            self._save_model()
            
            logger.info(f"✅ Custom model trained successfully!")
            logger.info(f"Training accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
            logger.info(f"Trained on {len(X)} samples from {len(feedback_data)} feedback entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train custom model: {e}")
            return False
    
    def predict(self, entry_data):
        """Make prediction using custom model"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Extract features for single entry
            features = np.array([self._extract_entry_features(entry_data)])
            
            # Get prediction and probability
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Return AI confidence (probability of class 1)
            ai_confidence = probabilities[1]
            
            return prediction, ai_confidence
            
        except Exception as e:
            logger.error(f"Custom model prediction failed: {e}")
            return None, 0.0
    
    def get_model_info(self):
        """Get information about the current custom model"""
        if self.model is None:
            return {
                'status': 'not_trained',
                'message': 'Custom model not yet trained'
            }
        
        return {
            'status': 'trained',
            'metadata': self.model_metadata,
            'training_history_count': len(self.training_history),
            'feature_count': self.model_metadata.get('feature_count', 0)
        }
    
    def should_retrain(self, new_feedback_threshold=10):
        """Check if model should be retrained based on new feedback"""
        if self.model is None:
            return True
        
        current_feedback = len(self.analyzer.load_feedback_data())
        last_training_count = self.model_metadata.get('total_feedback_entries', 0)
        
        new_feedback_count = current_feedback - last_training_count
        
        return new_feedback_count >= new_feedback_threshold
    
    def auto_retrain_if_needed(self):
        """Automatically retrain if enough new feedback is available"""
        if self.should_retrain():
            logger.info("🔄 Auto-retraining custom model with new feedback data")
            return self.train_model()
        return False
    
    def _save_model(self):
        """Save trained model and metadata"""
        if self.model is None:
            return
        
        try:
            # Save model
            model_file = os.path.join(self.model_dir, "custom_ai_detector.pkl")
            joblib.dump(self.model, model_file)
            
            # Save metadata
            metadata_file = os.path.join(self.model_dir, "model_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'metadata': self.model_metadata,
                    'training_history': self.training_history
                }, f, indent=2)
            
            logger.info(f"Custom model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save custom model: {e}")
    
    def _load_existing_model(self):
        """Load existing trained model if available"""
        model_file = os.path.join(self.model_dir, "custom_ai_detector.pkl")
        metadata_file = os.path.join(self.model_dir, "model_metadata.json")
        
        if os.path.exists(model_file) and os.path.exists(metadata_file):
            try:
                # Load model
                self.model = joblib.load(model_file)
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.model_metadata = data.get('metadata', {})
                    self.training_history = data.get('training_history', [])
                
                logger.info(f"✅ Loaded existing custom model (accuracy: {self.model_metadata.get('test_accuracy', 0):.3f})")
                
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                self.model = None

class ContinuousTrainer:
    """Manages continuous training of the custom model"""
    
    def __init__(self):
        self.trainer = CustomModelTrainer()
        self.last_feedback_count = 0
        
    def check_and_update(self):
        """Check for new feedback and retrain if needed"""
        try:
            # Check if retraining is needed
            retrained = self.trainer.auto_retrain_if_needed()
            
            if retrained:
                logger.info("🎯 Custom model updated with latest feedback!")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Continuous training update failed: {e}")
            return False
    
    def get_model_for_ensemble(self):
        """Get the custom model for use in ensemble prediction"""
        if self.trainer.model is None:
            return None
        
        return {
            'name': 'Custom Feedback Model',
            'model': self.trainer,
            'weight': 1.5,  # Good weight for feedback-trained model
            'predict_method': 'predict'
        }

# Global instance
custom_trainer = ContinuousTrainer()

def get_custom_model_prediction(entry_data):
    """Get prediction from custom model for ensemble use"""
    custom_model = custom_trainer.get_model_for_ensemble()
    if custom_model is None:
        return None, 0.0
    
    return custom_model['model'].predict(entry_data)
