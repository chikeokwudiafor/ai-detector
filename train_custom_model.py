
"""
Custom Model Training Script
Uses feedback data and patterns to train improved AI detection models
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from datetime import datetime

class CustomModelTrainer:
    """Trains custom models using feedback data and discovered patterns"""
    
    def __init__(self):
        self.models_dir = "custom_models"
        self.reports_dir = "training_reports"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def train_filename_classifier(self):
        """Train a custom filename-based classifier"""
        print("🔄 Training filename classifier...")
        
        # Load feedback data
        with open("feedback_data/user_feedback.json", 'r') as f:
            feedback_data = json.load(f)
        
        if len(feedback_data) < 10:
            print("❌ Insufficient data for training (need at least 10 samples)")
            return None
        
        # Prepare training data
        data = []
        for entry in feedback_data:
            if entry.get('true_label') and entry.get('filename'):
                features = self._extract_filename_features(entry['filename'])
                features['label'] = 1 if entry['true_label'] == 'ai_generated' else 0
                data.append(features)
        
        if len(data) < 5:
            print("❌ Insufficient filename data for training")
            return None
        
        df = pd.DataFrame(data)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'label']
        X = df[feature_cols]
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Filename classifier trained!")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Save model
        model_path = os.path.join(self.models_dir, "filename_classifier.joblib")
        feature_names_path = os.path.join(self.models_dir, "filename_features.json")
        
        joblib.dump(model, model_path)
        with open(feature_names_path, 'w') as f:
            json.dump(feature_cols, f)
        
        # Generate report
        report = {
            'model_type': 'filename_classifier',
            'accuracy': accuracy,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        report_path = os.path.join(self.reports_dir, f"filename_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Training report saved: {report_path}")
        
        return model
    
    def train_content_classifier(self):
        """Train a content-based text classifier using TF-IDF"""
        print("🔄 Training content classifier...")
        
        # This would require text content from your feedback
        # For now, we'll focus on the filename classifier
        print("⏳ Content classifier training coming soon...")
        return None
    
    def _extract_filename_features(self, filename):
        """Extract features from filename for training"""
        name_without_ext = os.path.splitext(filename)[0].lower()
        
        features = {
            'filename_length': len(filename),
            'has_numbers': 1 if any(c.isdigit() for c in filename) else 0,
            'has_spaces': 1 if ' ' in filename else 0,
            'has_underscores': 1 if '_' in filename else 0,
            'has_dashes': 1 if '-' in filename else 0,
            'special_char_count': sum(1 for c in filename if not c.isalnum() and c != '.'),
            'word_count': len(filename.split()),
            'avg_word_length': sum(len(word) for word in filename.split()) / max(len(filename.split()), 1),
        }
        
        # AI keyword features
        ai_keywords = ['chatgpt', 'gpt', 'dalle', 'midjourney', 'ai', 'generated', 'artificial', 'synthetic']
        for keyword in ai_keywords:
            features[f'has_{keyword}'] = 1 if keyword in name_without_ext else 0
        
        # File extension
        extension = filename.split('.')[-1] if '.' in filename else 'none'
        features['is_image'] = 1 if extension.lower() in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] else 0
        features['is_text'] = 1 if extension.lower() in ['txt', 'md'] else 0
        
        return features
    
    def load_filename_classifier(self):
        """Load trained filename classifier"""
        model_path = os.path.join(self.models_dir, "filename_classifier.joblib")
        feature_names_path = os.path.join(self.models_dir, "filename_features.json")
        
        if os.path.exists(model_path) and os.path.exists(feature_names_path):
            model = joblib.load(model_path)
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            return model, feature_names
        return None, None
    
    def predict_filename(self, filename):
        """Use trained filename classifier to predict if filename indicates AI content"""
        model, feature_names = self.load_filename_classifier()
        if model is None:
            return None
        
        features = self._extract_filename_features(filename)
        
        # Create feature vector in correct order
        feature_vector = [features.get(name, 0) for name in feature_names]
        
        # Get prediction and probability
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]
        
        ai_confidence = probabilities[1] if len(probabilities) > 1 else 0.5
        
        return {
            'prediction': 'ai_generated' if prediction == 1 else 'human_created',
            'ai_confidence': ai_confidence,
            'human_confidence': 1.0 - ai_confidence
        }

# Global instance
custom_trainer = CustomModelTrainer()

if __name__ == "__main__":
    print("🚀 Starting custom model training...")
    
    # Train filename classifier
    filename_model = custom_trainer.train_filename_classifier()
    
    if filename_model:
        print("\n🎯 Testing filename classifier:")
        test_filenames = [
            "ChatGPT Image Jul 2, 2025.png",
            "vacation_photo.jpg",
            "DALL-E generated art.png",
            "my_selfie.jpeg",
            "midjourney_creation.png"
        ]
        
        for filename in test_filenames:
            result = custom_trainer.predict_filename(filename)
            if result:
                print(f"  {filename}: {result['prediction']} ({result['ai_confidence']:.3f})")
    
    print("\n✅ Training complete! Run your test app with: python app_test.py")
