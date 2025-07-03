
"""
Training Data Generator
Converts feedback data and discovered patterns into training datasets
"""

import json
import csv
import os
from datetime import datetime
from pattern_discovery import pattern_discovery

class TrainingDataGenerator:
    """Generates training datasets from feedback and pattern analysis"""
    
    def __init__(self):
        self.output_dir = "training_data"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_all_datasets(self):
        """Generate all training datasets"""
        print("🔄 Generating training datasets...")
        
        # Load feedback data
        with open("feedback_data/user_feedback.json", 'r') as f:
            feedback_data = json.load(f)
        
        # Run pattern analysis
        patterns = pattern_discovery.analyze_all_patterns()
        
        # Generate different dataset formats
        datasets = {
            'binary_classification': self._create_binary_dataset(feedback_data),
            'confidence_regression': self._create_confidence_dataset(feedback_data),
            'feature_engineering': self._create_feature_dataset(feedback_data, patterns),
            'filename_classifier': self._create_filename_dataset(feedback_data)
        }
        
        # Save datasets
        for name, data in datasets.items():
            self._save_dataset(name, data)
        
        # Generate training metadata
        self._generate_metadata(datasets, patterns)
        
        return datasets
    
    def _create_binary_dataset(self, feedback_data):
        """Create binary classification dataset (AI vs Human)"""
        dataset = []
        
        for entry in feedback_data:
            if entry.get('true_label'):
                # Convert to binary labels
                label = 1 if entry['true_label'] == 'ai_generated' else 0
                
                dataset.append({
                    'filename': entry.get('filename', ''),
                    'file_type': entry.get('file_type', ''),
                    'label': label,
                    'label_text': entry['true_label'],
                    'model_prediction': entry.get('model_prediction', ''),
                    'timestamp': entry.get('timestamp', '')
                })
        
        return dataset
    
    def _create_confidence_dataset(self, feedback_data):
        """Create confidence regression dataset"""
        dataset = []
        
        for entry in feedback_data:
            if entry.get('true_label'):
                # Estimate ground truth confidence
                confidence = 1.0 if entry['true_label'] == 'ai_generated' else 0.0
                
                # Extract model confidence from prediction
                model_pred = entry.get('model_prediction', '').lower()
                model_confidence = self._extract_model_confidence(model_pred)
                
                dataset.append({
                    'filename': entry.get('filename', ''),
                    'file_type': entry.get('file_type', ''),
                    'true_confidence': confidence,
                    'model_confidence': model_confidence,
                    'confidence_error': abs(model_confidence - confidence),
                    'prediction_text': model_pred
                })
        
        return dataset
    
    def _create_feature_dataset(self, feedback_data, patterns):
        """Create dataset with engineered features"""
        dataset = []
        
        # Get discovered patterns
        ai_keywords = patterns['patterns']['filename_analysis']['ai_indicators']
        human_keywords = patterns['patterns']['filename_analysis']['human_indicators']
        
        for entry in feedback_data:
            if entry.get('true_label'):
                filename = entry.get('filename', '').lower()
                
                # Engineer features
                features = self._engineer_features(filename, entry, ai_keywords, human_keywords)
                
                # Add label
                features['label'] = 1 if entry['true_label'] == 'ai_generated' else 0
                features['true_label'] = entry['true_label']
                
                dataset.append(features)
        
        return dataset
    
    def _create_filename_dataset(self, feedback_data):
        """Create dataset specifically for filename classification"""
        dataset = []
        
        for entry in feedback_data:
            if entry.get('true_label') and entry.get('filename'):
                filename = entry['filename']
                
                # Extract filename features
                features = {
                    'filename': filename,
                    'filename_lower': filename.lower(),
                    'filename_length': len(filename),
                    'has_numbers': any(c.isdigit() for c in filename),
                    'has_spaces': ' ' in filename,
                    'has_underscores': '_' in filename,
                    'has_dashes': '-' in filename,
                    'word_count': len(filename.split()),
                    'extension': filename.split('.')[-1] if '.' in filename else '',
                    'label': 1 if entry['true_label'] == 'ai_generated' else 0
                }
                
                # Add AI keyword indicators
                ai_keywords = ['chatgpt', 'gpt', 'dalle', 'midjourney', 'ai', 'generated']
                for keyword in ai_keywords:
                    features[f'has_{keyword}'] = keyword in filename.lower()
                
                dataset.append(features)
        
        return dataset
    
    def _engineer_features(self, filename, entry, ai_keywords, human_keywords):
        """Engineer features for machine learning"""
        features = {
            # Basic features
            'filename': filename,
            'file_type': entry.get('file_type', ''),
            'filename_length': len(filename),
            
            # Character features
            'has_numbers': 1 if any(c.isdigit() for c in filename) else 0,
            'has_spaces': 1 if ' ' in filename else 0,
            'has_underscores': 1 if '_' in filename else 0,
            'special_char_count': sum(1 for c in filename if not c.isalnum() and c != '.'),
            
            # Word features
            'word_count': len(filename.split()),
            'avg_word_length': sum(len(word) for word in filename.split()) / max(len(filename.split()), 1),
            
            # AI keyword features
            'ai_keyword_count': sum(1 for keyword in ai_keywords if keyword in filename),
            'human_keyword_count': sum(1 for keyword in human_keywords if keyword in filename),
        }
        
        # Specific keyword flags
        important_keywords = ['chatgpt', 'gpt', 'dalle', 'midjourney', 'ai', 'generated', 'image', 'photo']
        for keyword in important_keywords:
            features[f'has_{keyword}'] = 1 if keyword in filename else 0
        
        # File extension
        extension = filename.split('.')[-1] if '.' in filename else 'none'
        features['extension'] = extension
        features['is_image'] = 1 if extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] else 0
        features['is_text'] = 1 if extension in ['txt', 'md'] else 0
        
        return features
    
    def _extract_model_confidence(self, prediction_text):
        """Extract confidence from model prediction text"""
        if 'likely' in prediction_text and 'ai' in prediction_text:
            return 0.85
        elif 'possibly' in prediction_text:
            return 0.65
        elif 'likely' in prediction_text and 'human' in prediction_text:
            return 0.15  # Low AI confidence
        elif 'almost certainly' in prediction_text and 'human' in prediction_text:
            return 0.05
        elif 'unsure' in prediction_text:
            return 0.5
        else:
            return 0.6  # Default
    
    def _save_dataset(self, name, data):
        """Save dataset in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = f"{self.output_dir}/{name}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save as CSV
        if data:
            csv_file = f"{self.output_dir}/{name}_{timestamp}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        
        print(f"✅ Saved {name} dataset: {len(data)} samples")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")
    
    def _generate_metadata(self, datasets, patterns):
        """Generate metadata about the datasets"""
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_samples': sum(len(dataset) for dataset in datasets.values()),
            'datasets': {
                name: {
                    'sample_count': len(data),
                    'features': list(data[0].keys()) if data else [],
                    'ai_samples': sum(1 for item in data if item.get('label') == 1 or item.get('true_label') == 'ai_generated'),
                    'human_samples': sum(1 for item in data if item.get('label') == 0 or item.get('true_label') == 'human_created')
                }
                for name, data in datasets.items()
            },
            'pattern_insights': patterns['insights'] if patterns else [],
            'recommended_models': [
                'Random Forest (for filename classification)',
                'Logistic Regression (for binary classification)',
                'XGBoost (for feature-based classification)',
                'Neural Network (for complex pattern learning)'
            ]
        }
        
        metadata_file = f"{self.output_dir}/training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Generated metadata: {metadata_file}")

# Create global instance
training_generator = TrainingDataGenerator()

if __name__ == "__main__":
    training_generator.generate_all_datasets()
