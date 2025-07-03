
"""
Feedback-Based Model Fine-Tuning System
Uses user feedback to improve model weights and thresholds
"""

import json
import numpy as np
from datetime import datetime
import os
from config import *

class FeedbackTuner:
    """Fine-tune model weights based on user feedback"""
    
    def __init__(self):
        self.feedback_file = "feedback_data/user_feedback.json"
        self.tuning_results_file = "fine_tuning_results.json"
        
    def load_feedback_data(self):
        """Load user feedback data"""
        if not os.path.exists(self.feedback_file):
            return []
        
        with open(self.feedback_file, 'r') as f:
            return json.load(f)
    
    def analyze_model_performance(self):
        """Analyze which models are performing best based on feedback"""
        feedback_data = self.load_feedback_data()
        
        if len(feedback_data) < 5:
            print("❌ Need at least 5 feedback samples for analysis")
            return None
        
        # Track model accuracy by type
        model_stats = {
            'Organika/sdxl-detector': {'correct': 0, 'total': 0, 'confidence_sum': 0},
            'umm-maybe/AI-image-detector': {'correct': 0, 'total': 0, 'confidence_sum': 0},
            'saltacc/anime-ai-detect': {'correct': 0, 'total': 0, 'confidence_sum': 0}
        }
        
        for entry in feedback_data:
            true_label = entry.get('true_label')
            model_prediction = entry.get('model_prediction', '').lower()
            
            # Determine if prediction was correct
            is_correct = False
            if true_label == 'ai_generated' and any(word in model_prediction for word in ['ai', 'likely_ai', 'possibly_ai']):
                is_correct = True
            elif true_label == 'human_created' and any(word in model_prediction for word in ['human', 'likely_human', 'almost_certainly_human']):
                is_correct = True
            
            # Update stats for each model (simplified - in real app, track individual model predictions)
            for model_name in model_stats.keys():
                model_stats[model_name]['total'] += 1
                if is_correct:
                    model_stats[model_name]['correct'] += 1
        
        # Calculate accuracy rates
        results = {}
        for model_name, stats in model_stats.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                results[model_name] = {
                    'accuracy': accuracy,
                    'total_samples': stats['total'],
                    'correct_predictions': stats['correct']
                }
        
        return results
    
    def suggest_weight_adjustments(self):
        """Suggest new model weights based on performance analysis"""
        performance = self.analyze_model_performance()
        
        if not performance:
            return None
        
        print("📊 Model Performance Analysis:")
        print("=" * 50)
        
        total_accuracy = 0
        model_count = 0
        
        suggested_weights = {}
        
        for model_name, stats in performance.items():
            accuracy = stats['accuracy']
            total_accuracy += accuracy
            model_count += 1
            
            print(f"{model_name}:")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Samples: {stats['total_samples']}")
            print(f"  Correct: {stats['correct_predictions']}")
            
            # Calculate suggested weight based on accuracy
            if accuracy >= 0.8:
                suggested_weight = 1.5  # High performer
            elif accuracy >= 0.6:
                suggested_weight = 1.0  # Average performer  
            elif accuracy >= 0.4:
                suggested_weight = 0.7  # Below average
            else:
                suggested_weight = 0.5  # Poor performer
            
            suggested_weights[model_name] = suggested_weight
            print(f"  Suggested weight: {suggested_weight}")
            print()
        
        avg_accuracy = total_accuracy / model_count if model_count > 0 else 0
        print(f"Average ensemble accuracy: {avg_accuracy:.1%}")
        
        # Save results
        results = {
            'analysis_date': datetime.now().isoformat(),
            'model_performance': performance,
            'suggested_weights': suggested_weights,
            'average_accuracy': avg_accuracy,
            'total_feedback_samples': len(self.load_feedback_data())
        }
        
        with open(self.tuning_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to {self.tuning_results_file}")
        
        return suggested_weights
    
    def generate_config_update(self):
        """Generate updated config.py with optimized weights"""
        weights = self.suggest_weight_adjustments()
        
        if not weights:
            print("❌ Cannot generate config update - insufficient data")
            return
        
        print("\n🔧 Suggested config.py updates:")
        print("=" * 50)
        print("IMAGE_MODELS = [")
        
        for model_name, weight in weights.items():
            fallback = '"umm-maybe/AI-image-detector"' if model_name != "umm-maybe/AI-image-detector" else "None"
            print(f'    {{')
            print(f'        "name": "{model_name}",')
            print(f'        "weight": {weight},')
            print(f'        "fallback": {fallback}')
            print(f'    }},')
        
        print("]")
        
        return weights

# Global instance
feedback_tuner = FeedbackTuner()

if __name__ == "__main__":
    print("🚀 Starting feedback-based model tuning...")
    feedback_tuner.generate_config_update()
