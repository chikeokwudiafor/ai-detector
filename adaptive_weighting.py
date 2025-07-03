
"""
Adaptive Model Weighting System
Uses feedback data to continuously adjust model weights for better accuracy
"""

import json
import numpy as np
from datetime import datetime, timedelta
from config import *

class AdaptiveWeightManager:
    """Manages dynamic model weights based on feedback performance"""
    
    def __init__(self):
        self.feedback_file = "feedback_data/user_feedback.json"
        self.weights_file = "adaptive_weights.json"
        self.performance_window_days = 7  # Look at last week's performance
        
    def load_recent_feedback(self):
        """Load feedback from recent time window"""
        try:
            with open(self.feedback_file, 'r') as f:
                all_feedback = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=self.performance_window_days)
            
            recent_feedback = [
                entry for entry in all_feedback
                if datetime.fromisoformat(entry.get('timestamp', '1970-01-01')) > cutoff_date
            ]
            
            return recent_feedback
        except:
            return []
    
    def calculate_model_accuracy(self):
        """Calculate accuracy for each model based on feedback"""
        feedback = self.load_recent_feedback()
        
        if len(feedback) < 5:
            return None
        
        # Track correct/incorrect predictions per model type
        model_performance = {}
        
        for entry in feedback:
            true_label = entry.get('true_label')
            prediction = entry.get('model_prediction', '').lower()
            
            # Determine if prediction was correct
            is_correct = False
            if true_label == 'ai_generated' and any(word in prediction for word in ['ai', 'likely_ai', 'possibly_ai']):
                is_correct = True
            elif true_label == 'human_created' and any(word in prediction for word in ['human', 'likely_human', 'almost_certainly_human']):
                is_correct = True
            
            # For each model type, track performance
            # (This is simplified - in production, track individual model contributions)
            for model_config in IMAGE_MODELS:
                model_name = model_config['name']
                if model_name not in model_performance:
                    model_performance[model_name] = {'correct': 0, 'total': 0}
                
                model_performance[model_name]['total'] += 1
                if is_correct:
                    model_performance[model_name]['correct'] += 1
        
        # Calculate accuracy rates
        accuracies = {}
        for model_name, stats in model_performance.items():
            if stats['total'] > 0:
                accuracies[model_name] = stats['correct'] / stats['total']
        
        return accuracies
    
    def generate_adaptive_weights(self):
        """Generate new model weights based on recent performance"""
        accuracies = self.calculate_model_accuracy()
        
        if not accuracies:
            print("❌ Insufficient feedback data for weight adaptation")
            return None
        
        # Calculate relative performance and adjust weights
        base_weight = 1.0
        adaptive_weights = {}
        
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        
        for model_name, accuracy in accuracies.items():
            # Adjust weight based on relative performance
            if accuracy > avg_accuracy + 0.1:  # Significantly above average
                weight_multiplier = 1.3
            elif accuracy > avg_accuracy:      # Above average  
                weight_multiplier = 1.1
            elif accuracy < avg_accuracy - 0.1: # Significantly below average
                weight_multiplier = 0.7
            else:                              # Average performance
                weight_multiplier = 1.0
            
            # Find original weight
            original_weight = base_weight
            for model_config in IMAGE_MODELS:
                if model_config['name'] == model_name:
                    original_weight = model_config['weight']
                    break
            
            adaptive_weights[model_name] = original_weight * weight_multiplier
        
        # Save adaptive weights
        weight_data = {
            'timestamp': datetime.now().isoformat(),
            'feedback_samples': len(self.load_recent_feedback()),
            'average_accuracy': avg_accuracy,
            'model_accuracies': accuracies,
            'adaptive_weights': adaptive_weights
        }
        
        with open(self.weights_file, 'w') as f:
            json.dump(weight_data, f, indent=2)
        
        print(f"✅ Generated adaptive weights based on {len(self.load_recent_feedback())} feedback samples")
        print(f"📊 Average accuracy: {avg_accuracy:.1%}")
        
        return adaptive_weights
    
    def get_current_weights(self):
        """Get current adaptive weights or fall back to defaults"""
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
            
            # Check if weights are recent (within 24 hours)
            weight_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - weight_time < timedelta(days=1):
                return data['adaptive_weights']
        except:
            pass
        
        # Fall back to default weights
        return {model['name']: model['weight'] for model in IMAGE_MODELS}

# Global instance
adaptive_weights = AdaptiveWeightManager()
