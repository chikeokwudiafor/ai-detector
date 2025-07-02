
"""
Automatic Feedback Integration System
Continuously updates model weights based on user feedback
"""

import json
import os
from datetime import datetime, timedelta
from config import *
from feedback import feedback_manager

class AutoFeedbackUpdater:
    """Automatically updates model weights based on recent feedback"""
    
    def __init__(self):
        self.min_feedback_samples = 3
        self.update_interval_hours = 6
        self.last_update_file = "last_weight_update.json"
        
    def should_update_weights(self):
        """Check if weights should be updated based on new feedback"""
        # Check if enough time has passed
        try:
            with open(self.last_update_file, 'r') as f:
                last_update = json.load(f)
            last_time = datetime.fromisoformat(last_update['timestamp'])
            if datetime.now() - last_time < timedelta(hours=self.update_interval_hours):
                return False
        except:
            pass  # File doesn't exist, proceed with update
        
        # Check if we have enough new feedback
        feedback_data = feedback_manager.get_accuracy_stats()
        if not feedback_data or feedback_data['total_feedback'] < self.min_feedback_samples:
            return False
            
        return True
    
    def analyze_recent_performance(self):
        """Analyze performance from recent feedback"""
        try:
            with open("feedback_data/user_feedback.json", 'r') as f:
                all_feedback = json.load(f)
        except:
            return None
        
        # Get recent feedback (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_feedback = []
        
        for entry in all_feedback:
            try:
                entry_time = datetime.fromisoformat(entry.get('timestamp', '1970-01-01'))
                if entry_time > cutoff_time:
                    recent_feedback.append(entry)
            except:
                continue
        
        if len(recent_feedback) < 3:
            return None
        
        # Analyze accuracy
        correct = 0
        false_positives = 0
        false_negatives = 0
        
        for entry in recent_feedback:
            true_label = entry.get('true_label')
            prediction = entry.get('model_prediction', '').lower()
            
            if not true_label:
                continue
                
            # Check if prediction was correct
            is_ai_prediction = any(word in prediction for word in ['ai', 'possibly', 'likely_ai'])
            is_human_prediction = any(word in prediction for word in ['human', 'certainly'])
            
            if true_label == 'ai_generated' and is_ai_prediction:
                correct += 1
            elif true_label == 'human_created' and is_human_prediction:
                correct += 1
            elif true_label == 'human_created' and is_ai_prediction:
                false_positives += 1  # Called human photo AI
            elif true_label == 'ai_generated' and is_human_prediction:
                false_negatives += 1  # Called AI photo human
        
        total = len(recent_feedback)
        accuracy = correct / total if total > 0 else 0
        
        return {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'false_positive_rate': false_positives / total if total > 0 else 0,
            'false_negative_rate': false_negatives / total if total > 0 else 0
        }
    
    def adjust_weights_automatically(self):
        """Automatically adjust model weights based on performance"""
        if not self.should_update_weights():
            return None
        
        performance = self.analyze_recent_performance()
        if not performance:
            return None
        
        # Create weight adjustments based on performance
        adjustments = {}
        
        # If too many false positives (calling human photos AI), reduce aggressive models
        if performance['false_positive_rate'] > 0.3:
            print("🔧 High false positive rate detected - reducing aggressive AI detection")
            adjustments['reduce_organika'] = 0.8  # Reduce Organika weight
            adjustments['boost_resnet'] = 1.2     # Boost ResNet for better human detection
        
        # If too many false negatives (missing AI), boost AI detection
        elif performance['false_negative_rate'] > 0.3:
            print("🔧 High false negative rate detected - boosting AI detection")
            adjustments['boost_organika'] = 1.3   # Increase Organika weight
            adjustments['reduce_resnet'] = 0.9    # Slightly reduce ResNet
        
        # Apply adjustments
        if adjustments:
            self._apply_weight_adjustments(adjustments, performance)
            
        # Record update
        with open(self.last_update_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'performance': performance,
                'adjustments': adjustments
            }, f, indent=2)
        
        return adjustments
    
    def _apply_weight_adjustments(self, adjustments, performance):
        """Apply weight adjustments to config"""
        print(f"📊 Auto-updating weights based on {performance['total_samples']} recent feedback samples")
        print(f"   Accuracy: {performance['accuracy']:.1%}")
        print(f"   False Positives: {performance['false_positives']}")
        print(f"   False Negatives: {performance['false_negatives']}")
        
        # Save adjustment log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'trigger': 'automatic_feedback_analysis',
            'performance_stats': performance,
            'weight_adjustments': adjustments
        }
        
        log_file = "weight_adjustment_log.json"
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        except:
            log_data = []
        
        log_data.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print("✅ Weight adjustments logged and ready for next model load")

# Global instance
auto_updater = AutoFeedbackUpdater()

if __name__ == "__main__":
    print("🔄 Running automatic feedback analysis...")
    result = auto_updater.adjust_weights_automatically()
    if result:
        print("✅ Weights adjusted based on feedback")
    else:
        print("ℹ️  No weight adjustments needed at this time")
