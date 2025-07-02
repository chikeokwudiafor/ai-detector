
import json
import os
from datetime import datetime

class FeedbackCollector:
    """Collects user feedback for model improvement"""
    
    def __init__(self, feedback_dir="feedback_data"):
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "user_feedback.json")
        self._ensure_feedback_dir()
    
    def _ensure_feedback_dir(self):
        """Create feedback directory if it doesn't exist"""
        if not os.path.exists(self.feedback_dir):
            os.makedirs(self.feedback_dir)
    
    def save_feedback(self, content, content_type, model_prediction, user_correction, confidence):
        """Save user feedback for future training"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'content_type': content_type,
            'content_preview': content[:200] if content_type == 'text' else f"Image file: {content}",
            'model_prediction': model_prediction,
            'model_confidence': confidence,
            'user_correction': user_correction,
            'feedback_value': 1 if model_prediction == user_correction else -1
        }
        
        # Load existing feedback
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                try:
                    feedback_data = json.load(f)
                except json.JSONDecodeError:
                    feedback_data = []
        else:
            feedback_data = []
        
        # Append new feedback
        feedback_data.append(feedback_entry)
        
        # Save updated feedback
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return True
    
    def get_feedback_stats(self):
        """Get statistics about collected feedback"""
        if not os.path.exists(self.feedback_file):
            return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}
        
        with open(self.feedback_file, 'r') as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}
        
        total = len(feedback_data)
        correct = sum(1 for entry in feedback_data if entry['feedback_value'] == 1)
        incorrect = total - correct
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            "total": total,
            "correct": correct, 
            "incorrect": incorrect,
            "accuracy": accuracy
        }
