
"""
Feedback system for tracking model accuracy
"""

import json
import os
from datetime import datetime

class FeedbackManager:
    """Manages user feedback on model predictions"""
    
    def __init__(self, feedback_dir="feedback_data"):
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "user_feedback.json")
        self._ensure_feedback_dir()
        self._init_feedback_file()
    
    def _ensure_feedback_dir(self):
        """Create feedback directory if it doesn't exist"""
        if not os.path.exists(self.feedback_dir):
            os.makedirs(self.feedback_dir)
    
    def _init_feedback_file(self):
        """Initialize feedback file if it doesn't exist"""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)
    
    def save_feedback(self, session_id, file_type, filename, model_result, true_label):
        """
        Save user feedback on model prediction
        
        Args:
            session_id: Unique session identifier
            file_type: Type of content (image, text, video)
            filename: Name of analyzed file
            model_result: The model's prediction result
            true_label: User's indication of actual content source (ai_generated/human_created)
        """
        try:
            # Load existing feedback
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            # Create new feedback entry
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'file_type': file_type,
                'filename': filename,
                'model_prediction': model_result,
                'true_label': true_label,
                'feedback_type': 'true_label_verification'
            }
            
            # Add to data
            feedback_data.append(feedback_entry)
            
            # Save back to file
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_accuracy_stats(self):
        """Get accuracy statistics from user feedback"""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            if not feedback_data:
                return None
            
            total_feedback = len(feedback_data)
            correct_predictions = 0
            
            for entry in feedback_data:
                model_pred = entry.get('model_prediction', '').lower()
                true_label = entry.get('true_label', '')
                
                # Check if model prediction matches true label
                if true_label == 'ai_generated' and 'ai' in model_pred:
                    correct_predictions += 1
                elif true_label == 'human_created' and 'human' in model_pred:
                    correct_predictions += 1
            
            stats = {
                'total_feedback': total_feedback,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': total_feedback - correct_predictions,
                'accuracy_rate': (correct_predictions / total_feedback) * 100 if total_feedback > 0 else 0,
                'ai_labeled_count': sum(1 for entry in feedback_data if entry.get('true_label') == 'ai_generated'),
                'human_labeled_count': sum(1 for entry in feedback_data if entry.get('true_label') == 'human_created')
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting accuracy stats: {e}")
            return None

# Global feedback manager instance
feedback_manager = FeedbackManager()
