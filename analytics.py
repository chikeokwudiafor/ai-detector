
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

class ModelAnalytics:
    """Track and analyze model performance over time"""
    
    def __init__(self, logs_dir="logs", feedback_dir="feedback_data"):
        self.logs_dir = logs_dir
        self.feedback_dir = feedback_dir
    
    def get_model_accuracy(self, days=30):
        """Calculate model accuracy based on user feedback"""
        feedback_file = os.path.join(self.feedback_dir, "user_feedback.json")
        
        if not os.path.exists(feedback_file):
            return {"error": "No feedback data available"}
        
        with open(feedback_file, 'r') as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                return {"error": "Invalid feedback data"}
        
        # Filter recent feedback
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = [
            entry for entry in feedback_data
            if datetime.fromisoformat(entry['timestamp']) > cutoff_date
        ]
        
        if not recent_feedback:
            return {"error": f"No feedback in last {days} days"}
        
        # Calculate accuracy by content type
        accuracy_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for entry in recent_feedback:
            content_type = entry['content_type']
            accuracy_by_type[content_type]['total'] += 1
            if entry['feedback_value'] == 1:
                accuracy_by_type[content_type]['correct'] += 1
        
        # Calculate percentages
        results = {}
        for content_type, stats in accuracy_by_type.items():
            results[content_type] = {
                "accuracy": (stats['correct'] / stats['total']) * 100,
                "total_samples": stats['total'],
                "correct_predictions": stats['correct']
            }
        
        return results
    
    def identify_problem_areas(self):
        """Identify content types or confidence ranges with low accuracy"""
        feedback_file = os.path.join(self.feedback_dir, "user_feedback.json")
        
        if not os.path.exists(feedback_file):
            return {"error": "No feedback data available"}
        
        with open(feedback_file, 'r') as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                return {"error": "Invalid feedback data"}
        
        # Analyze by confidence ranges
        confidence_ranges = {
            "high": {"range": (0.8, 1.0), "correct": 0, "total": 0},
            "medium": {"range": (0.4, 0.8), "correct": 0, "total": 0},
            "low": {"range": (0.0, 0.4), "correct": 0, "total": 0}
        }
        
        for entry in feedback_data:
            confidence = entry['model_confidence']
            
            for range_name, range_data in confidence_ranges.items():
                if range_data['range'][0] <= confidence < range_data['range'][1]:
                    range_data['total'] += 1
                    if entry['feedback_value'] == 1:
                        range_data['correct'] += 1
                    break
        
        # Calculate accuracy for each range
        problem_areas = []
        for range_name, data in confidence_ranges.items():
            if data['total'] > 0:
                accuracy = (data['correct'] / data['total']) * 100
                if accuracy < 70:  # Flag ranges with <70% accuracy
                    problem_areas.append({
                        "range": range_name,
                        "confidence_range": data['range'],
                        "accuracy": accuracy,
                        "sample_size": data['total']
                    })
        
        return problem_areas
