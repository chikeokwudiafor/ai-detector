
"""
Model Comparison Tool
Compares performance between base models and custom models
"""

import json
import os
from datetime import datetime
from detection import AIDetector
from detection_test import AIDetectorTest
from train_custom_model import custom_trainer

class ModelComparison:
    """Compare base vs custom model performance"""
    
    def __init__(self):
        self.comparison_dir = "model_comparisons"
        os.makedirs(self.comparison_dir, exist_ok=True)
    
    def run_comparison(self):
        """Run comparison between base and custom models"""
        print("🔍 Running model comparison...")
        
        # Load feedback data for testing
        with open("feedback_data/user_feedback.json", 'r') as f:
            feedback_data = json.load(f)
        
        if len(feedback_data) < 5:
            print("❌ Insufficient feedback data for comparison")
            return
        
        results = {
            'comparison_date': datetime.now().isoformat(),
            'total_samples': len(feedback_data),
            'base_model_results': [],
            'custom_model_results': [],
            'filename_classifier_results': []
        }
        
        for entry in feedback_data:
            if not entry.get('true_label'):
                continue
            
            filename = entry.get('filename', 'unknown')
            true_label = entry['true_label']
            
            # Test filename classifier if available
            filename_result = custom_trainer.predict_filename(filename)
            if filename_result:
                results['filename_classifier_results'].append({
                    'filename': filename,
                    'true_label': true_label,
                    'predicted_label': filename_result['prediction'],
                    'confidence': filename_result['ai_confidence'],
                    'correct': filename_result['prediction'] == true_label
                })
        
        # Calculate accuracy
        if results['filename_classifier_results']:
            correct = sum(1 for r in results['filename_classifier_results'] if r['correct'])
            total = len(results['filename_classifier_results'])
            accuracy = correct / total
            
            results['filename_classifier_accuracy'] = accuracy
            
            print(f"📊 Filename Classifier Results:")
            print(f"   Accuracy: {accuracy:.3f} ({correct}/{total})")
            print(f"   Samples tested: {total}")
        
        # Save comparison results
        comparison_file = os.path.join(
            self.comparison_dir, 
            f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Comparison saved: {comparison_file}")
        
        return results

if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_comparison()
