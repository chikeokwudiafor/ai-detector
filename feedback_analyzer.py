
import json
import os
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
from config import IMAGE_MODELS, TEXT_MODELS

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """Analyzes user feedback to calculate model performance and suggest weight adjustments"""
    
    def __init__(self, feedback_file="feedback_data/user_feedback.json"):
        self.feedback_file = feedback_file
        self.model_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'accuracy': 0.0})
        
    def load_feedback_data(self):
        """Load and parse feedback data"""
        if not os.path.exists(self.feedback_file):
            logger.warning("No feedback file found")
            return []
            
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            logger.info(f"Loaded {len(feedback_data)} feedback entries")
            return feedback_data
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return []
    
    def analyze_model_performance(self):
        """Analyze feedback to calculate per-model accuracy"""
        feedback_data = self.load_feedback_data()
        
        # Reset stats
        self.model_stats.clear()
        
        # Categorize predictions by confidence level and accuracy
        prediction_mapping = {
            "🤖 Likely AI-Generated": "ai_generated",
            "⚠️ Possibly AI-Generated": "ai_generated", 
            "🤔 Unsure – Needs a Closer Look": "unsure",
            "🧠 Likely Human": "human_created",
            "✅ Almost Certainly Human": "human_created",
            "Likely AI-Generated": "ai_generated",
            "Likely Human-Created": "human_created",
            "Needs Manual Review": "unsure"
        }
        
        results = {
            'overall_accuracy': 0.0,
            'total_feedback': len(feedback_data),
            'ai_detection_accuracy': 0.0,
            'human_detection_accuracy': 0.0,
            'category_breakdown': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'model_insights': {}
        }
        
        correct_predictions = 0
        ai_correct = 0
        ai_total = 0
        human_correct = 0
        human_total = 0
        
        for entry in feedback_data:
            if 'true_label' not in entry or 'model_prediction' not in entry:
                continue
                
            prediction = entry['model_prediction']
            true_label = entry['true_label']
            
            # Map prediction to simplified category
            predicted_category = prediction_mapping.get(prediction, "unknown")
            
            # Count totals by true label
            if true_label == "ai_generated":
                ai_total += 1
            elif true_label == "human_created":
                human_total += 1
            
            # Check if prediction was correct
            is_correct = False
            if predicted_category == true_label:
                is_correct = True
                correct_predictions += 1
                
                if true_label == "ai_generated":
                    ai_correct += 1
                elif true_label == "human_created":
                    human_correct += 1
            
            # Track category performance
            category = f"{predicted_category}_for_{true_label}"
            results['category_breakdown'][category]['total'] += 1
            if is_correct:
                results['category_breakdown'][category]['correct'] += 1
        
        # Calculate accuracy metrics
        if len(feedback_data) > 0:
            results['overall_accuracy'] = correct_predictions / len(feedback_data)
        
        if ai_total > 0:
            results['ai_detection_accuracy'] = ai_correct / ai_total
        
        if human_total > 0:
            results['human_detection_accuracy'] = human_correct / human_total
        
        # Calculate model insights based on patterns
        results['model_insights'] = self._analyze_failure_patterns(feedback_data, prediction_mapping)
        
        logger.info(f"Analysis complete: {results['overall_accuracy']:.2%} overall accuracy")
        logger.info(f"AI Detection: {results['ai_detection_accuracy']:.2%}, Human Detection: {results['human_detection_accuracy']:.2%}")
        
        return results
    
    def _analyze_failure_patterns(self, feedback_data, prediction_mapping):
        """Analyze patterns in model failures"""
        patterns = {
            'organika_overconfident_human': 0,  # Organika said human, was AI
            'organika_underconfident_ai': 0,    # Organika missed AI
            'filename_ai_indicators_ignored': 0, # ChatGPT files called human
            'ensemble_vs_organika_conflicts': 0
        }
        
        for entry in feedback_data:
            if 'true_label' not in entry or 'model_prediction' not in entry:
                continue
                
            prediction = entry['model_prediction']
            true_label = entry['true_label']
            filename = entry.get('filename', '')
            
            predicted_category = prediction_mapping.get(prediction, "unknown")
            
            # Pattern 1: Organika said human but it was AI
            if (predicted_category == "human_created" and true_label == "ai_generated" and
                prediction in ["✅ Almost Certainly Human", "🧠 Likely Human"]):
                patterns['organika_overconfident_human'] += 1
            
            # Pattern 2: AI indicators in filename but called human
            ai_filename_indicators = ['chatgpt', 'gpt', 'dalle', 'midjourney', 'ai']
            if (any(indicator in filename.lower() for indicator in ai_filename_indicators) and
                predicted_category == "human_created" and true_label == "ai_generated"):
                patterns['filename_ai_indicators_ignored'] += 1
        
        return patterns
    
    def suggest_weight_adjustments(self, current_weights=None):
        """Suggest weight adjustments based on performance analysis"""
        analysis = self.analyze_model_performance()
        
        suggestions = {
            'weight_changes': {},
            'reasoning': [],
            'preserve_organika_override': True,  # Always preserve override
            'analysis_summary': analysis
        }
        
        # Get current model weights
        if current_weights is None:
            current_weights = {}
            for model in IMAGE_MODELS:
                current_weights[model['name']] = model['weight']
            for model in TEXT_MODELS:
                current_weights[model['name']] = model['weight']
        
        # RULE: Never change Organika override (≥95% confidence)
        organika_name = "Organika/sdxl-detector"
        if organika_name in current_weights:
            suggestions['weight_changes'][organika_name] = current_weights[organika_name]
            suggestions['reasoning'].append(f"✅ Preserved Organika weight at {current_weights[organika_name]} (override protection)")
        
        # Adjust other models based on performance
        overall_accuracy = analysis['overall_accuracy']
        ai_accuracy = analysis['ai_detection_accuracy']
        human_accuracy = analysis['human_detection_accuracy']
        
        # If overall AI detection is poor, boost non-Organika AI detection models
        if ai_accuracy < 0.7:  # Less than 70% AI detection
            for model in IMAGE_MODELS:
                if "Organika" not in model['name']:  # Skip Organika
                    old_weight = current_weights.get(model['name'], model['weight'])
                    new_weight = min(old_weight * 1.2, 3.0)  # Boost by 20%, cap at 3.0
                    suggestions['weight_changes'][model['name']] = new_weight
                    suggestions['reasoning'].append(f"📈 Boosted {model['name']}: {old_weight:.1f} → {new_weight:.1f} (poor AI detection)")
        
        # If human detection is poor, boost general classifiers
        if human_accuracy < 0.7:
            resnet_name = "microsoft/resnet-50"
            if resnet_name in current_weights:
                old_weight = current_weights[resnet_name]
                new_weight = min(old_weight * 1.15, 2.5)
                suggestions['weight_changes'][resnet_name] = new_weight
                suggestions['reasoning'].append(f"📈 Boosted ResNet-50: {old_weight:.1f} → {new_weight:.1f} (improve human detection)")
        
        # Check for filename indicator failures
        insights = analysis['model_insights']
        if insights.get('filename_ai_indicators_ignored', 0) > 3:
            suggestions['reasoning'].append("⚠️ Multiple ChatGPT files misclassified - filename heuristics may need strengthening")
        
        return suggestions

class AdaptiveWeightManager:
    """Manages automatic weight updates based on feedback analysis"""
    
    def __init__(self, weights_file="logs/adaptive_weights.json"):
        self.weights_file = weights_file
        self.analyzer = FeedbackAnalyzer()
        self.last_update = None
        
    def get_updated_weights(self):
        """Get current adaptive weights or generate new ones"""
        # Check if we have recent weights
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                self.last_update = data.get('last_update')
                return data.get('weights', {})
            except Exception as e:
                logger.warning(f"Error loading adaptive weights: {e}")
        
        return {}
    
    def update_weights_from_feedback(self, force_update=False):
        """Update weights based on latest feedback"""
        # Only update if we have enough new feedback or forced
        feedback_data = self.analyzer.load_feedback_data()
        
        if len(feedback_data) < 10 and not force_update:
            logger.info("Not enough feedback data for weight update")
            return None
        
        # Get current weights
        current_weights = self.get_updated_weights()
        
        # Get suggestions
        suggestions = self.analyzer.suggest_weight_adjustments(current_weights)
        
        # Apply suggestions (but preserve Organika override)
        updated_weights = suggestions['weight_changes']
        
        # Save updated weights
        weights_data = {
            'weights': updated_weights,
            'last_update': datetime.now().isoformat(),
            'feedback_count': len(feedback_data),
            'analysis': suggestions['analysis_summary'],
            'reasoning': suggestions['reasoning']
        }
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        
        with open(self.weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        logger.info(f"Updated adaptive weights based on {len(feedback_data)} feedback entries")
        for reason in suggestions['reasoning']:
            logger.info(f"Weight change: {reason}")
        
        return updated_weights

# Global instance
adaptive_weight_manager = AdaptiveWeightManager()
