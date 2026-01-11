
import json
import os
import logging
from datetime import datetime, timedelta
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

    def generate_comprehensive_report(self):
        """Generate a comprehensive accuracy report with trends and insights"""
        feedback_data = self.load_feedback_data()
        analysis = self.analyze_model_performance()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_feedback_entries': len(feedback_data),
                'report_version': '1.0',
                'analysis_period': 'all_time'
            },
            'summary': {
                'overall_accuracy': analysis['overall_accuracy'],
                'ai_detection_accuracy': analysis['ai_detection_accuracy'], 
                'human_detection_accuracy': analysis['human_detection_accuracy'],
                'total_feedback': analysis['total_feedback']
            },
            'performance_trends': self._analyze_performance_trends(feedback_data),
            'model_insights': analysis['model_insights'],
            'problem_patterns': self._identify_problem_patterns(feedback_data),
            'improvement_recommendations': self._generate_recommendations(analysis),
            'detailed_breakdown': analysis['category_breakdown'],
            'feedback_timeline': self._create_feedback_timeline(feedback_data)
        }
        
        return report
    
    def _analyze_performance_trends(self, feedback_data):
        """Analyze performance trends over time"""
        if len(feedback_data) < 5:
            return {'status': 'insufficient_data', 'message': 'Need at least 5 entries for trend analysis'}
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback_data, key=lambda x: x.get('timestamp', ''))
        
        # Split into time windows
        recent_count = min(15, len(sorted_feedback))
        older_count = max(15, len(sorted_feedback) - recent_count)
        
        recent_feedback = sorted_feedback[-recent_count:]
        older_feedback = sorted_feedback[:older_count] if older_count > 0 else []
        
        def calculate_accuracy(entries):
            if not entries:
                return 0.0
            correct = 0
            for entry in entries:
                if self._is_prediction_correct(entry):
                    correct += 1
            return correct / len(entries)
        
        recent_accuracy = calculate_accuracy(recent_feedback)
        older_accuracy = calculate_accuracy(older_feedback) if older_feedback else recent_accuracy
        
        trend_direction = "improving" if recent_accuracy > older_accuracy else "declining" if recent_accuracy < older_accuracy else "stable"
        trend_magnitude = abs(recent_accuracy - older_accuracy)
        
        return {
            'recent_accuracy': recent_accuracy,
            'older_accuracy': older_accuracy,
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude,
            'recent_sample_size': recent_count,
            'older_sample_size': len(older_feedback)
        }
    
    def _identify_problem_patterns(self, feedback_data):
        """Identify specific problem patterns in the feedback"""
        patterns = {
            'chatgpt_filename_misses': [],
            'high_confidence_errors': [],
            'consistent_failures': defaultdict(int),
            'ai_called_human_errors': 0,
            'human_called_ai_errors': 0
        }
        
        for entry in feedback_data:
            filename = entry.get('filename', '').lower()
            prediction = entry.get('model_prediction', '') or entry.get('model_result', '')
            true_label = entry.get('true_label', '')
            
            # ChatGPT filename misses
            if 'chatgpt' in filename and true_label == 'ai_generated' and not self._is_prediction_correct(entry):
                patterns['chatgpt_filename_misses'].append({
                    'filename': entry.get('filename'),
                    'prediction': prediction,
                    'timestamp': entry.get('timestamp')
                })
            
            # High confidence errors (Almost Certainly Wrong)
            if prediction in ['✅ Almost Certainly Human', '🧠 Likely Human'] and true_label == 'ai_generated':
                patterns['high_confidence_errors'].append({
                    'filename': entry.get('filename'),
                    'prediction': prediction,
                    'true_label': true_label,
                    'error_type': 'AI called Human with high confidence'
                })
                patterns['ai_called_human_errors'] += 1
            
            if prediction in ['🤖 Likely AI-Generated'] and true_label == 'human_created':
                patterns['high_confidence_errors'].append({
                    'filename': entry.get('filename'),
                    'prediction': prediction,
                    'true_label': true_label,
                    'error_type': 'Human called AI'
                })
                patterns['human_called_ai_errors'] += 1
            
            # Consistent failure types
            if not self._is_prediction_correct(entry):
                error_type = f"{true_label}_predicted_as_{self._get_predicted_category(prediction)}"
                patterns['consistent_failures'][error_type] += 1
        
        return patterns
    
    def _generate_recommendations(self, analysis):
        """Generate specific improvement recommendations"""
        recommendations = []
        
        overall_acc = analysis['overall_accuracy']
        ai_acc = analysis['ai_detection_accuracy']
        human_acc = analysis['human_detection_accuracy']
        insights = analysis['model_insights']
        
        # Overall accuracy recommendations
        if overall_acc < 0.7:
            recommendations.append({
                'priority': 'high',
                'category': 'overall_performance',
                'issue': f"Overall accuracy is {overall_acc:.1%}, below 70% threshold",
                'recommendation': "Consider retraining models or adjusting ensemble weights"
            })
        
        # AI detection recommendations
        if ai_acc < 0.75:
            recommendations.append({
                'priority': 'high',
                'category': 'ai_detection',
                'issue': f"AI detection accuracy is {ai_acc:.1%}, missing too many AI images",
                'recommendation': "Increase weights for AI-specialized models, strengthen filename heuristics"
            })
        
        # Human detection recommendations
        if human_acc < 0.75:
            recommendations.append({
                'priority': 'medium',
                'category': 'human_detection', 
                'issue': f"Human detection accuracy is {human_acc:.1%}, too many false positives",
                'recommendation': "Reduce AI detection aggressiveness, improve human confidence boosting"
            })
        
        # ChatGPT filename issues
        if insights.get('filename_ai_indicators_ignored', 0) > 2:
            recommendations.append({
                'priority': 'high',
                'category': 'filename_heuristics',
                'issue': f"{insights['filename_ai_indicators_ignored']} ChatGPT files misclassified",
                'recommendation': "Strengthen filename semantic analysis for AI indicators"
            })
        
        # Organika specific
        if insights.get('organika_overconfident_human', 0) > 3:
            recommendations.append({
                'priority': 'medium',
                'category': 'organika_tuning',
                'issue': f"Organika incorrectly confident about {insights['organika_overconfident_human']} human classifications",
                'recommendation': "Review Organika's human classification threshold"
            })
        
        return recommendations
    
    def _create_feedback_timeline(self, feedback_data):
        """Create a timeline of feedback entries with accuracy"""
        timeline = []
        
        # Group by day
        daily_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'entries': []})
        
        for entry in feedback_data:
            timestamp = entry.get('timestamp', '')
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                    date_str = date.isoformat()
                    
                    daily_stats[date_str]['total'] += 1
                    daily_stats[date_str]['entries'].append(entry)
                    
                    if self._is_prediction_correct(entry):
                        daily_stats[date_str]['correct'] += 1
                        
                except ValueError:
                    continue
        
        # Convert to timeline format
        for date_str, stats in sorted(daily_stats.items()):
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            timeline.append({
                'date': date_str,
                'total_feedback': stats['total'],
                'correct_predictions': stats['correct'],
                'daily_accuracy': accuracy,
                'sample_entries': stats['entries'][:3]  # First 3 entries as examples
            })
        
        return timeline
    
    def _is_prediction_correct(self, entry):
        """Check if a prediction was correct"""
        prediction = entry.get('model_prediction', '') or entry.get('model_result', '')
        true_label = entry.get('true_label', '')
        
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
        
        predicted_category = prediction_mapping.get(prediction, "unknown")
        return predicted_category == true_label
    
    def _get_predicted_category(self, prediction):
        """Get simplified predicted category"""
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
        return prediction_mapping.get(prediction, "unknown")
    
    def save_report_to_file(self, report=None):
        """Save comprehensive report to a file"""
        if report is None:
            report = self.generate_comprehensive_report()
        
        # Ensure reports directory exists
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(reports_dir, f"accuracy_report_{timestamp}.json")
        
        # Also save as latest report
        latest_report_file = os.path.join(reports_dir, "latest_accuracy_report.json")
        
        try:
            # Save timestamped report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save as latest
            with open(latest_report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Accuracy report saved to {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return None

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
    
    def generate_and_save_report(self):
        """Generate and save accuracy report"""
        try:
            report = self.analyzer.generate_comprehensive_report()
            report_file = self.analyzer.save_report_to_file(report)
            
            if report_file:
                logger.info(f"📊 Accuracy report generated: {report_file}")
                return report_file
            else:
                logger.error("Failed to generate accuracy report")
                return None
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None

class ContinuousFeedbackMonitor:
    """Monitors feedback continuously and triggers reports/updates"""
    
    def __init__(self):
        self.analyzer = FeedbackAnalyzer()
        self.weight_manager = AdaptiveWeightManager()
        self.last_feedback_count = 0
        self.last_report_time = None
        
    def check_for_updates(self):
        """Check if new feedback requires processing"""
        feedback_data = self.analyzer.load_feedback_data()
        current_count = len(feedback_data)
        
        # Check if we have new feedback
        if current_count > self.last_feedback_count:
            new_entries = current_count - self.last_feedback_count
            logger.info(f"📥 {new_entries} new feedback entries detected (total: {current_count})")
            
            # Generate report for every 5 new entries, or daily
            should_generate_report = (
                new_entries >= 5 or 
                self._should_generate_daily_report() or
                self.last_report_time is None
            )
            
            if should_generate_report:
                self._process_feedback_update(current_count)
            
            self.last_feedback_count = current_count
            
        return current_count
    
    def _should_generate_daily_report(self):
        """Check if daily report should be generated"""
        if self.last_report_time is None:
            return True
        
        time_since_last = datetime.now() - self.last_report_time
        return time_since_last > timedelta(hours=24)
    
    def _process_feedback_update(self, feedback_count):
        """Process feedback updates - generate report and update weights"""
        logger.info(f"🔄 Processing feedback update with {feedback_count} total entries")
        
        try:
            # Generate comprehensive report
            report_file = self.weight_manager.generate_and_save_report()
            
            # Update weights if enough feedback
            if feedback_count >= 10:
                updated_weights = self.weight_manager.update_weights_from_feedback()
                if updated_weights:
                    logger.info("✅ Adaptive weights updated based on latest feedback")
            
            self.last_report_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing feedback update: {e}")

# Global instances
adaptive_weight_manager = AdaptiveWeightManager()
continuous_monitor = ContinuousFeedbackMonitor()
