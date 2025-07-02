
"""
Pattern Discovery System for AI Detection
Analyzes feedback data to discover patterns for building custom models
"""

import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatternDiscovery:
    """Discovers patterns from user feedback to improve detection accuracy"""
    
    def __init__(self, feedback_file="feedback_data/user_feedback.json"):
        self.feedback_file = feedback_file
        self.patterns = {
            'filename_patterns': defaultdict(list),
            'text_patterns': defaultdict(list),
            'confidence_patterns': defaultdict(list),
            'misclassified_patterns': defaultdict(list)
        }
        
    def analyze_all_patterns(self):
        """Run comprehensive pattern analysis"""
        feedback_data = self._load_feedback_data()
        if not feedback_data:
            return None
            
        patterns = {
            'filename_analysis': self._analyze_filename_patterns(feedback_data),
            'confidence_analysis': self._analyze_confidence_patterns(feedback_data),
            'misclassification_analysis': self._analyze_misclassifications(feedback_data),
            'temporal_analysis': self._analyze_temporal_patterns(feedback_data),
            'file_type_analysis': self._analyze_file_type_patterns(feedback_data)
        }
        
        # Generate actionable insights
        insights = self._generate_insights(patterns)
        
        # Save patterns for future use
        self._save_patterns(patterns, insights)
        
        return {
            'patterns': patterns,
            'insights': insights,
            'total_samples': len(feedback_data)
        }
    
    def _load_feedback_data(self):
        """Load feedback data from JSON file"""
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return []
    
    def _analyze_filename_patterns(self, feedback_data):
        """Discover filename patterns that correlate with AI/human content"""
        ai_filenames = []
        human_filenames = []
        
        for entry in feedback_data:
            filename = entry.get('filename', '').lower()
            true_label = entry.get('true_label', '')
            
            if true_label == 'ai_generated':
                ai_filenames.append(filename)
            elif true_label == 'human_created':
                human_filenames.append(filename)
        
        # Extract keywords and patterns
        ai_keywords = self._extract_keywords(ai_filenames)
        human_keywords = self._extract_keywords(human_filenames)
        
        # Find AI-specific patterns
        ai_specific = {k: v for k, v in ai_keywords.items() if k not in human_keywords or v > human_keywords[k] * 2}
        human_specific = {k: v for k, v in human_keywords.items() if k not in ai_keywords or v > ai_keywords[k] * 2}
        
        return {
            'ai_indicators': dict(sorted(ai_specific.items(), key=lambda x: x[1], reverse=True)[:10]),
            'human_indicators': dict(sorted(human_specific.items(), key=lambda x: x[1], reverse=True)[:10]),
            'total_ai_files': len(ai_filenames),
            'total_human_files': len(human_filenames)
        }
    
    def _extract_keywords(self, filenames):
        """Extract keywords from filenames"""
        keywords = Counter()
        
        for filename in filenames:
            # Split on common separators
            words = re.split(r'[_\-\s\.]+', filename.lower())
            
            # Filter meaningful words (longer than 2 characters)
            meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
            
            keywords.update(meaningful_words)
        
        return keywords
    
    def _analyze_confidence_patterns(self, feedback_data):
        """Analyze confidence score patterns vs actual accuracy"""
        confidence_buckets = {
            'high_confidence': [],  # >0.8
            'medium_confidence': [], # 0.4-0.8
            'low_confidence': []     # <0.4
        }
        
        for entry in feedback_data:
            model_pred = entry.get('model_prediction', '').lower()
            true_label = entry.get('true_label', '')
            
            # Estimate confidence from prediction text
            confidence = self._estimate_confidence_from_prediction(model_pred)
            
            # Determine if prediction was correct
            is_correct = self._is_prediction_correct(model_pred, true_label)
            
            bucket_data = {
                'filename': entry.get('filename'),
                'prediction': model_pred,
                'true_label': true_label,
                'is_correct': is_correct,
                'estimated_confidence': confidence
            }
            
            if confidence > 0.8:
                confidence_buckets['high_confidence'].append(bucket_data)
            elif confidence > 0.4:
                confidence_buckets['medium_confidence'].append(bucket_data)
            else:
                confidence_buckets['low_confidence'].append(bucket_data)
        
        # Calculate accuracy per bucket
        analysis = {}
        for bucket, data in confidence_buckets.items():
            if data:
                correct = sum(1 for d in data if d['is_correct'])
                analysis[bucket] = {
                    'total': len(data),
                    'correct': correct,
                    'accuracy': correct / len(data),
                    'samples': data[:3]  # Include sample data
                }
            else:
                analysis[bucket] = {'total': 0, 'correct': 0, 'accuracy': 0, 'samples': []}
        
        return analysis
    
    def _analyze_misclassifications(self, feedback_data):
        """Identify patterns in misclassified samples"""
        misclassified = []
        
        for entry in feedback_data:
            model_pred = entry.get('model_prediction', '').lower()
            true_label = entry.get('true_label', '')
            
            if not self._is_prediction_correct(model_pred, true_label):
                misclassified.append({
                    'filename': entry.get('filename'),
                    'file_type': entry.get('file_type'),
                    'model_prediction': model_pred,
                    'true_label': true_label,
                    'error_type': self._classify_error_type(model_pred, true_label)
                })
        
        # Group by error patterns
        error_patterns = defaultdict(list)
        for error in misclassified:
            error_patterns[error['error_type']].append(error)
        
        # Analyze filename patterns in errors
        filename_error_patterns = self._analyze_error_filename_patterns(misclassified)
        
        return {
            'total_errors': len(misclassified),
            'error_types': {k: len(v) for k, v in error_patterns.items()},
            'error_samples': {k: v[:3] for k, v in error_patterns.items()},
            'filename_patterns': filename_error_patterns
        }
    
    def _analyze_temporal_patterns(self, feedback_data):
        """Analyze patterns over time"""
        daily_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'ai_labels': 0, 'human_labels': 0})
        
        for entry in feedback_data:
            timestamp = entry.get('timestamp', '')
            if timestamp:
                date = timestamp.split('T')[0]  # Extract date part
                
                daily_stats[date]['total'] += 1
                
                if self._is_prediction_correct(entry.get('model_prediction', ''), entry.get('true_label', '')):
                    daily_stats[date]['correct'] += 1
                
                if entry.get('true_label') == 'ai_generated':
                    daily_stats[date]['ai_labels'] += 1
                else:
                    daily_stats[date]['human_labels'] += 1
        
        # Calculate daily accuracy
        for date, stats in daily_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
        
        return dict(daily_stats)
    
    def _analyze_file_type_patterns(self, feedback_data):
        """Analyze patterns by file type"""
        file_type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'ai_true': 0, 'human_true': 0})
        
        for entry in feedback_data:
            file_type = entry.get('file_type', 'unknown')
            true_label = entry.get('true_label', '')
            
            file_type_stats[file_type]['total'] += 1
            
            if self._is_prediction_correct(entry.get('model_prediction', ''), true_label):
                file_type_stats[file_type]['correct'] += 1
            
            if true_label == 'ai_generated':
                file_type_stats[file_type]['ai_true'] += 1
            else:
                file_type_stats[file_type]['human_true'] += 1
        
        # Calculate accuracy per file type
        for file_type, stats in file_type_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
        
        return dict(file_type_stats)
    
    def _estimate_confidence_from_prediction(self, prediction):
        """Estimate confidence level from prediction text"""
        prediction = prediction.lower()
        
        if 'likely' in prediction and 'ai' in prediction:
            return 0.85
        elif 'possibly' in prediction:
            return 0.65
        elif 'likely' in prediction and 'human' in prediction:
            return 0.85
        elif 'unsure' in prediction or 'review' in prediction:
            return 0.5
        elif 'almost certainly' in prediction:
            return 0.95
        else:
            return 0.6  # Default moderate confidence
    
    def _is_prediction_correct(self, model_pred, true_label):
        """Check if model prediction matches true label"""
        model_pred = model_pred.lower()
        
        if true_label == 'ai_generated':
            return 'ai' in model_pred and 'human' not in model_pred
        elif true_label == 'human_created':
            return 'human' in model_pred and 'ai' not in model_pred
        
        return False
    
    def _classify_error_type(self, model_pred, true_label):
        """Classify the type of error"""
        if true_label == 'ai_generated' and 'human' in model_pred:
            return 'false_negative'  # AI content classified as human
        elif true_label == 'human_created' and 'ai' in model_pred:
            return 'false_positive'  # Human content classified as AI
        else:
            return 'uncertain_classification'
    
    def _analyze_error_filename_patterns(self, misclassified):
        """Find patterns in filenames of misclassified samples"""
        false_positives = [e for e in misclassified if e['error_type'] == 'false_positive']
        false_negatives = [e for e in misclassified if e['error_type'] == 'false_negative']
        
        fp_keywords = self._extract_keywords([e['filename'] for e in false_positives])
        fn_keywords = self._extract_keywords([e['filename'] for e in false_negatives])
        
        return {
            'false_positive_patterns': dict(fp_keywords.most_common(5)),
            'false_negative_patterns': dict(fn_keywords.most_common(5))
        }
    
    def _generate_insights(self, patterns):
        """Generate actionable insights from discovered patterns"""
        insights = []
        
        # Filename insights
        filename_analysis = patterns.get('filename_analysis', {})
        ai_indicators = filename_analysis.get('ai_indicators', {})
        
        if ai_indicators:
            top_ai_keyword = list(ai_indicators.keys())[0]
            insights.append({
                'type': 'filename_heuristic',
                'priority': 'high',
                'insight': f"Keyword '{top_ai_keyword}' appears {ai_indicators[top_ai_keyword]} times in AI-generated files",
                'action': f"Add '{top_ai_keyword}' to AI keyword detection list",
                'confidence_boost': 1.5
            })
        
        # Confidence insights
        confidence_analysis = patterns.get('confidence_analysis', {})
        high_conf = confidence_analysis.get('high_confidence', {})
        
        if high_conf.get('accuracy', 0) < 0.8:
            insights.append({
                'type': 'confidence_calibration',
                'priority': 'medium',
                'insight': f"High confidence predictions only {high_conf['accuracy']:.1%} accurate",
                'action': 'Reduce confidence thresholds or add more validation'
            })
        
        # Error pattern insights
        misclass_analysis = patterns.get('misclassification_analysis', {})
        filename_patterns = misclass_analysis.get('filename_patterns', {})
        
        for pattern_type, keywords in filename_patterns.items():
            if keywords:
                top_keyword = list(keywords.keys())[0]
                insights.append({
                    'type': 'error_pattern',
                    'priority': 'medium',
                    'insight': f"Files with '{top_keyword}' often misclassified as {pattern_type}",
                    'action': f"Create specific rule for '{top_keyword}' pattern"
                })
        
        return insights
    
    def _save_patterns(self, patterns, insights):
        """Save discovered patterns to file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'patterns': patterns,
            'insights': insights,
            'metadata': {
                'analysis_version': '1.0',
                'total_patterns': len(insights)
            }
        }
        
        os.makedirs('pattern_analysis', exist_ok=True)
        
        with open('pattern_analysis/discovered_patterns.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(insights)} insights to pattern_analysis/discovered_patterns.json")

# Global pattern discovery instance
pattern_discovery = PatternDiscovery()
