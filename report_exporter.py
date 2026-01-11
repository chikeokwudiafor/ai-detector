
import json
import csv
import os
from datetime import datetime
from feedback_analyzer import FeedbackAnalyzer

class ReportExporter:
    """Export feedback data and accuracy reports to CSV and Markdown formats"""
    
    def __init__(self):
        self.analyzer = FeedbackAnalyzer()
    
    def export_feedback_to_csv(self, output_file="reports/feedback_data.csv"):
        """Export all feedback data to CSV format"""
        feedback_data = self.analyzer.load_feedback_data()
        
        if not feedback_data:
            print("No feedback data found")
            return None
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Define CSV headers
        headers = [
            'timestamp',
            'session_id', 
            'filename',
            'file_type',
            'model_prediction',
            'true_label',
            'is_correct',
            'error_type'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for entry in feedback_data:
                # Determine if prediction was correct
                is_correct = self.analyzer._is_prediction_correct(entry)
                
                # Determine error type if incorrect
                error_type = ""
                if not is_correct:
                    prediction = entry.get('model_prediction', '') or entry.get('model_result', '')
                    true_label = entry.get('true_label', '')
                    predicted_category = self.analyzer._get_predicted_category(prediction)
                    error_type = f"{true_label}_predicted_as_{predicted_category}"
                
                row = [
                    entry.get('timestamp', ''),
                    entry.get('session_id', ''),
                    entry.get('filename', ''),
                    entry.get('file_type', ''),
                    entry.get('model_prediction', '') or entry.get('model_result', ''),
                    entry.get('true_label', ''),
                    is_correct,
                    error_type
                ]
                writer.writerow(row)
        
        print(f"✅ Exported {len(feedback_data)} feedback entries to {output_file}")
        return output_file
    
    def export_accuracy_report_to_markdown(self, output_file="reports/accuracy_report.md"):
        """Export comprehensive accuracy report to Markdown format"""
        report = self.analyzer.generate_comprehensive_report()
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        md_content = self._generate_markdown_report(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ Exported accuracy report to {output_file}")
        return output_file
    
    def _generate_markdown_report(self, report):
        """Generate markdown content from report data"""
        md = []
        
        # Header
        md.append("# AI Detection System - Accuracy Report")
        md.append("")
        md.append(f"**Generated:** {report['report_metadata']['generated_at']}")
        md.append(f"**Total Feedback Entries:** {report['report_metadata']['total_feedback_entries']}")
        md.append("")
        
        # Summary Section
        md.append("## 📊 Performance Summary")
        md.append("")
        summary = report['summary']
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Overall Accuracy | {summary['overall_accuracy']:.1%} |")
        md.append(f"| AI Detection Accuracy | {summary['ai_detection_accuracy']:.1%} |")
        md.append(f"| Human Detection Accuracy | {summary['human_detection_accuracy']:.1%} |")
        md.append(f"| Total Feedback Entries | {summary['total_feedback']} |")
        md.append("")
        
        # Performance Trends
        if 'performance_trends' in report and 'status' not in report['performance_trends']:
            md.append("## 📈 Performance Trends")
            md.append("")
            trends = report['performance_trends']
            md.append(f"- **Recent Accuracy:** {trends['recent_accuracy']:.1%} ({trends['recent_sample_size']} samples)")
            md.append(f"- **Historical Accuracy:** {trends['older_accuracy']:.1%} ({trends['older_sample_size']} samples)")
            md.append(f"- **Trend Direction:** {trends['trend_direction'].title()}")
            md.append(f"- **Change Magnitude:** {trends['trend_magnitude']:.1%}")
            md.append("")
        
        # Problem Patterns
        if 'problem_patterns' in report:
            md.append("## ⚠️ Problem Patterns")
            md.append("")
            patterns = report['problem_patterns']
            
            if patterns['ai_called_human_errors'] > 0:
                md.append(f"- **AI called Human (high confidence):** {patterns['ai_called_human_errors']} errors")
            if patterns['human_called_ai_errors'] > 0:
                md.append(f"- **Human called AI:** {patterns['human_called_ai_errors']} errors")
            if patterns['chatgpt_filename_misses']:
                md.append(f"- **ChatGPT filename misses:** {len(patterns['chatgpt_filename_misses'])} files")
            
            # Consistent failures
            if patterns['consistent_failures']:
                md.append("")
                md.append("### Most Common Error Types")
                md.append("")
                for error_type, count in sorted(patterns['consistent_failures'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    md.append(f"- **{error_type.replace('_', ' ').title()}:** {count} occurrences")
            md.append("")
        
        # Recommendations
        if 'improvement_recommendations' in report and report['improvement_recommendations']:
            md.append("## 🎯 Recommendations")
            md.append("")
            for rec in report['improvement_recommendations']:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                md.append(f"### {priority_emoji} {rec['category'].replace('_', ' ').title()}")
                md.append(f"**Issue:** {rec['issue']}")
                md.append(f"**Recommendation:** {rec['recommendation']}")
                md.append("")
        
        # Timeline
        if 'feedback_timeline' in report and report['feedback_timeline']:
            md.append("## 📅 Feedback Timeline")
            md.append("")
            md.append("| Date | Feedback Count | Accuracy | Sample Entries |")
            md.append("|------|----------------|----------|----------------|")
            
            for day in report['feedback_timeline'][-10:]:  # Last 10 days
                date = day['date']
                count = day['total_feedback']
                accuracy = f"{day['daily_accuracy']:.1%}"
                sample_files = ", ".join([entry.get('filename', 'unknown')[:20] for entry in day['sample_entries'][:2]])
                md.append(f"| {date} | {count} | {accuracy} | {sample_files}... |")
            md.append("")
        
        # Model Insights
        if 'model_insights' in report:
            md.append("## 🔍 Model Insights")
            md.append("")
            insights = report['model_insights']
            if insights.get('organika_overconfident_human', 0) > 0:
                md.append(f"- Organika overconfident on human classifications: {insights['organika_overconfident_human']} cases")
            if insights.get('filename_ai_indicators_ignored', 0) > 0:
                md.append(f"- ChatGPT filename indicators ignored: {insights['filename_ai_indicators_ignored']} cases")
            md.append("")
        
        # Footer
        md.append("---")
        md.append("*Report generated by AIDetector Feedback Analysis System*")
        
        return "\n".join(md)
    
    def export_analytics_to_csv(self, output_file="reports/analytics_data.csv"):
        """Export analytics data to CSV format"""
        analytics_file = "analytics/user_activity.json"
        
        if not os.path.exists(analytics_file):
            print("No analytics data found")
            return None
        
        # Read analytics data (JSONL format)
        analytics_data = []
        try:
            with open(analytics_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        analytics_data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading analytics: {e}")
            return None
        
        if not analytics_data:
            print("No analytics entries found")
            return None
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Define CSV headers
        headers = [
            'timestamp',
            'event_type',
            'ip_address',
            'user_agent',
            'referrer',
            'page',
            'filename',
            'file_type',
            'result_type',
            'confidence'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for entry in analytics_data:
                data = entry.get('data', {})
                
                row = [
                    entry.get('timestamp', ''),
                    entry.get('event_type', ''),
                    entry.get('ip_address', ''),
                    entry.get('user_agent', ''),
                    entry.get('referrer', ''),
                    data.get('page', ''),
                    data.get('filename', ''),
                    data.get('file_type', ''),
                    data.get('result_type', ''),
                    data.get('confidence', '')
                ]
                writer.writerow(row)
        
        print(f"✅ Exported {len(analytics_data)} analytics entries to {output_file}")
        return output_file

def export_all_reports():
    """Export all available reports to CSV and Markdown"""
    exporter = ReportExporter()
    
    print("🔄 Exporting all reports...")
    
    # Export feedback to CSV
    feedback_csv = exporter.export_feedback_to_csv()
    
    # Export accuracy report to Markdown
    accuracy_md = exporter.export_accuracy_report_to_markdown()
    
    # Export analytics to CSV
    analytics_csv = exporter.export_analytics_to_csv()
    
    print("✅ All exports completed!")
    return {
        'feedback_csv': feedback_csv,
        'accuracy_md': accuracy_md,
        'analytics_csv': analytics_csv
    }

if __name__ == "__main__":
    export_all_reports()
