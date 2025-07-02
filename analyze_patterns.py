
#!/usr/bin/env python3
"""
Pattern Analysis Runner
Executes pattern discovery and displays results
"""

from pattern_discovery import pattern_discovery
import json
from datetime import datetime

def main():
    """Run pattern discovery analysis"""
    print("🔍 Starting Pattern Discovery Analysis...")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = pattern_discovery.analyze_all_patterns()
    
    if not results:
        print("❌ No feedback data found to analyze")
        return
    
    print(f"📊 Analyzed {results['total_samples']} feedback samples")
    print()
    
    # Display filename patterns
    filename_analysis = results['patterns']['filename_analysis']
    print("🏷️ FILENAME PATTERNS:")
    print("-" * 20)
    
    ai_indicators = filename_analysis['ai_indicators']
    if ai_indicators:
        print("AI Content Indicators:")
        for keyword, count in list(ai_indicators.items())[:5]:
            print(f"  • '{keyword}': {count} occurrences")
    
    human_indicators = filename_analysis['human_indicators']
    if human_indicators:
        print("\nHuman Content Indicators:")
        for keyword, count in list(human_indicators.items())[:5]:
            print(f"  • '{keyword}': {count} occurrences")
    
    print()
    
    # Display confidence analysis
    confidence_analysis = results['patterns']['confidence_analysis']
    print("📈 CONFIDENCE ANALYSIS:")
    print("-" * 22)
    
    for bucket, data in confidence_analysis.items():
        if data['total'] > 0:
            print(f"{bucket.replace('_', ' ').title()}: {data['accuracy']:.1%} accuracy ({data['correct']}/{data['total']})")
    
    print()
    
    # Display misclassification patterns
    misclass_analysis = results['patterns']['misclassification_analysis']
    print("❌ MISCLASSIFICATION PATTERNS:")
    print("-" * 30)
    print(f"Total Errors: {misclass_analysis['total_errors']}")
    
    for error_type, count in misclass_analysis['error_types'].items():
        print(f"  • {error_type.replace('_', ' ').title()}: {count}")
    
    print()
    
    # Display file type analysis
    file_type_analysis = results['patterns']['file_type_analysis']
    print("📁 FILE TYPE PERFORMANCE:")
    print("-" * 25)
    
    for file_type, stats in file_type_analysis.items():
        if stats['total'] > 0:
            print(f"{file_type.title()}: {stats['accuracy']:.1%} accuracy ({stats['correct']}/{stats['total']})")
    
    print()
    
    # Display actionable insights
    insights = results['insights']
    print("💡 ACTIONABLE INSIGHTS:")
    print("-" * 21)
    
    for i, insight in enumerate(insights, 1):
        priority = insight['priority'].upper()
        print(f"{i}. [{priority}] {insight['insight']}")
        print(f"   Action: {insight['action']}")
        if 'confidence_boost' in insight:
            print(f"   Suggested boost: {insight['confidence_boost']}")
        print()
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"pattern_analysis/report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")
    print()
    print("🚀 Use these insights to improve your AI detection system!")

if __name__ == "__main__":
    main()
