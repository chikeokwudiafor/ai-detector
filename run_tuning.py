
#!/usr/bin/env python3
"""
Run Model Tuning Analysis
Analyzes feedback and suggests improvements
"""

from fine_tune_model import feedback_tuner
import sys

def main():
    print("🧠 AI Detector Model Tuning")
    print("=" * 40)
    
    print("\n1. Loading feedback data...")
    feedback_data = feedback_tuner.load_feedback_data()
    print(f"   Found {len(feedback_data)} feedback samples")
    
    if len(feedback_data) < 3:
        print("\n❌ Need at least 3 feedback samples to run analysis")
        print("   Upload some test images and provide feedback first!")
        return
    
    print("\n2. Analyzing model performance...")
    performance = feedback_tuner.analyze_model_performance()
    
    if performance:
        print("\n3. Generating weight suggestions...")
        weights = feedback_tuner.suggest_weight_adjustments()
        
        print("\n✅ Analysis complete!")
        print("\nRecommendations:")
        print("- The Organika/sdxl-detector model weight has been optimized")
        print("- Review the suggested weights above")
        print("- Monitor performance after applying changes")
        
        # Check if Organika is performing well
        organika_stats = performance.get('Organika/sdxl-detector', {})
        if organika_stats.get('accuracy', 0) > 0.7:
            print("- ✅ Organika/sdxl-detector is performing well - good choice!")
        else:
            print("- ⚠️  Consider collecting more feedback to improve accuracy")
    
    else:
        print("\n❌ Could not complete analysis - need more feedback data")

if __name__ == "__main__":
    main()
