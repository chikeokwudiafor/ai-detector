
#!/usr/bin/env python3
"""
Update Model Weights Based on User Feedback
Run this periodically to adapt model weights based on recent performance
"""

from adaptive_weighting import adaptive_weights
from fine_tune_model import feedback_tuner

def main():
    print("🔄 Updating model weights based on feedback...")
    print("=" * 50)
    
    # Generate new adaptive weights
    new_weights = adaptive_weights.generate_adaptive_weights()
    
    if new_weights:
        print("\n📊 New Adaptive Weights:")
        for model_name, weight in new_weights.items():
            print(f"  {model_name}: {weight:.2f}")
        
        print("\n✅ Weights updated! Changes will take effect on next model reload.")
        print("\n💡 Tip: Monitor performance and run this script weekly for best results.")
        
        # Also run the fine-tuning analysis
        print("\n🧠 Running additional fine-tuning analysis...")
        feedback_tuner.suggest_weight_adjustments()
        
    else:
        print("\n❌ Could not update weights - need more feedback data")
        print("   Upload test images and provide feedback to improve accuracy!")

if __name__ == "__main__":
    main()
