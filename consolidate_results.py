
#!/usr/bin/env python3
"""
Results Consolidation Script
Run this to consolidate all scattered result files into organized structure
"""

from results_manager import results_manager
import os

def main():
    print("🔧 AI Detector Results Consolidation")
    print("=" * 50)
    
    print("\n📋 Current scattered files detected:")
    scattered_files = [
        "fine_tuning_results.json",
        "last_weight_update.json", 
        "pattern_analysis/discovered_patterns.json",
        "pattern_analysis/report_20250702_091533.json",
        "pattern_analysis/report_20250702_092452.json"
    ]
    
    existing_files = []
    for file_path in scattered_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({file_size} bytes)")
            existing_files.append(file_path)
        else:
            print(f"   ❌ {file_path} (not found)")
    
    if not existing_files:
        print("\n🎉 No scattered files found - system is already clean!")
        return
    
    print(f"\n🔄 Consolidating {len(existing_files)} files...")
    
    # Run consolidation
    results_manager.consolidate_existing_files()
    
    # Generate readable report
    print("\n📊 Generating readable report...")
    report_path = results_manager.generate_readable_report()
    print(f"   Report saved: {report_path}")
    
    # Show summary
    summary = results_manager.get_latest_summary()
    overview = summary.get("overview", {})
    
    print(f"\n📈 Consolidated Results Summary:")
    print(f"   - Total feedback samples: {overview.get('total_feedback_samples', 0)}")
    print(f"   - Latest accuracy: {overview.get('latest_accuracy', 0):.1%}")
    print(f"   - Pattern files processed: {overview.get('total_pattern_files', 0)}")
    print(f"   - Weight updates: {overview.get('total_weight_updates', 0)}")
    
    print(f"\n✅ All results now organized in: model_results/")
    print("   📁 consolidated_results.json - Complete data")
    print("   📁 performance_summary.json - Quick overview") 
    print("   📁 readable_report.md - Human-readable report")
    
    # Offer cleanup
    print(f"\n🗑️  Old files can be cleaned up:")
    for file_path in existing_files:
        print(f"   - {file_path}")
    
    cleanup = input("\nRemove old files? (creates backup first) [y/N]: ").lower().strip()
    if cleanup == 'y':
        results_manager.cleanup_old_files(backup=True)
        print("✅ Cleanup complete with backup!")
    else:
        print("📦 Old files preserved - you can remove them manually later")
    
    print(f"\n🎯 Next steps:")
    print("   1. Check model_results/readable_report.md for insights")
    print("   2. Use results_manager.get_latest_summary() in your code")
    print("   3. All future results will be automatically organized")

if __name__ == "__main__":
    main()
