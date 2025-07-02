
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class ResultsManager:
    """Unified management for all model results, feedback, and analysis data"""
    
    def __init__(self):
        self.results_dir = "model_results"
        self.consolidated_file = os.path.join(self.results_dir, "consolidated_results.json")
        self.summary_file = os.path.join(self.results_dir, "performance_summary.json")
        self._ensure_results_dir()
        self._init_consolidated_file()
    
    def _ensure_results_dir(self):
        """Create results directory if it doesn't exist"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def _init_consolidated_file(self):
        """Initialize consolidated results file"""
        if not os.path.exists(self.consolidated_file):
            initial_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "model_performance": {},
                "feedback_analysis": {},
                "pattern_discoveries": {},
                "weight_adjustments": [],
                "tuning_history": []
            }
            with open(self.consolidated_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def consolidate_existing_files(self):
        """Consolidate all existing result files into unified format"""
        print("🔄 Consolidating existing result files...")
        
        consolidated_data = self._load_consolidated_data()
        
        # Files to consolidate
        files_to_process = {
            "fine_tuning_results.json": "model_performance",
            "last_weight_update.json": "weight_adjustments", 
            "pattern_analysis/discovered_patterns.json": "pattern_discoveries",
            "pattern_analysis/report_20250702_091533.json": "pattern_discoveries",
            "pattern_analysis/report_20250702_092452.json": "pattern_discoveries",
            "feedback_data/user_feedback.json": "feedback_analysis"
        }
        
        for file_path, category in files_to_process.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    timestamp = datetime.now().isoformat()
                    
                    if category == "model_performance":
                        consolidated_data["model_performance"][timestamp] = data
                    elif category == "weight_adjustments":
                        consolidated_data["weight_adjustments"].append({
                            "timestamp": timestamp,
                            "source_file": file_path,
                            "data": data
                        })
                    elif category == "pattern_discoveries":
                        consolidated_data["pattern_discoveries"][file_path] = {
                            "timestamp": timestamp,
                            "data": data
                        }
                    elif category == "feedback_analysis":
                        consolidated_data["feedback_analysis"]["raw_feedback"] = data
                        consolidated_data["feedback_analysis"]["last_updated"] = timestamp
                    
                    print(f"✅ Consolidated: {file_path}")
                    
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
        
        # Update metadata
        consolidated_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save consolidated data
        with open(self.consolidated_file, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        
        # Generate summary
        self._generate_performance_summary(consolidated_data)
        
        print(f"✅ Consolidation complete! Results saved to: {self.results_dir}/")
    
    def _load_consolidated_data(self):
        """Load existing consolidated data"""
        with open(self.consolidated_file, 'r') as f:
            return json.load(f)
    
    def _generate_performance_summary(self, data):
        """Generate a readable performance summary"""
        summary = {
            "generated_at": datetime.now().isoformat(),
            "overview": {
                "total_feedback_samples": 0,
                "total_pattern_files": len(data.get("pattern_discoveries", {})),
                "total_weight_updates": len(data.get("weight_adjustments", [])),
                "latest_accuracy": 0.0
            },
            "model_stats": {},
            "recent_patterns": [],
            "performance_trends": []
        }
        
        # Analyze feedback data
        feedback_data = data.get("feedback_analysis", {}).get("raw_feedback", [])
        if feedback_data:
            summary["overview"]["total_feedback_samples"] = len(feedback_data)
            
            # Calculate recent accuracy
            recent_samples = feedback_data[-10:] if len(feedback_data) >= 10 else feedback_data
            correct_predictions = 0
            
            for sample in recent_samples:
                true_label = sample.get("true_label", "")
                model_pred = sample.get("model_prediction", "").lower()
                
                if true_label == "ai_generated" and any(word in model_pred for word in ["ai", "likely_ai", "possibly_ai"]):
                    correct_predictions += 1
                elif true_label == "human_created" and any(word in model_pred for word in ["human", "likely_human"]):
                    correct_predictions += 1
            
            if recent_samples:
                summary["overview"]["latest_accuracy"] = correct_predictions / len(recent_samples)
        
        # Extract model performance
        for timestamp, perf_data in data.get("model_performance", {}).items():
            if "model_performance" in perf_data:
                summary["model_stats"][timestamp] = perf_data["model_performance"]
        
        # Extract recent patterns
        for file_path, pattern_data in data.get("pattern_discoveries", {}).items():
            if "insights" in pattern_data.get("data", {}):
                summary["recent_patterns"].extend(pattern_data["data"]["insights"])
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def add_new_result(self, result_type: str, data: Dict[str, Any], source: str = "unknown"):
        """Add new result to consolidated file"""
        consolidated_data = self._load_consolidated_data()
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "source": source,
            "data": data
        }
        
        if result_type == "tuning":
            consolidated_data["tuning_history"].append(entry)
        elif result_type == "weight_update":
            consolidated_data["weight_adjustments"].append(entry)
        elif result_type == "pattern_analysis":
            consolidated_data["pattern_discoveries"][f"analysis_{timestamp}"] = entry
        elif result_type == "model_performance":
            consolidated_data["model_performance"][timestamp] = data
        
        # Update metadata
        consolidated_data["metadata"]["last_updated"] = timestamp
        
        # Save
        with open(self.consolidated_file, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        
        # Regenerate summary
        self._generate_performance_summary(consolidated_data)
    
    def get_latest_summary(self):
        """Get latest performance summary"""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_performance_timeline(self, limit: int = 10):
        """Get timeline of recent performance data"""
        consolidated_data = self._load_consolidated_data()
        
        timeline = []
        
        # Add weight adjustments
        for adjustment in consolidated_data.get("weight_adjustments", [])[-limit:]:
            timeline.append({
                "timestamp": adjustment["timestamp"],
                "type": "weight_adjustment",
                "data": adjustment
            })
        
        # Add tuning history
        for tuning in consolidated_data.get("tuning_history", [])[-limit:]:
            timeline.append({
                "timestamp": tuning["timestamp"], 
                "type": "tuning",
                "data": tuning
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return timeline[:limit]
    
    def cleanup_old_files(self, backup: bool = True):
        """Clean up old scattered result files (with backup option)"""
        files_to_cleanup = [
            "fine_tuning_results.json",
            "last_weight_update.json", 
            "pattern_analysis/discovered_patterns.json",
            "pattern_analysis/report_20250702_091533.json",
            "pattern_analysis/report_20250702_092452.json"
        ]
        
        if backup:
            backup_dir = os.path.join(self.results_dir, "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_path in files_to_cleanup:
                if os.path.exists(file_path):
                    import shutil
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, backup_path)
                    print(f"📦 Backed up: {file_path} -> {backup_path}")
        
        for file_path in files_to_cleanup:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Removed: {file_path}")
    
    def generate_readable_report(self):
        """Generate a human-readable report"""
        summary = self.get_latest_summary()
        timeline = self.get_performance_timeline()
        
        report_path = os.path.join(self.results_dir, "readable_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# AI Detector Performance Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            overview = summary.get("overview", {})
            f.write(f"- **Total Feedback Samples**: {overview.get('total_feedback_samples', 0)}\n")
            f.write(f"- **Latest Accuracy**: {overview.get('latest_accuracy', 0):.1%}\n")
            f.write(f"- **Pattern Files**: {overview.get('total_pattern_files', 0)}\n")
            f.write(f"- **Weight Updates**: {overview.get('total_weight_updates', 0)}\n\n")
            
            # Model Performance
            f.write("## Model Performance\n\n")
            model_stats = summary.get("model_stats", {})
            if model_stats:
                latest_stats = list(model_stats.values())[-1] if model_stats else {}
                for model_name, stats in latest_stats.items():
                    accuracy = stats.get("accuracy", 0)
                    f.write(f"- **{model_name}**: {accuracy:.1%} accuracy\n")
            else:
                f.write("No model performance data available.\n")
            f.write("\n")
            
            # Recent Activity
            f.write("## Recent Activity\n\n")
            for item in timeline[:5]:
                timestamp = item["timestamp"][:19].replace("T", " ")
                activity_type = item["type"].replace("_", " ").title()
                f.write(f"- **{timestamp}**: {activity_type}\n")
            
            f.write(f"\n---\n\n*Report saved to: {report_path}*\n")
        
        return report_path

# Global instance
results_manager = ResultsManager()

if __name__ == "__main__":
    print("🔧 Results Manager - Consolidating Files")
    print("=" * 50)
    
    # Consolidate existing files
    results_manager.consolidate_existing_files()
    
    # Generate readable report
    report_path = results_manager.generate_readable_report()
    print(f"\n📊 Readable report generated: {report_path}")
    
    # Ask about cleanup
    response = input("\n🗑️  Remove old scattered files? (y/n): ").lower().strip()
    if response == 'y':
        results_manager.cleanup_old_files(backup=True)
        print("✅ Cleanup complete with backup!")
    
    print(f"\n✅ All results now consolidated in: model_results/")
    print("   - consolidated_results.json (complete data)")
    print("   - performance_summary.json (quick overview)")
    print("   - readable_report.md (human-readable)")
