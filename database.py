import json
import os
from datetime import datetime

# Simple file-based storage paths
ANALYTICS_FILE = 'analytics/simple_analytics.json'
FEEDBACK_FILE = 'feedback_data/user_feedback.json'

def ensure_directories():
    """Ensure required directories exist"""
    for directory in ['analytics', 'feedback_data']:
        os.makedirs(directory, exist_ok=True)

def init_database():
    """Initialize file-based storage"""
    ensure_directories()

    # Initialize analytics file
    if not os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump([], f)

def log_analytics_db(event_type, ip_address, user_agent, referrer, data):
    """Log analytics to file (simplified)"""
    try:
        # Only log important events
        if event_type == 'page_visit':
            return

        analytics_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data or {}
        }

        # Simple append to file
        with open(ANALYTICS_FILE, 'a') as f:
            f.write(json.dumps(analytics_data) + '\n')
    except Exception:
        pass  # Ignore analytics errors

def save_feedback_db(session_id, file_type, filename, model_result, true_label):
    """Save feedback to file"""
    try:
        feedback_data = {
            'session_id': session_id,
            'file_type': file_type,
            'filename': filename,
            'model_result': model_result,
            'true_label': true_label,
            'timestamp': datetime.now().isoformat()
        }

        # Load existing feedback
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                existing_feedback = json.load(f)
        else:
            existing_feedback = []

        existing_feedback.append(feedback_data)

        # Save back to file
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(existing_feedback, f, indent=2)

        return True
    except Exception:
        return False

def get_analytics_summary():
    """Get simple analytics summary"""
    try:
        total_analyses = 0
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        total_analyses += 1

        return {
            'total_analyses': total_analyses,
        }
    except Exception:
        return {'total_analyses': 0}

# Initialize on import
ensure_directories()