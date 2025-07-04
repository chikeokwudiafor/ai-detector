import json
import os
from datetime import datetime
from contextlib import contextmanager

# Simple file-based storage paths
ANALYTICS_FILE = 'analytics/simple_analytics.json'
FEEDBACK_FILE = 'feedback_data/user_feedback.json'
CACHE_FILE = 'cache/analysis_cache.json'

def ensure_directories():
    """Ensure required directories exist"""
    for directory in ['analytics', 'feedback_data', 'cache']:
        os.makedirs(directory, exist_ok=True)

def get_db_connection():
    """Simple function for SQLite compatibility"""
    return None

def init_database():
    """Initialize file-based storage"""
    ensure_directories()

    # Initialize analytics file
    if not os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump([], f)

    # Initialize cache file
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'w') as f:
            json.dump({}, f)

def log_analytics_db(event_type, ip_address, user_agent, referrer, data):
    """Log analytics to file (simplified)"""
    try:
        # Skip detailed analytics for performance
        if event_type == 'page_visit':
            return  # Skip page visits entirely

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
            'total_page_visits': 0,
            'total_analyses': total_analyses,
            'recent_activity': []
        }
    except Exception:
        return {'total_page_visits': 0, 'total_analyses': 0, 'recent_activity': []}

def cache_analysis_result(file_hash, file_type, result_type, confidence):
    """Cache analysis result"""
    try:
        cache_data = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)

        cache_data[file_hash] = {
            'file_type': file_type,
            'result_type': result_type,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }

        # Keep only last 50 entries for performance
        if len(cache_data) > 50:
            sorted_items = sorted(cache_data.items(), key=lambda x: x[1]['timestamp'])
            cache_data = dict(sorted_items[-50:])

        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception:
        pass

def get_cached_result(file_hash):
    """Get cached analysis result"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)

            if file_hash in cache_data:
                cached = cache_data[file_hash]
                # Check if cache is still fresh (1 hour)
                cached_time = datetime.fromisoformat(cached['timestamp'])
                if (datetime.now() - cached_time).total_seconds() < 3600:
                    return cached['result_type'], cached['confidence']
    except Exception:
        pass

    return None, None

# Initialize on import
ensure_directories()