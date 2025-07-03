
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = 'aithentic.db'

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database tables"""
    with get_db_connection() as conn:
        # Analytics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                referrer TEXT,
                data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feedback table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                model_result TEXT NOT NULL,
                true_label TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analysis results cache
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                result_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

def log_analytics_db(event_type, ip_address, user_agent, referrer, data):
    """Log analytics to database"""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO analytics (timestamp, event_type, ip_address, user_agent, referrer, data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                event_type,
                ip_address,
                user_agent,
                referrer,
                json.dumps(data) if data else None
            ))
            conn.commit()
    except Exception as e:
        print(f"Analytics DB error: {e}")

def save_feedback_db(session_id, file_type, filename, model_result, true_label):
    """Save feedback to database"""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO feedback (session_id, file_type, filename, model_result, true_label)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, file_type, filename, model_result, true_label))
            conn.commit()
            return True
    except Exception as e:
        print(f"Feedback DB error: {e}")
        return False

def get_analytics_summary():
    """Get analytics summary from database"""
    try:
        with get_db_connection() as conn:
            # Total visits
            visits = conn.execute(
                "SELECT COUNT(*) as count FROM analytics WHERE event_type = 'page_visit'"
            ).fetchone()['count']
            
            # Total analyses
            analyses = conn.execute(
                "SELECT COUNT(*) as count FROM analytics WHERE event_type = 'analysis_completed'"
            ).fetchone()['count']
            
            # Recent activity
            recent = conn.execute('''
                SELECT * FROM analytics 
                ORDER BY created_at DESC 
                LIMIT 10
            ''').fetchall()
            
            return {
                'total_page_visits': visits,
                'total_analyses': analyses,
                'recent_activity': [dict(row) for row in recent]
            }
    except Exception as e:
        print(f"Analytics summary error: {e}")
        return {'total_page_visits': 0, 'total_analyses': 0, 'recent_activity': []}

def cache_analysis_result(file_hash, file_type, result_type, confidence):
    """Cache analysis result for performance"""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO analysis_cache (file_hash, file_type, result_type, confidence)
                VALUES (?, ?, ?, ?)
            ''', (file_hash, file_type, result_type, confidence))
            conn.commit()
    except Exception as e:
        print(f"Cache error: {e}")

def get_cached_result(file_hash):
    """Get cached analysis result"""
    try:
        with get_db_connection() as conn:
            result = conn.execute('''
                SELECT result_type, confidence FROM analysis_cache 
                WHERE file_hash = ? AND datetime(created_at) > datetime('now', '-1 hour')
            ''', (file_hash,)).fetchone()
            
            if result:
                return result['result_type'], result['confidence']
    except Exception as e:
        print(f"Cache retrieval error: {e}")
    
    return None, None

# Initialize database on import
if not os.path.exists(DB_PATH):
    init_database()
