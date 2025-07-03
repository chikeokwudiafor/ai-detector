import sqlite3
import json
from datetime import datetime, timedelta
import os
from contextlib import contextmanager
import threading

DB_PATH = 'aithentic.db'

class ConnectionPool:
    def __init__(self, db_path, max_connections=5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        for _ in range(max_connections):
            self._pool.append(self._create_connection())

    def _create_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self):
        with self._lock:
            while not self._pool:
                self._condition.wait()  # Wait for a connection to become available
            return self._pool.pop()

    def release_connection(self, conn):
        with self._lock:
            self._pool.append(conn)
            self._condition.notify()  # Notify waiting threads that a connection is available

    def close_all_connections(self):
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool = []

# Initialize the connection pool
connection_pool = ConnectionPool(DB_PATH)

@contextmanager
def get_db_connection():
    """Context manager for database connections using connection pool"""
    conn = connection_pool.get_connection()
    try:
        yield conn
    finally:
        connection_pool.release_connection(conn)

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

        # Add index for file_hash on analysis_cache table
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_hash ON analysis_cache (file_hash)
        ''')

        # Optimize database
        conn.execute("VACUUM")

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