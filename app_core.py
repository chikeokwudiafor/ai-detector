
from flask import Flask, render_template, request, flash, session, jsonify
import os
import uuid
import json
from datetime import datetime
from config import *

def create_app(test_mode=False):
    """Factory function to create Flask app"""
    app = Flask(__name__)
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'aithentic-detector-2025-secure-key')
    
    # Import the appropriate detection module
    if test_mode:
        from detection_test import AIDetector, get_result_classification
        analytics_file = 'analytics/user_activity_test.json'
    else:
        from detection import AIDetector, get_result_classification
        analytics_file = 'analytics/user_activity.json'
    
    def track_user_activity(event_type, data=None):
        """Track user activities for analytics"""
        try:
            if request.remote_addr == "172.31.128.93":
                return

            analytics_data = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'ip_address': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', ''),
                'referrer': request.headers.get('Referer', ''),
                'data': data or {}
            }

            os.makedirs('analytics', exist_ok=True)
            with open(analytics_file, 'a') as f:
                f.write(json.dumps(analytics_data) + '\n')

        except Exception as e:
            app.logger.error(f"Analytics tracking error: {str(e)}")

    def validate_file(file):
        """Validate uploaded file"""
        if not file or file.filename == '':
            return False, None, ERROR_MESSAGES["no_file"]

        filename = file.filename.lower()

        if filename.endswith(SUPPORTED_IMAGE_FORMATS):
            return True, "image", None
        elif filename.endswith(SUPPORTED_TEXT_FORMATS):
            return True, "text", None
        elif filename.endswith(SUPPORTED_VIDEO_FORMATS):
            return False, None, "🎬 Video Detection Coming Soon\n\nVideo analysis is in development. We're working on this feature!\n\nTry images or text for now."
        else:
            return False, None, ERROR_MESSAGES["unsupported_format"]

    def process_text_content(text_content, filename="direct_input.txt"):
        """Process text content directly"""
        try:
            if len(text_content.strip()) == 0:
                return None, 0.0, "Text content is empty."

            result_type, confidence, raw_scores = AIDetector.detect_text(text_content, filename)

            if result_type in ["model_unavailable", "processing_error"]:
                error_msg = ERROR_MESSAGES.get(result_type, ERROR_MESSAGES["processing_error"])
                return None, 0.0, error_msg

            return result_type, confidence, None

        except Exception as e:
            app.logger.error(f"Text processing error: {str(e)}")
            return None, 0.0, ERROR_MESSAGES["processing_error"]

    def process_file(file, file_type, filename="unknown"):
        """Process file based on its type"""
        try:
            if file_type == "image":
                result_type, confidence, raw_scores = AIDetector.detect_image(file, filename)
            elif file_type == "text":
                text_content = file.read().decode('utf-8')
                if len(text_content.strip()) == 0:
                    return None, 0.0, "Text file is empty."
                result_type, confidence, raw_scores = AIDetector.detect_text(text_content, filename)
            else:
                return None, 0.0, ERROR_MESSAGES["unsupported_format"]

            if result_type in ["model_unavailable", "processing_error"]:
                error_msg = ERROR_MESSAGES.get(result_type, ERROR_MESSAGES["processing_error"])
                return None, 0.0, error_msg

            return result_type, confidence, None

        except UnicodeDecodeError:
            return None, 0.0, "Unable to read text file. Please ensure it's in UTF-8 format."
        except Exception as e:
            app.logger.error(f"File processing error: {str(e)}")
            return None, 0.0, ERROR_MESSAGES["processing_error"]

    # Store functions in app context
    app.track_user_activity = track_user_activity
    app.validate_file = validate_file
    app.process_text_content = process_text_content
    app.process_file = process_file
    app.AIDetector = AIDetector
    app.get_result_classification = get_result_classification
    app.test_mode = test_mode
    
    return app
