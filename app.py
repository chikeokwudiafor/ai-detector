import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, flash, session, jsonify, make_response
from config import *
from database import init_database, log_analytics_db, save_feedback_db, get_analytics_summary

# Global variables for lazy loading
_models_loaded = False
AIDetector = None
get_result_classification = None

# Removed response cache to avoid threading issues in Replit

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'aithentic-detector-2025-secure-key')

def ensure_models_loaded():
    """Load AI detection models on first request"""
    global _models_loaded, AIDetector, get_result_classification

    if not _models_loaded:
        print("🚀 Loading AI detection modules...")
        try:
            # Import detection modules (models load lazily)
            from detection import AIDetector as _AIDetector, get_result_classification as _get_result_classification
            AIDetector = _AIDetector
            get_result_classification = _get_result_classification

            print("✅ Detection modules loaded")
            _models_loaded = True
        except Exception as e:
            print(f"❌ Failed to load modules: {e}")
            raise e

def track_user_activity(event_type, data=None):
    """Track user activities for analytics (simplified)"""
    try:
        # Only track important events
        if event_type == 'analysis_completed':
            log_analytics_db(
                event_type=event_type,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', ''),
                referrer=request.headers.get('Referer', ''),
                data=data
            )
    except Exception:
        pass  # Ignore analytics errors

def validate_file(file):
    """
    Validate uploaded file
    Returns: (is_valid, file_type, error_message)
    """
    if not file or file.filename == '':
        return False, None, ERROR_MESSAGES["no_file"]

    filename = file.filename.lower()

    # Check file type
    if filename.endswith(SUPPORTED_IMAGE_FORMATS):
        return True, "image", None
    elif filename.endswith(SUPPORTED_TEXT_FORMATS):
        return True, "text", None
    elif filename.endswith(SUPPORTED_VIDEO_FORMATS):
        # Block video files with specific message
        return False, None, "🎬 Video Detection Coming Soon\n\nVideo analysis is in development. We're working on this feature!\n\nTry images or text for now."
    else:
        return False, None, ERROR_MESSAGES["unsupported_format"]

def process_text_content(text_content, filename="direct_input.txt"):
    """
    Process text content directly
    Returns: (result_type, confidence, error_message)
    """
    try:
        if len(text_content.strip()) == 0:
            return None, 0.0, "Text content is empty."

        result_type, confidence, raw_scores = AIDetector.detect_text(text_content, filename)

        # Handle model errors
        if result_type in ["model_unavailable", "processing_error"]:
            error_msg = ERROR_MESSAGES.get(result_type, ERROR_MESSAGES["processing_error"])
            return None, 0.0, error_msg

        return result_type, confidence, None

    except Exception as e:
        app.logger.error(f"Text processing error: {str(e)}")
        return None, 0.0, ERROR_MESSAGES["processing_error"]

def process_file(file, file_type, filename="unknown"):
    """
    Process file based on its type
    Returns: (result_type, confidence, error_message)
    """
    try:
        if file_type == "image":
            result_type, confidence, raw_scores = AIDetector.detect_image(file, filename)
        elif file_type == "text":
            text_content = file.read().decode('utf-8')
            if len(text_content.strip()) == 0:
                return None, 0.0, "Text file is empty."
            result_type, confidence, raw_scores = AIDetector.detect_text(text_content, filename)
        elif file_type == "video":
            # Even if you’re not doing video detection yet, log it:
            app.logger.warning("Video detection not implemented yet. Coming soon.")
            result_type = "video_not_implemented"
            confidence = 0.0
            # result_type, confidence, raw_scores = AIDetector.detect_video(file, filename) #Commented out until video is implemented
        else:
            return None, 0.0, ERROR_MESSAGES["unsupported_format"]

        # Handle model errors and special cases
        if result_type in ["model_unavailable", "processing_error"]:
            error_msg = ERROR_MESSAGES.get(result_type, ERROR_MESSAGES["processing_error"])
            return None, 0.0, error_msg
        elif result_type == "video_not_implemented":
            # For video not implemented, show as a result rather than an error
            return result_type, 0.0, None

        return result_type, confidence, None

    except UnicodeDecodeError:
        return None, 0.0, "Unable to read text file. Please ensure it's in UTF-8 format."
    except Exception as e:
        app.logger.error(f"File processing error: {str(e)}")
        return None, 0.0, ERROR_MESSAGES["processing_error"]

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route handler"""
    result = None
    confidence = None
    result_class = None
    result_icon = None
    result_description = None
    session_id = None

    # Skip page visit tracking for performance

    if request.method == "POST":
        # Ensure models are loaded before processing
        ensure_models_loaded()

        file = request.files.get("file")
        text_content = request.form.get("text_content")

        # Handle direct text input
        if text_content and text_content.strip():
            if len(text_content.strip()) < 10:
                flash("Please enter at least 10 characters for better analysis.", "error")
                return render_template("index.html")

            # Process text directly
            result_type, confidence, error_msg = process_text_content(text_content.strip())
            if error_msg:
                flash(error_msg, "error")
                return render_template("index.html")
        else:
            # Handle file upload
            # Validate file
            is_valid, file_type, error_msg = validate_file(file)
            if not is_valid:
                flash(error_msg, "error")
                return render_template("index.html")

            # Process file
            result_type, confidence, error_msg = process_file(file, file_type, file.filename)
            if error_msg:
                flash(error_msg, "error")
                return render_template("index.html")

        # Get result classification
        if result_type and confidence is not None:
            result, result_class, result_icon, result_description, result_footer = get_result_classification(result_type)

            # Generate session ID for feedback
            session_id = str(uuid.uuid4())

            # Store analysis data in session for feedback (except for video not implemented)
            filename = file.filename if file else "direct_text_input"
            file_type_name = file_type if 'file_type' in locals() else "text"

            # Don't store session data for video not implemented (no actual analysis occurred)
            if result_type != "video_not_implemented":
                session['last_analysis'] = {
                    'session_id': session_id,
                    'filename': filename,
                    'file_type': file_type_name,
                    'result': result,
                    'result_type': result_type,
                    'confidence': confidence
                }

            # Log successful analysis
            log_filename = file.filename if file else "direct_text_input"
            app.logger.info(f"Analysis complete: {log_filename} -> {result_type} ({confidence:.3f})")

            # Track analysis event
            track_user_activity('analysis_completed', {
                'filename': log_filename,
                'file_type': file_type_name,
                'result_type': result_type,
                'confidence': confidence
            })

    return render_template("index.html", 
                         result=result, 
                         confidence=confidence, 
                         result_class=result_class,
                         result_icon=result_icon,
                         result_description=result_description,
                         result_footer=result_footer if 'result_footer' in locals() else None,
                         session_id=session_id)

@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")

@app.route("/roadmap")
def roadmap():
    """Roadmap page"""
    return render_template("roadmap.html")

@app.route("/license")
def license_page():
    """License page"""
    return render_template("license.html")

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Handle user feedback on model accuracy"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        true_label = data.get('feedback')  # 'ai_generated' or 'human_created'

        if not session_id or not true_label:
            return jsonify({"error": "Missing required data"}), 400

        # Validate true_label values
        if true_label not in ['ai_generated', 'human_created']:
            return jsonify({"error": "Invalid feedback value"}), 400

        # Get session data
        session_data = session.get('last_analysis')
        if not session_data:
            return jsonify({"error": "No recent analysis found"}), 400

        # Save feedback to file
        try:
            feedback_data = {
                'session_id': session_id,
                'file_type': session_data['file_type'],
                'filename': session_data['filename'],
                'model_result': session_data['result'],
                'true_label': true_label,
                'timestamp': datetime.now().isoformat()
            }

            os.makedirs('feedback_data', exist_ok=True)
            feedback_file = 'feedback_data/user_feedback.json'

            # Load existing feedback
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    existing_feedback = json.load(f)
            else:
                existing_feedback = []

            # Add new feedback
            existing_feedback.append(feedback_data)

            # Save back to file
            with open(feedback_file, 'w') as f:
                json.dump(existing_feedback, f, indent=2)

            success = True
            
            # Trigger continuous feedback monitoring check
            try:
                from feedback_analyzer import continuous_monitor
                continuous_monitor.check_for_updates()
            except Exception as monitor_error:
                app.logger.warning(f"Feedback monitoring check failed: {monitor_error}")
                
        except Exception as e:
            app.logger.error(f"Feedback save error: {e}")
            success = False

        if success:
            return jsonify({"message": "Thank you for your feedback!"})
        else:
            return jsonify({"error": "Failed to save feedback"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ping")
def ping():
    """Quick health check endpoint"""
    return "pong", 200


@app.route('/admin/feedback-analysis')
def feedback_analysis():
    """Admin endpoint to view feedback analysis and weight suggestions"""
    try:
        from feedback_analyzer import FeedbackAnalyzer
        analyzer = FeedbackAnalyzer()
        analysis = analyzer.analyze_model_performance()
        suggestions = analyzer.suggest_weight_adjustments()
        
        return jsonify({
            'analysis': analysis,
            'suggestions': suggestions,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/update-weights', methods=['POST'])
def update_weights():
    """Force update adaptive weights from feedback"""
    try:
        from feedback_analyzer import adaptive_weight_manager
        updated_weights = adaptive_weight_manager.update_weights_from_feedback(force_update=True)
        
        if updated_weights:
            return jsonify({
                'message': 'Weights updated successfully',
                'updated_weights': updated_weights,
                'status': 'success'
            })
        else:
            return jsonify({
                'message': 'No weight updates needed',
                'status': 'no_changes'
            })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/accuracy-report')
def get_accuracy_report():
    """Get the latest comprehensive accuracy report"""
    try:
        from feedback_analyzer import FeedbackAnalyzer
        analyzer = FeedbackAnalyzer()
        report = analyzer.generate_comprehensive_report()
        
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/generate-report', methods=['POST'])
def generate_report():
    """Force generate and save a new accuracy report"""
    try:
        from feedback_analyzer import adaptive_weight_manager
        report_file = adaptive_weight_manager.generate_and_save_report()
        
        if report_file:
            return jsonify({
                'message': 'Report generated successfully',
                'report_file': report_file,
                'status': 'success'
            })
        else:
            return jsonify({
                'message': 'Failed to generate report',
                'status': 'error'
            })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/monitoring-status')
def monitoring_status():
    """Get status of continuous feedback monitoring"""
    try:
        from feedback_analyzer import continuous_monitor
        current_count = continuous_monitor.check_for_updates()
        
        return jsonify({
            'current_feedback_count': current_count,
            'last_feedback_count': continuous_monitor.last_feedback_count,
            'last_report_time': continuous_monitor.last_report_time.isoformat() if continuous_monitor.last_report_time else None,
            'status': 'active'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/export-csv')
def export_feedback_csv():
    """Export feedback data to CSV format"""
    try:
        from report_exporter import ReportExporter
        exporter = ReportExporter()
        csv_file = exporter.export_feedback_to_csv()
        
        if csv_file:
            return jsonify({
                'message': 'Feedback data exported to CSV',
                'file_path': csv_file,
                'status': 'success'
            })
        else:
            return jsonify({
                'message': 'No feedback data to export',
                'status': 'no_data'
            })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/export-markdown')
def export_accuracy_markdown():
    """Export accuracy report to Markdown format"""
    try:
        from report_exporter import ReportExporter
        exporter = ReportExporter()
        md_file = exporter.export_accuracy_report_to_markdown()
        
        return jsonify({
            'message': 'Accuracy report exported to Markdown',
            'file_path': md_file,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/export-analytics-csv')
def export_analytics_csv():
    """Export analytics data to CSV format"""
    try:
        from report_exporter import ReportExporter
        exporter = ReportExporter()
        csv_file = exporter.export_analytics_to_csv()
        
        if csv_file:
            return jsonify({
                'message': 'Analytics data exported to CSV',
                'file_path': csv_file,
                'status': 'success'
            })
        else:
            return jsonify({
                'message': 'No analytics data to export',
                'status': 'no_data'
            })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/export-all')
def export_all_reports():
    """Export all reports to CSV and Markdown formats"""
    try:
        from report_exporter import export_all_reports
        results = export_all_reports()
        
        return jsonify({
            'message': 'All reports exported successfully',
            'exported_files': results,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route("/health")
def health_check():
    """Detailed health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": _models_loaded,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/analytics")
def analytics_dashboard():
    """Simple analytics dashboard (basic auth recommended for production)"""
    try:
        summary = get_analytics_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("⚡ Lazy startup mode: skipping model loading at boot")
    print("🔥 Flask is starting on 0.0.0.0:5000")

    # Delay DB and model init until first request
    app.run(host='0.0.0.0', port=5000)