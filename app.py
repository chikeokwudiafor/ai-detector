from flask import Flask, render_template, request, flash, jsonify
import os
from detection import AIDetector, get_result_classification
from feedback import FeedbackCollector
from config import *

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'aithentic-detector-2025-secure-key')

# Initialize feedback collector
feedback_collector = FeedbackCollector()

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
        return True, "video", None
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
            result_type, confidence, raw_scores = AIDetector.detect_video(file, filename)
        else:
            return None, 0.0, ERROR_MESSAGES["unsupported_format"]

        # Handle model errors
        if result_type in ["model_unavailable", "processing_error"]:
            error_msg = ERROR_MESSAGES.get(result_type, ERROR_MESSAGES["processing_error"])
            return None, 0.0, error_msg

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

    if request.method == "POST":
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
            result, result_class, result_icon, result_description = get_result_classification(result_type)

            # Log successful analysis
            app.logger.info(f"Analysis complete: {file.filename} -> {result_type} ({confidence:.3f})")

    return render_template("index.html", 
                         result=result, 
                         confidence=confidence, 
                         result_class=result_class,
                         result_icon=result_icon,
                         result_description=result_description)

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
    """Handle user feedback submission"""
    try:
        data = request.get_json()
        
        # Extract feedback data
        feedback_type = data.get('feedback')  # 'correct' or 'incorrect'
        result_type = data.get('result_type')
        confidence = data.get('confidence')
        content_type = data.get('content_type')
        
        # Convert feedback to user correction
        if feedback_type == 'correct':
            user_correction = result_type
        else:
            # If user says it's incorrect, assume opposite
            if 'ai' in result_type.lower():
                user_correction = 'human'
            else:
                user_correction = 'ai'
        
        # Save feedback (using placeholder content since we don't store actual content)
        feedback_collector.save_feedback(
            content=f"User feedback session {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            content_type=content_type,
            model_prediction=result_type,
            user_correction=user_correction,
            confidence=confidence
        )
        
        # Log feedback
        app.logger.info(f"Feedback received: {feedback_type} for {result_type} ({confidence:.3f})")
        
        return jsonify({"success": True, "message": "Feedback saved successfully"})
        
    except Exception as e:
        app.logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({"success": False, "message": "Error saving feedback"}), 500

@app.route("/feedback/stats")
def feedback_stats():
    """Get feedback statistics"""
    try:
        stats = feedback_collector.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting feedback stats: {str(e)}")
        return jsonify({"error": "Unable to get feedback stats"}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)