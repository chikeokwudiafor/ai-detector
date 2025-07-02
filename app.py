from flask import Flask, render_template, request, flash
import os
from detection import AIDetector, get_result_classification
from config import *

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

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

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)