
from flask import Flask, render_template, request, flash
from PIL import Image
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import io
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize AI detection models
print("Loading AI detection models...")

# Text detection model (RoBERTa-based detector)
try:
    text_classifier = pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        tokenizer="roberta-base-openai-detector"
    )
    print("✓ Text detection model loaded")
except:
    # Fallback to a different model
    try:
        text_classifier = pipeline(
            "text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta"
        )
        print("✓ Text detection model loaded (fallback)")
    except:
        text_classifier = None
        print("✗ Text detection model failed to load")

# Image detection using CLIP-based approach
try:
    image_classifier = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector"
    )
    print("✓ Image detection model loaded")
except:
    image_classifier = None
    print("✗ Image detection model failed to load")

def detect_ai_image(image_file):
    """
    Detect AI-generated images using Hugging Face models
    """
    if not image_classifier:
        return "Model not available", 0.0
    
    try:
        image = Image.open(image_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get prediction
        results = image_classifier(image)
        
        # Parse results - look for AI-related labels
        ai_confidence = 0.0
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            # Check if label indicates AI generation
            if any(keyword in label for keyword in ['ai', 'artificial', 'generated', 'fake', 'synthetic']):
                ai_confidence = max(ai_confidence, score)
            elif any(keyword in label for keyword in ['real', 'human', 'natural', 'authentic']):
                ai_confidence = max(ai_confidence, 1.0 - score)
        
        # If no specific AI labels, use the highest confidence result
        if ai_confidence == 0.0 and results:
            # Assume first result is the primary classification
            ai_confidence = results[0]['score'] if 'ai' in results[0]['label'].lower() else 1.0 - results[0]['score']
        
        is_ai = ai_confidence > 0.5
        return "AI-generated" if is_ai else "Human-made", ai_confidence
        
    except Exception as e:
        print(f"Image detection error: {e}")
        return "Error processing image", 0.0

def detect_ai_text(text_content):
    """
    Detect AI-generated text using RoBERTa-based models
    """
    if not text_classifier:
        return "Model not available", 0.0
    
    try:
        # Truncate text to manageable size (RoBERTa models typically handle ~512 tokens)
        # This prevents memory issues and speeds up processing
        if len(text_content) > 1000:  # Rough character limit
            text_content = text_content[:1000]
        
        # Get prediction - no extra parameters needed
        results = text_classifier(text_content)
        
        # Parse results
        ai_confidence = 0.0
        
        if isinstance(results, list) and len(results) > 0:
            result = results[0]  # Take the first result
            label = result['label'].lower()
            score = result['score']
            
            # Different models use different labels
            if any(keyword in label for keyword in ['fake', 'ai', 'generated', 'machine', 'label_1', '1']):
                ai_confidence = score
            elif any(keyword in label for keyword in ['real', 'human', 'authentic', 'label_0', '0']):
                ai_confidence = 1.0 - score
            else:
                # Default: assume higher score means AI-generated
                ai_confidence = score
        else:
            # Single result format
            label = results['label'].lower()
            score = results['score']
            if any(keyword in label for keyword in ['fake', 'ai', 'generated', 'machine', 'label_1', '1']):
                ai_confidence = score
            elif any(keyword in label for keyword in ['real', 'human', 'authentic', 'label_0', '0']):
                ai_confidence = 1.0 - score
            else:
                ai_confidence = score
        
        is_ai = ai_confidence > 0.5
        return "AI-generated" if is_ai else "Human-made", ai_confidence
        
    except Exception as e:
        print(f"Text detection error: {e}")
        return "Error processing text", 0.0

def detect_ai_video(video_file):
    """
    Placeholder for AI video detection.
    Video detection requires more complex frame-by-frame analysis.
    """
    # For now, return a placeholder message
    return "Video detection coming soon", 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    
    if request.method == "POST":
        file = request.files.get("file")
        
        if not file or file.filename == '':
            flash("Please select a file to analyze.", "error")
            return render_template("index.html")
        
        filename = file.filename.lower()
        
        try:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                # Image detection
                result, confidence = detect_ai_image(file)
                
            elif filename.endswith('.txt'):
                # Text detection
                text_content = file.read().decode('utf-8')
                result, confidence = detect_ai_text(text_content)
                
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                # Video detection (placeholder)
                result, confidence = detect_ai_video(file)
                
            else:
                flash("Unsupported file type. Please upload an image, text file, or video.", "error")
                return render_template("index.html")
                
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
            return render_template("index.html")
    
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
