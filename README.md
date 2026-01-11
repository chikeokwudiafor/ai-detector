
# 🧠 AI Detector

A robust Flask-based web application that detects AI-generated content across multiple modalities (images, text, videos) using ensemble machine learning models.

## 🚀 Features

- **🖼️ Image Detection** - Identify AI-generated photos from models like DALL-E, Midjourney, Stable Diffusion
- **📝 Text Detection** - Flag AI-written content from ChatGPT, Claude, and other language models  
- **🎥 Video Detection** - Analyze video content (coming soon)
- **🎯 Ensemble Models** - Multiple models with weighted voting for higher accuracy
- **📊 Confidence Scoring** - Smart thresholds prevent uncertain results
- **🎨 Modern UI** - Drag & drop interface with real-time feedback
- **📱 Mobile Responsive** - Works seamlessly on all devices

## 🏗️ Architecture

```
ai-detector/
├── app.py              # Flask routes and main application logic
├── detection.py        # AI detection models and ensemble logic  
├── config.py          # Configuration constants and thresholds
├── requirements.txt   # Python dependencies
├── templates/
│   └── index.html     # Main web interface
├── static/
│   └── style.css      # Stylesheets (prepared for future use)
└── README.md         # This file
```

## 🔧 Key Components

### Detection Module (`detection.py`)
- **ModelManager**: Handles loading and caching of AI models with fallbacks
- **AIDetector**: Main detection class with ensemble support
- **Heuristic Adjustments**: Smart confidence adjustments based on content analysis

### Configuration (`config.py`)
- Model configurations with weights for ensemble voting
- Confidence thresholds for reliable results
- Supported file formats and processing limits
- Customizable UI messages and error handling

## 🚀 Getting Started

### Running on Replit
1. Fork this Repl
2. Click the "Run" button
3. The app will start on `https://your-repl-name.your-username.repl.co`

### Local Development
```bash
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000`

## 🎯 How It Works

1. **Upload**: Drag & drop or select files (images, text, videos)
2. **Analysis**: Multiple AI models analyze the content in parallel
3. **Ensemble**: Weighted voting combines model predictions
4. **Confidence**: Smart thresholds ensure reliable results
5. **Results**: Color-coded results with confidence meters

## 🔬 Model Ensemble

The app uses multiple models for each content type:

**Text Detection:**
- RoBERTa-based OpenAI detector (60% weight)
- ChatGPT detector (40% weight)
- Automatic fallbacks if models fail

**Image Detection:**
- AI image detector with heuristic adjustments
- Confidence scaling based on image properties

## 📊 Confidence Thresholds

- **🤖 Very High (85%+)**: Almost certainly AI-generated
- **🤖 High (45-85%)**: Likely AI-generated  
- **🧐 Medium (25-45%)**: Uncertain - manual review needed
- **🧠 Low (15-25%)**: Likely human-created
- **🧠 Very Low (<15%)**: Almost certainly human-created

## 🛡️ Error Handling

- Graceful model loading with fallbacks
- File validation and format checking
- Detailed error messages for users
- Logging for debugging and monitoring

## 🔮 Future Enhancements

- [ ] Video content analysis (frame-by-frame)
- [ ] Audio detection for AI-generated speech
- [ ] Batch processing for multiple files
- [ ] API endpoints for programmatic access
- [ ] Real-time text analysis as you type
- [ ] Export analysis reports

## 🏆 Performance

- **Ensemble Accuracy**: Higher accuracy through model voting
- **Smart Thresholds**: Reduces false positives/negatives
- **Optimized Loading**: Models cached for faster subsequent analyses
- **Responsive Design**: Works on desktop and mobile devices

## 📈 Technical Stack

- **Backend**: Flask, Python 3.8+
- **ML Models**: Transformers, PyTorch, Hugging Face
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Replit (with Autoscale support)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - see LICENSE file for details.
