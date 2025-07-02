from flask import Flask, render_template, request
from PIL import Image
import random

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image = Image.open(file)
            # TEMP: fake AI detection (random)
            result = "AI-generated" if random.random() > 0.5 else "Human-made"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
