from flask import Flask, request, render_template, jsonify
from flask_jsglue import JSGlue
from werkzeug.utils import secure_filename
import os
import util

app = Flask(__name__)
jsglue = JSGlue()
jsglue.init_app(app)

# Load the EfficientNetV2B2 model
util.load_artifacts()

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Classify Waste Image
@app.route("/classifywaste", methods=["POST"])
def classify_waste():
    if "file" not in request.files:
        return jsonify(error="No file part in request"), 400

    image_data = request.files["file"]
    if image_data.filename == "":
        return jsonify(error="No selected file"), 400

    # Ensure uploads directory exists inside static/
    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    # Save uploaded image securely
    image_path = os.path.join(upload_folder, secure_filename(image_data.filename))
    image_data.save(image_path)

    # Predict
    try:
        predicted_value, details, video1, video2 = util.classify_waste(image_path)
    except Exception as e:
        os.remove(image_path)
        return jsonify(error=f"Error during classification: {str(e)}"), 500

    os.remove(image_path)  # Clean up after classification

    return jsonify(
        predicted_value=predicted_value,
        details=details,
        video1=video1,
        video2=video2
    )

# 404 Page Not Found
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
