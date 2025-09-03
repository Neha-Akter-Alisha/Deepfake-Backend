from flask import Flask, render_template, request, url_for
import os
from model import HybridDeepFakeDetector
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load our new, smarter detector model
detector = HybridDeepFakeDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected for uploading!")

    analysis_type = request.form.get('analysis_type')
    if not analysis_type:
        return render_template('index.html', error="Please select an analysis type!")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        probability = 0.0
        result_text = "Error"
        processed_image_filename = ""

        # --- This is the new, smarter logic ---
        if analysis_type == 'image':
            # Use the AI model for images/faces
            probability = detector.predict_image_deepfake(file_path)
            processed_image_filename = detector.simulate_heatmap_for_image(file_path)
            result_text = "Fake" if probability > 0.5 else "Real"
            # We present probability as a percentage
            probability *= 100

        elif analysis_type == 'document':
            # Use only ELA for documents
            # The returned score is not a 0-1 probability, it's a forgery score.
            # We set a threshold to decide if it's suspicious.
            forgery_score, processed_image_filename = detector.analyze_document_forgery(file_path)
            
            # Let's set a threshold. A score > 15 is suspicious. Adjust as needed.
            FORGERY_THRESHOLD = 15.0
            result_text = "Forgery Detected (Suspicious)" if forgery_score > FORGERY_THRESHOLD else "Likely Authentic"
            probability = forgery_score # Pass the raw score to the template

        return render_template('result.html',
                               original_image=filename,
                               processed_image=processed_image_filename,
                               probability=probability,
                               result=result_text,
                               analysis_type=analysis_type)

    return render_template('index.html', error="File could not be saved.")

if __name__ == '__main__':
    app.run(debug=True)

