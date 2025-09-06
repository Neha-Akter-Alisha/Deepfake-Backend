from flask import Flask, request, jsonify, render_template, url_for
import os
from werkzeug.utils import secure_filename
from model import HybridDeepFakeDetector

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- Model Initialization ---
# Load the detector model once when the application starts
detector = HybridDeepFakeDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the file upload and analysis, returning JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    analysis_type = request.form.get('analysis_type')

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not analysis_type:
        return jsonify({'error': 'No analysis type selected.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        file_extension = filename.rsplit('.', 1)[1].lower()

        try:
            # --- PDF Analysis Logic ---
            if file_extension == 'pdf':
                if analysis_type == 'image':
                    return jsonify({'error': 'DeepFake analysis is for images (JPG, PNG), not PDFs.'}), 400
                
                pdf_results = detector.analyze_pdf_forgery(upload_path, app.config['PROCESSED_FOLDER'])
                
                # Convert local paths to URLs for the frontend
                for result in pdf_results:
                    if result.get('original_page_image_path'):
                        result['original_image_url'] = url_for('static', filename=f"processed/{os.path.basename(result['original_page_image_path'])}")
                    if result.get('analyzed_image_path'):
                        result['analyzed_image_url'] = url_for('static', filename=f"processed/{os.path.basename(result['analyzed_image_path'])}")

                return jsonify({
                    'analysis_type': 'pdf',
                    'results': pdf_results,
                    'original_filename': filename
                })

            # --- Image Analysis Logic ---
            elif analysis_type == 'image':
                probability, highlighted_path = detector.predict_image_deepfake(upload_path, app.config['PROCESSED_FOLDER'])
                
                real_score = (1 - probability) * 100
                fake_score = probability * 100
                verdict = "DeepFake Detected" if probability > 0.5 else "Likely Authentic"
                explanation = "The model detected subtle artifacts consistent with AI-generated images." if probability > 0.5 else "The model did not find significant evidence of AI manipulation."
                
                return jsonify({
                    'verdict': verdict,
                    'real_score': f"{real_score:.2f}",
                    'fake_score': f"{fake_score:.2f}",
                    'explanation': explanation,
                    'analysis_type': 'image',
                    'original_image_url': url_for('static', filename=f'uploads/{filename}'),
                    'analyzed_image_url': url_for('static', filename=f'processed/{os.path.basename(highlighted_path)}') if highlighted_path else None
                })

            elif analysis_type == 'document':
                forgery_score, verdict, analyzed_path = detector.analyze_document_forgery(upload_path, app.config['PROCESSED_FOLDER'])
                
                explanation = "ELA detected inconsistencies in JPEG compression levels, suggesting a potential digital modification." if verdict == "Suspicious Forgery" else "The document's compression levels appear consistent, indicating it is likely unmodified."

                return jsonify({
                    'verdict': verdict,
                    'forgery_score': f"{forgery_score:.2f}",
                    'explanation': explanation,
                    'analysis_type': 'document',
                    'original_image_url': url_for('static', filename=f'uploads/{filename}'),
                    'analyzed_image_url': url_for('static', filename=f'processed/{os.path.basename(analyzed_path)}') if analyzed_path else None
                })
            
            else:
                return jsonify({'error': 'Invalid analysis type specified.'}), 400

        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return jsonify({'error': 'An internal error occurred during processing. Please try again.'}), 500
    
    return jsonify({'error': 'Invalid file type.'}), 400


if __name__ == '__main__':
    app.run(debug=True)

