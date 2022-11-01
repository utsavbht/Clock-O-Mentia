from flask import Flask, request, jsonify
import gc
from app.torch_utils import get_prediction, transform_image
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello_world():
    return f"<p>Hello, World!</p>"

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'status': 'error',
                            'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'status': 'error',
                            'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction, probabilities = get_prediction(tensor)
            data = {'status': 'success',
                    'prediction': prediction,
                    'probabilities': probabilities}
            gc.collect()
            return jsonify(data)
        except:
            return jsonify({'status': 'error',
                            'error': 'error during prediction'})