from flask import Flask, request, jsonify, render_template
from model import model

# ------- APP INIT ------- #
app = Flask(__name__)

# Initiate model object #
classifier = model.HandwrittenDigitsClassifier()

# Load Model #
classifier.load_model(path='model/digits_classifier.h5')

# Check Model #
# classifier.load_data()
# classifier.evaluate_model()
# print(classifier.model.summary())

# ------- ROUTES ------- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    predictions = classifier.predict(image_data=data['image'])
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(port=4000)
