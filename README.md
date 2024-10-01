# Handwritten Digits Classifier
This project is a Convolutional Neural Network (CNN) model that classifies handwritten digits, which training was based on the MNIST dataset. 
The purpose of this project is to demonstrate the capabilities of deep learning techniques for image classification tasks, 
specifically focusing on recognizing handwritten numbers.

## Table of Contents
1. [Overview](#overview)
2. [Demo](#demo)
3. [Installation](#installation)
4. [Model Architecture](#model-architecture)
5. [Inference](#inference)
6. [Training](#training)
7. [Libraries](#libraries)
8. [Features](#features)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9). This project employs a stacked CNN to accurately classify these images. 
The motivation behind this project is to showcase how deep learning can be effectively utilized for image recognition tasks, 
enhancing understanding of neural networks and their applications.

## Demo
You can checkout the model demo [here](https://handwritten-digits-classifier.streamlit.app/).

## Model Architecture

| Layer Type     | Description                                                                              |
|----------------|------------------------------------------------------------------------------------------|
| Input Layer    | Input Shape: (28, 28, 1)                                                                 |
| Convolution 1  | 32 filters, 3x3 kernel, ReLU activation                                                   |
| MaxPooling 1   | 2x2 pool size, Strides: 2                                                                |
| Convolution 2  | 32 filters, 3x3 kernel, ReLU activation                                                   |
| MaxPooling 2   | 2x2 pool size, Strides: 2                                                                |
| Flattening     | Flatten the 2D matrix into a vector                                                       |
| Fully Connected| Dense Layer with 128 units, ReLU activation                                               |
| Output Layer   | Dense Layer with 10 units (for 10 classes), Softmax activation                            |

### Model Compilation
The model is compiled using:
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy

## Installation

Instructions on how to install the project locally.

```bash
# Clone this repository
$ git clone https://github.com/yourusername/handwritten-digits-classifier.git

# Go into the repository
$ cd handwritten-digits-classifier

# Install dependencies
$ pip install -r requirements.txt
```

## Inference

To run the model on inference, you can easily use either Streamlit or Flask interfaces to draw and predict from the model. You can also load directly a sample image using the predict method.
```bash
# To run using Streamlit frontend:
$ streamlit run main.py

# To run using Flask:
$ python flask/app.py
```

## Training
You can run your own model training by the following script:

```bash
# Init model
digits_classifier = HandwrittenDigitsClassifier()

# Load MNIST Dataset, Model Outline and Run Training
digits_classifier.load_data()
digits_classifier.model_outlining()
digits_classifier.train_model()

# Save to a file
digits_classifier.save_model(<path to file>)

# Load trained model
classifier.load_model(path=<path to file>)
classifier.load_data()

# Evaluate
classifier.evaluate_model()
```
The model can be updated directly by it class or taping in some of the available parameters.


## Libraries
- TensorFlow
- TensorFlow Datasets (MNIST)
- Keras
- NumPy
- Pillow (Image processing)
- Flask (Backend/Templating)
- Streamlit (Deploy/Frontend)

## Features
- **Image Preprocessing/Augmentation**: Generalize and prepare images for better model performance.
- **Model Outline**: Outline the model architecture, with number of layers and nodes, defining activation functions and so.
- **Training and Evaluation**: Trains the CNN model and evaluates its accuracy on the test dataset.
- **Frontend for Real-Time Prediction**: Allows users to draw on a canvas and receive predictions.

## Contributing
Contributions are welcome! Please read the contributing guidelines first.

1. Fork the repo
2. Create a new branch (git checkout -b feature/feature-name)
3. Commit your changes (git commit -m 'Add some feature')
4. Push to the branch (git push origin feature/feature-name)
5. Open a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
