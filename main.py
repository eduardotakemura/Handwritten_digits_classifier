import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
import base64
from model import model
from streamlit_drawable_canvas import st_canvas
import os


# ------- APP INIT ------- #
def initialize_classifier():
    """Load the trained classifier model."""
    model_path = os.path.join(os.getcwd(), 'model', 'digits_classifier.h5')
    classifier = model.HandwrittenDigitsClassifier()
    classifier.load_model(path=model_path)
    return classifier


# ------- AUXILIAR FUNCTIONS ------- #
def pil_image_to_base64(img):
    """Convert a PIL Image to a base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def preprocess_drawing(canvas_data):
    """Preprocess the canvas data to prepare it for prediction."""
    drawn_image_np = np.uint8(canvas_data * 255)
    rgb_image = drawn_image_np[:, :, :3]  # Only take RGB, ignore the alpha channel

    # Check if the canvas contains any non-black pixel
    if np.sum(rgb_image) > 0:
        drawn_image = Image.fromarray(drawn_image_np)
        drawn_image = drawn_image.convert('L')  # Convert to grayscale

        # Apply a binary threshold to make the strokes solid white
        threshold = 100
        drawn_image = drawn_image.point(lambda x: 255 if x > threshold else 0, mode='1')

        # Dilate the image to thicken strokes
        drawn_image = drawn_image.filter(ImageFilter.MaxFilter(5))
        return drawn_image
    return None


def make_prediction(image_data, classifier):
    """Predict the digit based on preprocessed image data."""
    predictions = classifier.predict(image_data=image_data, streamlit=True)
    return predictions


def display_predictions(predictions):
    """Display the top two predictions."""
    if predictions is not None:
        top_pred_idx = np.argmax(predictions)
        second_pred_idx = np.argsort(predictions)[-2]

        st.write(
            f"I predict this is a **{top_pred_idx}**, with a **{predictions[top_pred_idx] * 100:.2f}%** certainty.")
        st.write(
            f"My second guess is **{second_pred_idx}**, with a **{predictions[second_pred_idx] * 100:.2f}%** certainty.")


# ------- STREAMLIT INTERFACE ------- #
def setup_page():
    """Set up the Streamlit page with title and description."""
    st.set_page_config(page_title="✍️ Handwritten Digits Classifier", layout="centered")
    st.title("Handwritten Digits Classifier")
    st.write("""
        Hi there! I'm a CNN-based model, and I was trained on the MNIST dataset.
        I can predict handwritten digits with surprising accuracy!"""
             )
    st.write("""
        This is particularly useful for applications like scanning or digitizing handwritten documents.
        While I was trained on digits, this type of model can be extended to recognize any handwritten characters from any language. Pretty cool, right?
    """)
    st.subheader("""Give it a try! Draw something below, and I’ll try to predict what it is:""")
    st.caption(""" **(Try to center the drawing onto the canvas for better results)** """)


def setup_canvas():
    """Set up the canvas for drawing."""
    return st_canvas(
        fill_color="black",  # Background color for the canvas
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )


def manage_session_state():
    """Ensure session state keys for image and predictions exist."""
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None

    if 'last_image' not in st.session_state:
        st.session_state['last_image'] = None


# ------- MAIN FUNCTION ------- #
def main():
    # Initialize the classifier and setup the page
    classifier = initialize_classifier()
    setup_page()

    # Setup two columns: one for the canvas and one for predictions
    col1, col2 = st.columns(2)

    # Setup canvas and session state
    with col1:
        canvas_result = setup_canvas()

    manage_session_state()

    # If the user has drawn something, process and predict
    if canvas_result.image_data is not None:
        drawn_image = preprocess_drawing(canvas_result.image_data)

        if drawn_image is not None:
            # Check if the drawn image is new
            if st.session_state['last_image'] is None or not np.array_equal(np.array(drawn_image),
                                                                            np.array(st.session_state['last_image'])):
                st.session_state['last_image'] = drawn_image

                # Convert the image to base64 and make a prediction
                img_base64 = pil_image_to_base64(drawn_image)
                st.session_state['last_prediction'] = make_prediction(img_base64, classifier)

    # Display predictions in the second column if available
    with col2:
        if st.session_state['last_prediction'] is not None:
            display_predictions(st.session_state['last_prediction'])

if __name__ == '__main__':
    main()
