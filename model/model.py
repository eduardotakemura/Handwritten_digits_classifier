import tensorflow as tf
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
from PIL import Image
import io
import base64

class HandwrittenDigitsClassifier:
    def __init__(self):
        self.max_epochs = 200
        self.training_set = None
        self.valid_set = None
        self.test_set = None
        self.model = None
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=2,
            min_delta=0.0001,
            restore_best_weights=True
        )

    def _load_mnist(self):
        """ Load MNIST Datasets from Tensorflow """
        # Load dataset from TF with infos #
        mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

        # Training and Test splits #
        mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

        # Validation split #
        valid_samples = 0.1 * mnist_info.splits['train'].num_examples
        valid_samples = tf.cast(valid_samples, tf.int64)  # Cast count to a int
        mnist_valid = mnist_train.take(valid_samples)
        mnist_train = mnist_train.skip(valid_samples)

        return mnist_train, mnist_valid, mnist_test

    def _preprocess_mnist(self, mnist_train, mnist_valid, mnist_test):
        """ Take a MNIST dataset from TF, augment and reshape it, then extract training, validation and test sets. """
        train_images, train_labels = self._dataset_to_numpy(mnist_train)
        valid_images, valid_labels = self._dataset_to_numpy(mnist_valid)
        test_images, test_labels = self._dataset_to_numpy(mnist_test)

        # Reshape train_images for ImageDataGenerator
        train_images = train_images.reshape(-1, 28, 28, 1)  # Grayscale images (1 channel)
        valid_images = valid_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        training_set = train_datagen.flow(
            train_images,
            train_labels,
            batch_size=32,
            shuffle=True
        )

        # ## Test and Validation Sets - Only Rescale ##
        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        valid_set = valid_datagen.flow(
            valid_images,
            valid_labels,
            batch_size=32
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_set = test_datagen.flow(
            test_images,
            test_labels,
            batch_size=32
        )
        return training_set, valid_set, test_set

    def _dataset_to_numpy(self, dataset, batch_size=60000):
        """ Auxiliar method to extract images and labels from the TF MNIST set. """
        dataset = dataset.batch(batch_size)
        for images, labels in dataset:
            return images.numpy(), labels.numpy()

    def model_outlining(self):
        """ Outline the Model's Layers and Compiler. """
        ## Initializing the CNN ##
        self.model = tf.keras.models.Sequential()

        ## Step 1 - First Convolution Layer ##
        self.model.add(
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu',
                input_shape=[28, 28, 1]
            ))

        ## Step 2 - First Pooling Layer ##
        self.model.add(
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2
            ))

        ## Step 3 (optional) - Adding a second convolutional layer (convolutional + pooling) ##
        self.model.add(
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu'
            ))
        self.model.add(
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2
            ))

        ## Step 4 - Flattening ##
        self.model.add(tf.keras.layers.Flatten())

        ## Step 5 - Full Connection (ANN) ##
        self.model.add(
            tf.keras.layers.Dense(
                units=128,
                activation='relu'
            ))

        ## Step 6 - Output Layer ##
        self.model.add(
            tf.keras.layers.Dense(
                units=10,
                activation='softmax'
            ))

        ## Step 7 - Set Model Compiler #
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def load_mnist(self):
        """ Pipeline to load MNIST Datasets and Preprocess it. """
        mnist_train, mnist_valid, mnist_test = self._load_mnist()
        self.training_set, self.valid_set, self.test_set = self._preprocess_mnist(mnist_train, mnist_valid, mnist_test)

    def train_model(self):
        """ Model training. The datasets need to be loaded with load_mnist(). """
        if not self.training_set or not self.valid_set:
            print("Training or validation set not loaded. Call load_mnist() first.")
            return

        self.model.fit(
            x=self.training_set,
            validation_data=self.valid_set,
            epochs=self.max_epochs,
            callbacks=[self.early_stopping]
        )

    def evaluate_model(self):
        """ Evaluate current model. The datasets need to be loaded with load_mnist(). """
        if not self.test_set:
            print("Test set not loaded. Call load_mnist() first.")
            return

        test_loss, test_accuracy = self.model.evaluate(self.test_set)
        print(f'Test loss: {test_loss}')
        print(f'Test accuracy: {test_accuracy}')

    def save_model(self, path):
        """ Save current model. Provide a path along with the .h5 extension. """
        self.model.save(path)

    def load_model(self, path):
        """ Load a pre-trained model. Provide a path along with the .h5 extension. """
        self.model = tf.keras.models.load_model(path)

    def predict(self, image_path=None, image_data=None, streamlit=False):
        """ Take a path to a local image (image_path) or a base64 image data (image_data), convert and make a
        prediction for it. streamlit=True is used to deploy using Streamlit. """
        if image_path:
            image = Image.open(image_path).convert('L')

        elif image_data:
            if streamlit:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert('L')
            else:
                # Decode base64 image data #
                image_bytes = base64.b64decode(image_data.split(',')[1]) # Strip the header from base64
                image = Image.open(io.BytesIO(image_bytes))

                # Add a white background to the image (Flask canvas is transparent) #
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, (0, 0), image)  # Merge the alpha channel with the white background
                image = background.convert('L')

        else:
            raise ValueError("Either image_path or image_data must be provided.")

        # Preprocess image to model training format #
        image_preprocessed = self.preprocess_input(image, streamlit)

        ## Display processed image ##
        #self.display_image_from_np(image_preprocessed)

        # Make prediction #
        predictions = self.model.predict(image_preprocessed)
        return predictions.tolist()[0]

    def preprocess_input(self, image, streamlit):
        """ Preprocess the input image to match the Training Set (MNIST). """
        image = image.resize((28, 28))  # Resize to 28x28
        image = np.array(image) / 255.0  # Normalize pixel values

        # Invert colors if not coming from streamlit canvas #
        if not streamlit:
            image = 1.0 - image

        image = image.reshape(1, 28, 28, 1)  # Reshape for model input
        return image

    def display_image_from_np(self, image):
        """ Display image from np format. """
        image_to_display = (image[0] * 255).astype(np.uint8)  # Rescale to original pixel range (0-255)
        image_to_display = Image.fromarray(image_to_display.reshape(28, 28))  # Reshape and convert to PIL image

        image_to_display.show()