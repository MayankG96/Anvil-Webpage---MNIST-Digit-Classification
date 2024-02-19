import subprocess
import sys
import pkg_resources
import logging


def install_package_if_not_installed(package):
    try:
        pkg_resources.require(package)
        print(f"{package} is already installed.")
    except pkg_resources.DistributionNotFound:
        print(f"{package} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} has been installed.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {str(e)}")

required_packages = ['tensorflow', 'pandas', 'numpy', 'Pillow','anvil-uplink']

for package in required_packages:
    install_package_if_not_installed(package)

#pip install anvil-uplink
import anvil.server
import pandas as pd
import numpy as np
import tensorflow as tf
import io
from tensorflow.keras.models import load_model
import anvil.server
from PIL import Image, ImageDraw, ImageFont, ImageDraw
import io
import pandas as pd
import anvil.media
from anvil import BlobMedia
import base64

anvil.server.connect("server_VR2VGNCGTIKH5AO7RFI5AT73-JZTZHZFTWJPZRUWX")
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


n = 4
m = 4
block_size = 49
#hidden_dim = 128
#num_layers = 8
#num_heads = 8
#key_dim = hidden_dim//num_heads
#mlp_dim = hidden_dim
#dropout_rate = 0.1
#num_classes = 10

def classify_cnn_image(model, image):
    print("entered classify func")
    #preprocessing
    img_array = np.array(image)
    preprocessed_image = img_array.reshape((1, 28, 28, 1)) 
    #processed_image = preprocess_cnn(image)
    predictions = model.predict(preprocessed_image)
    print(predictions)
    predicted_class = np.argmax(predictions)
    print(predicted_class)
    return predicted_class

def cnn_pred(image):
    model = load_model('cnn_model.h5')
    predicted_class = classify_cnn_image(model, image)
    print(f'Predicted class for the new image: {predicted_class}')

    generated_image_cnn = generate_digit_image(predicted_class)

    return predicted_class, generated_image_cnn

def classify_transformer_image(model, image):

    image = image.to_numpy()
    image = image.reshape(n*m, block_size)
    image = np.expand_dims(image, axis=0)
    pos_feed = np.array([list(range(16))])
    predicted_probabilities = model.predict([image,pos_feed])
    print(predicted_probabilities)
    predicted_class = np.argmax(predicted_probabilities)
    return predicted_class

class ClassToken(tf.keras.layers.Layer):
    def _init_(self):
        super()._init_()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def transformer_pred(image):
    model = load_model('transformer_model.h5', custom_objects={'ClassToken': ClassToken})
    predicted_class = classify_transformer_image(model, image)
    print(f'Predicted class for the new image: {predicted_class}')
    generated_image_trans = generate_digit_image(predicted_class)
    return predicted_class,generated_image_trans

def image_to_blob_media(image):
   # Convert PIL Image to BlobMedia
    print("entered image to blob media")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Convert the image to a byte string
    byte_string = image_bytes.read()

    # Create a BlobMedia instance with the byte string
    blob_media = BlobMedia("image/png", byte_string)
    print("blob media created")
    return blob_media

def generate_digit_image(digit):
    try:
        # Assuming digit is between 0 and 9
        image_path = f'digit_{digit}.png'
        img = Image.open(image_path)
        img_bytes = image_to_blob_media(img)
        return img_bytes
    except FileNotFoundError:
        print(f"Image not found for digit {digit}")
        return None

 
@anvil.server.callable
def handle_csv_upload_test(file):
    # Read the CSV file and get the data
    print("entered main function")
    logging.info("Entered handle_csv_upload --logging")

    status = 1
    status_message = 'Ok'
    
    
    with anvil.media.TempFile(file) as f:
        image = pd.read_csv(f, header=None)

    print(image.shape)

    if image.shape == (28, 28):

        image_array = np.array(image)

        if image_array.max() <= 1:
            image_array *= 255 
        img = Image.fromarray(image_array.astype(np.uint8), 'L')
        uploaded_image = image_to_blob_media(img)
        cnn_prediction,cnn_image = cnn_pred(image)
        trans_prediction, trans_image = transformer_pred(image)
    else:
        status = 0
        status_message = 'Wrong format: The input CSV file should have dimensions 28x28.'
        uploaded_image = None
        cnn_prediction = None
        cnn_image = None
        trans_prediction = None
        trans_image = None

    return  uploaded_image, cnn_prediction, cnn_image, trans_prediction, trans_image, status, status_message

# Run the Uplink server in the background
anvil.server.wait_forever()
