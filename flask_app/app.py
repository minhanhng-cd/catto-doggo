import os
from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import tensorflow as tf

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + '/' + 'catdog_classifier_Xception.h5')

IMAGE_SIZE = 224

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(preprocessed_imgage, (1,IMAGE_SIZE ,IMAGE_SIZE ,3))

    prob = cnn_model.predict(preprocessed_imgage)
    label = "Cat" if prob >= 0.5 else "Dog"
    classified_prob = prob if prob >= 0.5 else 1 - prob
    
    return label, classified_prob

# home page
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/classify', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('home.html')


    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(cnn_model, upload_image_path)

        prob = round((prob[0][0] * 100), 2)

    return render_template('classify.html', image_file_name = file.filename, label = label, prob = prob)

@app.route('/classify/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/model')
def model():
   return render_template('model.html')

@app.route('/about')
def about():
   return render_template('about.html')

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True