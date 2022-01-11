from flask import Flask
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform


model = tf.keras.models.load_model('./model/my_model')

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello world!'

@app.route('/process')
def process():
    # imagefile = flask.request.files.get('imagefile', '')
    image = Image.open('./input/00001.jpg')
    image = np.array(image).astype('float32')/255
    image = transform.resize(image, (120, 120, 3))
    image = np.expand_dims(image, axis=0)

    print(model.predict(image))

    return 'Processed'