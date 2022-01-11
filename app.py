from flask import Flask, request, Response
from json import dumps
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform


model = tf.keras.models.load_model('./model/my_model')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello world!'

@app.route('/process', methods=['POST'])
def process():
    # imagefile = flask.request.files.get('imagefile', '')
    # if 'file' in request.files:
    #     print('yes')
    print(request.files)

    image = Image.open('./input/00001.jpg')
    #image = request.files['file']
    image = np.array(image).astype('float32')/255
    image = transform.resize(image, (120, 120, 3))
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)
    print(result)

    resp = Response('casd')
    resp.headers['Access-Control-Allow-Credentials'] = 'true'
    resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'

    return resp