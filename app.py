from flask import Flask, request, Response
from json import dumps
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform
# from os.path import join, dirname, abspath
from os import path, remove


model = tf.keras.models.load_model('./model/my_model')

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = 'C:\\Users\Alex\\Desktop\\flask\\input'

@app.route('/')
def hello_world():
    return 'Hello world!'

@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    file.save(path.join(app.config['IMAGE_UPLOADS'], 'file.jpg'))
    file.close()

    image = Image.open('./input/file.jpg')
    image = np.array(image).astype('float32')/255
    image = transform.resize(image, (120, 120, 3))
    image = np.expand_dims(image, axis=0)

    remove('C:\\Users\Alex\\Desktop\\flask\\input\\file.jpg')

    result = model.predict(image)
    print(result)

    resp = Response('casd')
    resp.headers['Access-Control-Allow-Credentials'] = 'true'
    resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'

    return resp