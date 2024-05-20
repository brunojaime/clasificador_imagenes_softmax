from flask import Flask,request,jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import numpy as np
import requests

app = Flask(__name__)

def download_image(url):
    response = requests.get(url)
    img = load_img(BytesIO(response.content),target_size=(224,224))
    img_array = img_to_array(img) 
    resize_image = img_array /255.0
    img_array = np.expand_dims(resize_image, axis=0)
    return img_array

modelo = load_model("vallado_model.h5")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'url' not in data:
        return jsonify({"error":"No se encuentra ninguna url"})
    image_url = data['url']
    image_array = download_image(image_url)
    prediction = modelo.predict(image_array)
    resultado = {"prediction": float(prediction[0][0])}
    return jsonify(resultado)


if __name__ == '__main__':
    print("Iniciando el servidor Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)