from __future__ import division, print_function

# Keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

# Flask 
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import os
import numpy as np
import cv2

# Configuración de tamaño de imagen
width_shape = 128
height_shape = 128

# Clases de predicción
class_names = ['NORMAL', 'NEUMONIA']

# Definimos una instancia de Flask
app = Flask(__name__)

# Path del modelo preentrenado
MODEL_PATH = './models/modelo_mlp_radiografia.h5'

# Cargamos el modelo preentrenado
model = load_model(MODEL_PATH)

print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')

# Realizamos la predicción usando la imagen cargada y el modelo
def model_predict(img_path, model):
    try:
        # Cargar la imagen
        img = cv2.imread(img_path)

        # Verificar si la imagen se cargó correctamente
        if img is None:
            return "Error: No se pudo cargar la imagen. Verifica el formato."

        # Redimensionar la imagen a 128x128
        img = cv2.resize(img, (width_shape, height_shape))

        # Convertir la imagen a formato adecuado para la red (tensor)
        img = np.asarray(img)

        # Normalización de la imagen
        img = img / 255.0  # Dividimos entre 255 para normalizar los píxeles entre 0 y 1

        # Expandir las dimensiones de la imagen (para que se ajuste al modelo)
        img = np.expand_dims(img, axis=0)

        # Predicción con el modelo
        preds = model.predict(img)

        return preds
    except Exception as e:
        return str(e)

@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Obtiene el archivo del request
        f = request.files['file']

        # Graba el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Predicción
        preds = model_predict(file_path, model)

        # Verificar si hubo un error
        if isinstance(preds, str):
            return preds  # Devuelve el error como respuesta

        # Interpretar predicción (para modelo binario)
        if preds[0] > 0.5:
            predicted_class = class_names[1]  # NEUMONIA
        else:
            predicted_class = class_names[0]  # NORMAL
        
        print('PREDICCIÓN:', predicted_class)
        
        # Enviamos el resultado de la predicción
        result = f"La predicción es: {predicted_class}"
        return result
    return None

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
