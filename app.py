from __future__ import division, print_function

# Flask
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest

import os
import numpy as np
import tensorflow as tf
import cv2

# Configuración de tamaño de imagen
width_shape = 128
height_shape = 128

# Clases de predicción
class_names = ['NORMAL', 'NEUMONIA']

# Definimos una instancia de Flask
app = Flask(__name__)

# Path del modelo preentrenado
MODEL_PATH = './models/modelo_mlp_radiografia.tflite'

# Cargar el modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')

# Función para realizar la predicción usando TensorFlow Lite
def model_predict_tflite(img_path):
    try:
        # Cargar la imagen
        img = cv2.imread(img_path)

        # Verificar si la imagen se cargó correctamente
        if img is None:
            return "Error: No se pudo cargar la imagen. Verifica el formato."

        # Redimensionar la imagen a 128x128
        img = cv2.resize(img, (width_shape, height_shape))

        # Convertir la imagen a formato adecuado para el modelo
        img = np.asarray(img, dtype=np.float32)

        # Normalizar la imagen
        img = img / 255.0  # Dividimos entre 255 para normalizar los píxeles entre 0 y 1

        # Expandir las dimensiones para que se ajuste al modelo (1, 128, 128, 3)
        img = np.expand_dims(img, axis=0)

        # Establecer la entrada del modelo
        interpreter.set_tensor(input_details[0]['index'], img)

        # Realizar la inferencia
        interpreter.invoke()

        # Obtener los resultados
        output_data = interpreter.get_tensor(output_details[0]['index'])

        return output_data

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

        # Validación del archivo subido
        if not f or not f.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise BadRequest("Por favor sube un archivo de imagen válido (png, jpg, jpeg).")

        # Graba el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Predicción usando el modelo tflite
        preds = model_predict_tflite(file_path)

        # Eliminar archivo después de procesar
        os.remove(file_path)

        # Verificar si hubo un error
        if isinstance(preds, str):
            return preds  # Devuelve el error como respuesta

        # Interpretar predicción (para modelo binario)
        if preds[0][0] > 0.5:
            predicted_class = class_names[1]  # NEUMONIA
        else:
            predicted_class = class_names[0]  # NORMAL
        
        print('PREDICCIÓN:', predicted_class)
        
        # Enviamos el resultado de la predicción
        result = f"La predicción es: {predicted_class}"
        return result
    return None

if __name__ == '__main__':
    app.run(debug=False)
