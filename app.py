from flask import Flask, request, jsonify, render_template 
import base64
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO



# Cargar el modelo de TensorFlow Lite
MODEL_PATH = './models/modelo_mlp_radiografia.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Configuración de tamaño de imagen
width_shape = 128
height_shape = 128
class_names = ['NORMAL', 'NEUMONIA']

app = Flask(__name__)

def model_predict_tflite(img_data):
    # Convertir la imagen de base64 a un array numpy
    img = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "Error: No se pudo cargar la imagen. Verifica el formato."
    
    # Redimensionar la imagen
    img = cv2.resize(img, (width_shape, height_shape))

    # Normalizar la imagen
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Establecer la entrada del modelo
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], img)

    # Realizar la inferencia
    interpreter.invoke()

    # Obtener los resultados
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    img_data = data['image']
    preds = model_predict_tflite(img_data)

    if isinstance(preds, str):
        return jsonify({'error': preds}), 400

    # Interpretar la predicción
    predicted_class = class_names[1] if preds[0][0] > 0.5 else class_names[0]

    return jsonify({'prediction': f"La predicción es: {predicted_class}"})


@app.route('/')
def home():
    return render_template('index.html')  # Renderiza el archivo index.html


if __name__ == '__main__':
    app.run(debug=False)
