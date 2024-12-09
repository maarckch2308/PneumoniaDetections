import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Cargar el modelo preentrenado
MODEL_PATH = 'models/mlp_neumonia_model.h5'  # Cambia esta ruta si es necesario
model = load_model(MODEL_PATH)

# Clases de predicción
class_names = ['NORMAL', 'NEUMONIA']

# Función para realizar la predicción
def model_predict(img_path, model):
    try:
        # Cargar la imagen
        img = cv2.imread(img_path)

        # Verificar si la imagen se cargó correctamente
        if img is None:
            return "Error: No se pudo cargar la imagen. Verifica el formato."

        # Redimensionar la imagen a 150x150 (el tamaño que espera el modelo)
        img = cv2.resize(img, (150, 150))

        # Convertir la imagen a formato adecuado para la red (tensor)
        img = np.asarray(img)

        # Normalización de la imagen
        img = img / 255.0  # Dividimos entre 255 para normalizar los píxeles entre 0 y 1

        # Expandir las dimensiones de la imagen (para que se ajuste al modelo)
        img = np.expand_dims(img, axis=0)

        # Predicción con el modelo
        preds = model.predict(img)

        # Interpretar predicción (para modelo binario)
        if preds[0] > 0.5:
            predicted_class = class_names[1]  # NEUMONIA
        else:
            predicted_class = class_names[0]  # NORMAL
        
        return f"La predicción es: {predicted_class}"
    
    except Exception as e:
        return str(e)

# Ruta a la imagen que deseas probar
img_path = 'dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg' # Cambia esta ruta por la imagen que quieras probar

# Realizar la predicción
result = model_predict(img_path, model)
print(result)
