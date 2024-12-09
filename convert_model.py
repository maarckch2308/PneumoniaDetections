import tensorflow as tf

# Cargar el modelo .h5
model = tf.keras.models.load_model('./models/modelo_mlp_radiografia.h5')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido en un archivo .tflite
with open('./models/modelo_mlp_radiografia.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo convertido a TensorFlow Lite y guardado correctamente.")
