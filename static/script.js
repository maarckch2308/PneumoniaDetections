document.getElementById('submit-button').addEventListener('click', async () => {
    const file = document.getElementById('file-input').files[0];
    const resultDiv = document.getElementById('result');

    if (!file) {
        resultDiv.textContent = "Por favor, selecciona una imagen antes de enviar.";
        return;
    }

    const reader = new FileReader();
    reader.onload = async function (e) {
        const imageBase64 = e.target.result.split(',')[1];  // Eliminar el prefijo "data:image/jpeg;base64,"

        const data = {
            image: imageBase64  // Crear el objeto con la imagen base64
        };

        try {
            const response = await fetch("https://pneumoniadetections.onrender.com/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)  // Enviar la imagen como JSON
            });

            if (response.ok) {
                const result = await response.json();
                resultDiv.textContent = `Predicción: ${result.prediction}`;  // Mostrar la predicción recibida del servidor
            } else {
                resultDiv.textContent = "Error en la predicción. Intenta nuevamente.";  // Error si la respuesta no es OK
            }
        } catch (error) {
            console.error('Error:', error);
            resultDiv.textContent = "Ocurrió un error. Por favor, inténtalo nuevamente.";  // Error en el fetch
        }
    };

    reader.readAsDataURL(file);  // Convertir la imagen a base64
});
