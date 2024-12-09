document.getElementById('file-input').addEventListener('change', () => {
    const file = document.getElementById('file-input').files[0];
    const previewDiv = document.querySelector('.preview');
    const previewImg = document.getElementById('preview-img');

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewDiv.style.display = 'block';
            previewImg.src = e.target.result;  // Mostrar la imagen seleccionada
        };
        reader.readAsDataURL(file);
    } else {
        previewDiv.style.display = 'none';  // Ocultar la vista previa si no hay imagen
    }
});

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
            image: imageBase64
        };

        try {
            const response = await fetch("https://pneumoniadetections.onrender.com/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)  // Enviar datos en formato JSON
            });

            if (response.ok) {
                const result = await response.json();
                resultDiv.textContent = `Predicción: ${result.prediction}`;  // Mostrar la predicción
            } else {
                resultDiv.textContent = "Error en la predicción. Intenta nuevamente.";
            }
        } catch (error) {
            console.error('Error:', error);
            resultDiv.textContent = "Ocurrió un error. Por favor, inténtalo nuevamente.";
        }
    };

    reader.readAsDataURL(file);
});
