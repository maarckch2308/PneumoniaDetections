document.getElementById('uploadForm').onsubmit = async function (event) {
    event.preventDefault(); // Evitar que la página se recargue

    const formData = new FormData(event.target);

    // Enviar el archivo al backend usando Fetch API
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });

    // Obtener el resultado como texto y mostrarlo
    const result = await response.text();
    document.getElementById('result').innerText = result;
};
