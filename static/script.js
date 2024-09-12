const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');
const spinner = document.getElementById('spinner');

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    spinner.style.display = 'block'; // Show loading spinner

    const formData = new FormData(form);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    spinner.style.display = 'none'; // Hide loading spinner

    if (response.ok) {
        resultDiv.innerText = `Predicted Emotion: ${result.emotion}`;
    } else {
        resultDiv.innerText = 'Error: Unable to predict emotion. Please try again.';
    }
});
